# Log-Structured Distributed LLM Inference Engine with Dynamic Stage Rebalancing

**Final Report: Distributed Systems, Monsoon 2026, Project 55**
Sunil Kumar K (2025201099), individual project

---

## 1. Introduction

A large language model is often split across several machines, either because it does not fit on one, or to
spread the work. Splitting turns one program into a distributed system. Activations must cross the network for every
generated token, any worker can crash mid-request, and the parts can run at different speeds. The slowest one then
sets the speed of the whole pipeline.

This project builds an inference engine for GPT-2 (distilgpt2, 6 transformer blocks) from scratch in PyTorch,
split layer-wise across two worker processes that exchange tensors over gRPC. A coordinator admits requests through a
bounded queue, drives token-by-token generation with a distributed KV-cache, and appends every committed token to an
**append-only log**. When a worker crashes, it is restarted and its cache is rebuilt by **replaying the log**. Stale
replies are **fenced** off by generation numbers and step ids. When one stage becomes slower, the coordinator
**rebalances**: it moves boundary layers between workers, with their KV-cache slices, under a new epoch, without
stopping service.

**Headline results** (one 12-thread laptop, CPU only, `results/summary.md`):
- Output is **token-for-token identical** to single-process inference in every configuration: cache, batching,
  after a worker crash, and across layer moves.
- KV-cache: **13.1 → 44.9 tokens/s** for 200-token generations (3.4×).
- Continuous batching: **39.3 → 145.5 tokens/s** (3.7×) with identical outputs.
- Worker crash: detected in **0.26 s**, restarted in **2.6 s**. The interrupted request completes with **identical
  output**; replaying 100 tokens takes **92 ms** vs 2.7 s to regenerate.
- Dynamic rebalancing vs the best static split: **+7%** at steady state, **+39%** when one worker is slowed down.

## 2. Goals and status

| Goal (from the scope report) | Status | Evidence |
|---|---|---|
| G1 Output identical to single-process inference | Done | `tests/test_ch08.py` (and later chapters' tests); `reference` experiment: 3/3 |
| G2 Distributed KV-cache, ordering by step ids | Done | `src/worker/server.py` (`RequestState`); `test_ch09.py`; `kv_cache` experiment |
| G3 Bounded queue, timeouts, retries with backoff | Done | `src/coordinator/scheduler.py`, `pipeline.py`; `test_ch10.py`; `load` experiment |
| G4 Micro-batching, pipeline bubble | Done (continuous batching) | `ForwardBatch`, `BatchingScheduler`; `test_ch11.py`; `throughput` experiment |
| G5 Log-based crash recovery with fencing | Done | `supervisor.py`, `token_log.py`, `replay.py`; `test_ch12–14.py`; `recovery` experiment |
| G6 Dynamic stage rebalancing | Done | `rebalancer.py`, layout RPCs; `test_ch15.py`; `rebalance` experiment |

88 automated tests (`python -m pytest -q`). All measurements are reproduced by `python -m src.bench.run_all`.

## 3. Architecture

```
            Generate(request_id, prompt)            Forward / ForwardBatch (hidden state, step_id, generation, epoch)
 client ───────────────────────────► COORDINATOR ─────────────────────► WORKER A ──(hidden)──┐
        ◄─────────────────────────── │ bounded queue + scheduler        embeddings,         │
            tokens                   │ token log (append-only, CRC)     blocks 0..k-1       │
                                     │ supervisor (heartbeats, restart) KV-cache per request│
                                     │ rebalancer (stage-time model)                        ▼
                                     ◄────────────────────────────────── WORKER B ◄──────────┘
                                          next-token logits              blocks k..5, ln_f, output layer
```
*(Star topology: the coordinator calls A, then B, for each step. Replace with a drawn figure.)*

**Life of one token.** The scheduler picks a request (or a batch). The coordinator sends the last token id with
`step_id`, `start_pos`, the worker's expected `generation` and the layout `epoch` to A. A embeds it, runs its blocks
using that request's KV-cache, and returns the hidden state (3,072 bytes). The coordinator forwards it to B, which runs
its blocks, the final LayerNorm and the tied output layer, and returns logits. The coordinator picks the next token
(greedy), **appends it to the log and flushes**, and only then uses it for the next step or returns it to the client.

## 4. Design

### 4.1 Model and partitioning (G1)
GPT-2 is implemented by hand (attention, MLP, LayerNorm, blocks) and loads the pretrained distilgpt2 weights from
safetensors. Each worker reads **only its own tensors**. A stage is a contiguous range of blocks; the first stage also
owns the embeddings, the last the final LayerNorm and output layer. Tensors travel as protobuf messages with shape,
dtype and raw bytes. The protobuf overhead is 11–16 bytes; JSON was 5× larger and up to 100× slower to encode
(`wire` experiment). Correctness is checked against a stored single-process reference with greedy decoding, so
outputs must match exactly.

### 4.2 Distributed KV-cache and step ordering (G2)
Each worker keeps keys/values for its own layers, per request. Every `Forward` carries `step_id` and `start_pos`. A
worker accepts only the next step. An exact duplicate of the last step (a retried RPC) returns the stored reply
instead of appending K/V twice. `Release` frees a finished request's cache. *Rejected alternative:* a central cache at
the coordinator, which would ship 36 KB per token per request across the network.

### 4.3 Admission, deadlines, retries (G3)
Requests enter a **bounded queue**. When it is full the client gets `RESOURCE_EXHAUSTED` immediately. Each request's
deadline is propagated, so each worker call gets only the remaining time. Failed calls are retried with exponential
backoff and full jitter, never past the deadline. *Rejected alternative:* an unbounded queue. Under overload every
request waits until it expires (see §5.3).

### 4.4 Batching (G4)
With one request in flight, A and B alternate and each idles half the time (the pipeline bubble). Several requests in
flight (slots) fill the bubble. **Continuous batching** (`ForwardBatch`) goes further: each decode step processes up to
B requests in one matrix multiply, and requests join and leave the batch at every step. Two batch groups keep both
stages busy. Worker threads are pinned (`--threads`) to avoid CPU oversubscription.

### 4.5 Failure detection and restart (G5)
A supervisor starts the workers as processes and pings them (`Ping`, every 0.25 s). A worker is declared dead when
its process exits or after 3 missed heartbeats (a hang). It is restarted with **generation + 1**. gRPC channels use
fast-reconnect options so the coordinator notices the new process quickly.

*Two races found late, while rehearsing the demo, and fixed:*
1. The coordinator's failed RPC could reach the supervisor's "all workers up?" check before the next heartbeat had
   noticed the crash, so it replayed against the dead port. The check now inspects the worker processes itself.
2. "The supervisor sees the new worker" does not mean "the coordinator's own connection is back". The coordinator's
   channel could still be in reconnect backoff, so replay now waits for its own channels too.

Each wasted attempt used one of 3 allowed replays, so under load a request could be lost.

### 4.6 Append-only token log and replay (G5)
Every committed token is one line in an append-only file: a JSON record (request id, step, token, LSN) and its CRC32.
It is **flushed before the token is used** (write-ahead). After a worker restart the new process has an empty cache.
The coordinator rebuilds it by sending `prompt + committed tokens[:-1]` as **one prefill**, then continues. Each attempt
uses a fresh worker-side request id, so state from a dead attempt cannot mix in. On startup a CRC failure in the
**last** line is treated as a torn write and truncated; one in the **middle** is corruption and the log refuses to
open. *Choice:* flush (10.7 µs/token) rather than fsync (0.89 ms/token). The threat model is a process crash, not
power loss.

### 4.7 Fencing and idempotency (G5)
A worker declared dead may still be alive (a hang) and reply later: a *zombie*. Every request carries the expected
`generation`; replies from any other generation are rejected (`StaleReply`). A retried step is recognised by its step
id and a CRC32 digest of its input. Same digest: the stored reply is returned. Different digest: the request is
rejected (`ALREADY_EXISTS`) instead of silently computing a different answer. At the coordinator, concurrent
duplicates of a request id are **coalesced** into one computation, and a retry after completion is answered from the
log. The result is at-least-once delivery with idempotent processing, i.e. effectively-once.

### 4.8 Dynamic stage rebalancing (G6)
**Monitoring.** Every reply reports the time the worker spent in its blocks. The rebalancer fits, per stage,
`stage_time = fixed + blocks × per_block` over a sliding window. For each possible single-layer move at the boundary it
**predicts** the new bottleneck stage time. It moves only if the predicted drop exceeds 5%, then waits for a cooldown.
With **hysteresis**, a move that undoes the previous one needs a 10% predicted drop. Without it, two nearly equal
layouts (4/2 and 5/1, within ~1 ms) made the rebalancer move layer 4 back and forth four times in one demo rehearsal.
Each move was correctly predicted, but each gain was within the noise.

**Protocol.** (1) `PrepareLayer`: the receiver loads the layer's weights *while traffic continues*. (2) The layout
gate, a readers-writer lock with writer priority, pauses new steps. (3) `ExportKV` / `ImportKV` copy the layer's
KV-cache slices for all active requests. (4) `CommitLayout` on the **receiver first**, then the sender, with epoch + 1.
(5) Release the gate. Any failure before the receiver commits leads to `AbortLayout`, and the old layout stays.
After it, the sender's commit is retried, and steps carrying the old epoch are fenced. Measured pause: 6–31 ms.

**Rejected designs** (all implemented and measured, Chapter 15):
- *Ratio thermostat* (move when A/B time ratio exceeds a threshold): overshot to a 5/1 split and oscillated, because
  B's output layer is a fixed cost that does not move with layers.
- *Workers sharing all CPU cores:* no gain, since moving work to B slowed B down by the same amount. Rebalancing only
  helps with separate resources, so workers are pinned to their own cores (`--cpus`).
- *Concurrent requests inside a stage:* stage times became too noisy to act on (15 flapping moves). Stages process one
  step at a time (`--serial`).

## 5. Evaluation

**Setup.** AMD Ryzen laptop (AMD64 Family 23 Model 104, 12 logical CPUs), Windows 11, Python 3.14.7, PyTorch 2.14.1
CPU, gRPC 1.84.0, distilgpt2, greedy decoding, all processes on localhost. Raw data with environment and git commit:
`results/bench/*.json`. Charts: `results/charts/*.svg`. *(Update these numbers from the final `run_all` before
submitting.)*

### 5.1 Correctness
3 of 3 reference prompts reproduced token for token by 2 workers. Every throughput configuration, the crash
experiment, and all rebalancing runs also compare their outputs with the reference: **0 differing tokens**.

### 5.2 KV-cache and the split point
| | tokens/s (200 tokens) | last step |
|---|---|---|
| no cache | 13.1 | 125.8 ms |
| KV-cache | 44.9 | 23.1 ms |

| split (A/B blocks) | A ms | B ms | overhead ms | step ms |
|---|---|---|---|---|
| 1/5 | 2.75 | 20.45 | 2.96 | 26.26 |
| 3/3 | 7.54 | 15.64 | 2.98 | 26.26 |
| 4/2 | 10.39 | 13.47 | 3.05 | 27.06 |
| 5/1 | 12.48 | 10.73 | 3.09 | 26.42 |

With one request the step time is the *sum* of the stages, so the split barely matters (26–27 ms). The split matters
when stages overlap (§5.6). B is always heavier than its block count suggests: it owns the output layer
(768 × 50,257). Network + serialisation overhead is ~3 ms per step for two hops.

### 5.3 Backpressure (G3)
| queue | arrival rate (req/s) | sent | ok | rejected | expired | p99 (s) | L sampled | L = λW |
|---|---|---|---|---|---|---|---|---|
| bounded 8 | 1 | 8 | 8 | 0 | 0 | 0.44 | 0.19 | 0.19 |
| bounded 8 | 3 | 37 | 37 | 0 | 0 | 0.74 | 1.16 | 1.17 |
| bounded 8 | 6 | 61 | 49 | 12 | 0 | 2.54 | 5.80 | 5.82 |
| unbounded | 6 | 71 | 36 | 0 | 35 | 3.01 | 11.43 | 11.48 |

Above capacity, the bounded queue rejects 12 requests immediately and completes 49. The unbounded queue accepts all
of them and lets half expire after waiting: fewer completions and higher latency. Little's Law holds within 1%.

### 5.4 Batching (G4)
| configuration | tokens/s | mean latency (s) | outputs = reference |
|---|---|---|---|
| 1 at a time | 39.3 | 6.36 | yes |
| 4 slots | 67.7 | 4.26 | yes |
| batch 8 | 81.7 | 3.91 | yes |
| batch 24 | 138.2 | 3.44 | yes |
| 2 groups × batch 12 | 145.5 | 3.25 | yes |

Throughput rises 3.7× and latency *falls*: requests no longer wait for each other to finish. Two groups beat one
batch of 24 by filling the pipeline bubble.

### 5.5 Recovery (G5)
| heartbeat | crash detect | restart | hang detect | 100-token request | same, worker B killed | replays | identical |
|---|---|---|---|---|---|---|---|
| 0.25 s | 0.26 s | 2.56 s | 1.59 s | 3.02 s | 5.82 s | 1 | yes |

| tokens to recover | replay | regenerate |
|---|---|---|
| 25 | 56.5 ms | 674 ms |
| 100 | 92.2 ms | 2,696 ms |
| 400 | 307.2 ms | 10,783 ms |

A crash costs the request ~2.8 s, almost all of it the restart (process start + weight load). Replay itself is
12–35× faster than regenerating, because it is one batched prefill rather than N sequential steps.

### 5.6 Rebalancing (G6)
Each worker on its own cores, 4 requests in flight, 48 s run. At t = 28 s worker A is slowed by 8 ms per block.

| period | static 3/3 (tok/s) | dynamic (tok/s) | dynamic layout |
|---|---|---|---|
| steady (t = 4–24 s, mean) | 49.2 | 52.8 (+7%) | 4/2 |
| A slowed (t = 36–44 s, mean) | 27.5 | 38.3 (+39%) | 2/4 |

The rebalancer moved one layer to A at start-up (B carries the output layer), then two layers back to B after the
slowdown. Each predicted bottleneck time was confirmed by the next measurement within ~1 ms. In the live demo,
removing the slowdown returned the system to the original 4/2 layout and throughput. Pause per move: 6–31 ms.

### 5.7 Threats to validity
All processes share one laptop and talk over localhost, so network costs are far below a real cluster's.
CPU frequency scaling and background programs add run-to-run variation of roughly ±5%; only differences well
above that are claimed. The slowdown in §5.6 is injected (a fixed sleep per block), not a real co-located workload.

## 6. Deviations from the scope report and limitations

**Not done (promised as tooling in the scope report):**
- **Docker Compose, Prometheus/Grafana, Redis, GitHub Actions CI:** not used. The engine runs as local processes
  started by the supervisor. Metrics are collected by the coordinator's `Stats` RPC and the benchmark harness, and the log is a
  local file. These were optional extras in the plan and were cut for time in favour of the core goals.
- **Matplotlib:** charts are written as SVG by a small built-in writer (`src/bench/svg.py`), because Windows Smart
  App Control blocked parts of the scientific Python toolchain on this machine.

**Known limitations:**
- Token logging and replay work with the slot scheduler, **not with continuous batching**.
- Layout **epochs are not recorded in the log**. A coordinator crash during a layer move is not recovered
  automatically.
- The coordinator is a **single point of failure**. Its log is local and not replicated.
- Two workers only, star topology, distilgpt2. GPT-2 (12 layers) and three workers are supported by the design but
  not measured.
- A few timing-sensitive tests can fail under heavy machine load and pass on rerun.

## 7. Conclusion

The engine meets all six goals of the scope report with outputs identical to single-process inference throughout.
The main lessons:
- **Write-ahead logging turns worker state into a cache.** Losing it costs a replay (~0.1 s), not the request.
- **Fencing by generation and epoch** is what makes restarts and layer moves safe while old messages are still in flight.
- **Rebalancing needs a model, not a threshold.** It also needs isolated resources and quiet measurements to
  work at all.

Next steps would be a chain topology (A → B directly), a replicated coordinator log (Raft), epochs in the log, and
logging combined with continuous batching.

## Appendix A: How to run
```powershell
cd D:\Assignments\Distributed_Systems\final
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE=1
python -m pytest -q                        # 88 tests
python -m src.bench.run_all                # all experiments, ~7 min → results/
python -m playground.ch18.demo             # live demo (Enter between steps; --auto for no pauses)
```

## Appendix B: Code map
| Path | Contents |
|---|---|
| `src/model/` | GPT-2 layers, weights loader, tokenizer, KV-cache, generation |
| `src/proto/` | `inference.proto`, generated code, tensor codec |
| `src/worker/` | Stage (layer range, add/drop block), StageWorker gRPC server |
| `src/coordinator/` | pipeline client, scheduler(s), coordinator server, supervisor, token log, replay, rebalancer |
| `src/bench/` | experiments, SVG charts, `run_all` |
| `tests/` | one test file per chapter, 88 tests |

*(If the course requires it, add a statement on the use of AI assistance here.)*
