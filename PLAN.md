# Build Plan — Log-Structured Distributed LLM Inference Engine with Dynamic Stage Rebalancing

**Assignment:** Project 55 (student-proposed), Sunil Kumar K, 2025201099, individual.
**Scope report:** `report/Project_Scope_Report.pdf` (due 17 Oct 2026). This plan delivers exactly what it promises.

## What we're building (one paragraph)
A GPT-2 model (distilgpt2, 6 layers) split layer-wise across **two worker processes** that pass
activations over **gRPC**. A **coordinator** admits requests through a bounded queue, drives
token-by-token generation, and appends every committed token to an **append-only log**. A
crashed worker is restarted and its KV-cache rebuilt by **replaying the log**; stale replies are
**fenced** off by generation number and `(request_id, step_id)`. When one stage is slower, the
coordinator **rebalances**: it moves boundary layers between workers under a new epoch.

## How we work
- I write each chapter of the textbook in `book/` (theory → design → code with explanations → run → measure → exercises → interview Qs). Large chapters come **one section at a time**.
- **You type all the code yourself** into `src/`, run it, and send me the output.
- I review your code, run the chapter's test, and explain errors with hints, not by rewriting your code.
- Every chapter opens with a "Files you create" box (exact paths) and gives exact PowerShell commands, run from `D:\Assignments\Distributed_Systems\final`.

## Goals from the scope report, mapped to chapters

| Scope-report goal | Chapters |
|---|---|
| G1 Output identical to single-process inference | 4 (reference), 8 (test, re-run every chapter) |
| G2 Distributed KV-cache + ordering by step IDs | 9 |
| G3 Bounded queue, timeouts, retries with backoff | 10 |
| G4 Micro-batching, pipeline bubble | 11 |
| G5 Log-based crash recovery with fencing | 12, 13, 14 |
| G6 Dynamic stage rebalancing | 15 |
| Measurements, baselines, report, demo | 16, 18 (measuring starts in Ch 8) |

## The chapters (= build stages)

| Part | Ch | What you build | What you see working |
|---|---|---|---|
| I Foundations | 0 | Setup: `.venv`, packages, `check_env.py`, git | "environment ready" |
| | 1 | Python for systems: classes, dataclasses, threads, `queue`, `subprocess` | Small demos of each |
| | 2 | Tensors and PyTorch on CPU | Shapes, matmul, timing |
| | 3 | Transformer pieces: tokenizer, embeddings, attention, MLP, LayerNorm | Each piece tested on its own |
| | 4 | Our own GPT-2 with distilgpt2 weights + `generate()` | Readable text; **reference output** |
| II Communication | 5 | What makes a system distributed | Two processes exchanging messages |
| | 6 | gRPC: `.proto`, server, client, deadlines, status codes | Client/server in two terminals |
| | 7 | Sending tensors over gRPC | Tensor round-trip, bytes measured |
| III Engine | 8 | Split GPT-2 across worker A and worker B | Same text as Ch 4, from 2 processes |
| | 9 | KV-cache per request + step-ID ordering | Speed-up vs no cache, same output |
| | 10 | Bounded queue, deadlines, retries with backoff | Fast rejections under overload |
| | 11 | Micro-batching | Throughput up with requests in flight |
| IV Failure | 12 | Heartbeats, detect crash, restart worker | Crash detected and worker back |
| | 13 | Append-only token log + replay | KV-cache rebuilt after a crash, same text |
| | 14 | Generation fencing + dedup | Late/duplicate replies rejected |
| V Rebalancing | 15 | **Dynamic stage rebalancing**: monitor stage times, prepare/commit layer move under a new epoch, migrate KV slices | Slow one worker → layers move → throughput recovers, output unchanged |
| VI Proof | 16 | Measurement script + charts | Latency, throughput, bytes, recovery, rebalance charts |
| | 17 | Extras (only if time): Docker Compose, FastAPI, Prometheus/Grafana, Redis Streams, CI | Production-shaped demo |
| | 18 | Final report, demo script, interview prep | Submission-ready |

## Timeline (the 6-week plan from the scope report)

| Week | Chapters |
|---|---|
| 1 | 0–4 (foundations, working GPT-2) |
| 2 | 5–8 (gRPC, two-worker pipeline) |
| 3 | 9–11 (KV-cache, backpressure, micro-batching) |
| 4 | 12–14 (crash recovery) |
| 5 | 15 (rebalancing) |
| 6 | 16–18 (measurements, report, demo) |

If time runs short, cut in this order: Ch 17 extras, then Ch 11 micro-batching (mark it partial).
Never cut 13–15. Those are what the project title promises.

## Stack
Python 3.14 · PyTorch 2.14 CPU · transformers (weights only) · gRPC 1.84 + protobuf · pytest · NumPy/Matplotlib.
No GPU, cluster or RCE needed.

## Code layout (grows chapter by chapter)
```
final/
├── PLAN.md                this file
├── book/                  the textbook (00_CONTENTS.md, ch00_setup.md, ...)
├── report/                scope report, later the final report
├── src/
│   ├── check_env.py       Ch 0
│   ├── model/             Ch 2–4   GPT-2, weights, generate
│   ├── proto/             Ch 6–7   .proto + generated code
│   ├── worker/            Ch 8+    stage server (runs as A and B)
│   ├── coordinator/       Ch 8+    queue, scheduler, log, recovery, rebalancer
│   └── bench/             Ch 16    experiments
├── tests/                 one test file per chapter from Ch 3
├── results/               charts and tables
└── requirements.txt
```

## Where we are
- Done: plan, textbook outline, scope report, Chapters 0–16 and 18 written. ALL scope goals G1–G6 implemented and measured. Ch 17 (extras) skipped; declared in the report's deviations.
- Your side: Ch 0–16 done and committed. Ch 10–18 code applied by Claude at your request.
- **Next: you** rehearse `python -m playground.ch18.demo --auto`, commit Ch 18, finish `report/Final_Report_draft.md` (rerun `run_all` on the final commit, update §5, add figure + AI-use statement if required, convert to PDF), rehearse the demo with the talking script (book 18.5) and the interview bank (18.9).
- Ch 18: `--block-delay-file` (slowdown only while a file exists) → `playground/ch18/demo.py` (G1–G6 in 5 steps). Rehearsal: 55 → 22.5 tok/s when A slowed, 43 after 3 moves, back to 4/2 and 55 when the slowdown ended; crash mid-request: replays 1, identical. **Two real bugs found by the rehearsal and fixed:** (1) `Supervisor.wait_all_up` said "all up" in the ~0.25 s before the next heartbeat noticed a crash → replays retried at the dead port; (2) after the restart the coordinator's channel was still in reconnect backoff → `replay.py` now waits `pipe.wait_ready()`. Both burned replays (max 3) and could fail a request under load. Suite: 86 tests.
- Ch 16: bench writes results/bench/*.json (with env + commit), results/charts/*.svg (no matplotlib: SVG writer, avoids Smart App Control), results/summary.md. Quick-run headline: G1 3/3; cache 20.8→45.8 tok/s; batching 141.6 tok/s (3.7×); Little within 1%; crash detect 0.26 s + restart 2.6 s; rebalance +16%/+43%. Suite: 83 tests (1 of 3 full runs had 2 timing flakes).
- Ch 15 numbers (serial workers, A cores 0-1, B 2-3, coordinator 4-5, 4 in flight): static 3/3 46.1 → dynamic 4/2 49.4 tok/s (+7%); A +8 ms/block: static 26.4 vs dynamic 2/4 34.6 tok/s (+31%); predictions within ~1 ms; move pause 8–14 ms; 0 changed tokens. Failed designs: ratio thermostat (overshoot to 5/1), shared CPU (no gain), concurrent stages (15 flapping moves). Suite: 79 tests.
- Ch 14 numbers: fingerprint 1.9 µs/step (3 KB), fencing cost within noise; 5 duplicate requests → 1 computation (1.36 s vs 6.86 s). Suite: 73 tests.
- Ch 13 numbers: commit 10.7 µs/token (flush) vs 0.89 ms (fsync), 66 B/token; replay 12–33× faster than regenerating; worker crash in a 100-token request: 3.43 → 6.19 s, identical output, 0 mismatches. Suite: 67 tests.
- Ch 12 numbers: crash detect ≤ 1 interval (0.1/0.25/0.5 s), hang detect ≈ 3×(T+τ) (0.65/1.58/3.07 s), restart ≈ 2.5 s; no false positives even at 2 ms ping timeout (ping RTT ≈ 0.7 ms). Suite: 61 tests.
- Ch 11 numbers (3 threads/worker): 1-at-a-time 34 tok/s → 2 slots 1.42× → batch 24 3.48× → 2 groups × batch 12 3.63× (124.6 tok/s); all outputs = reference; --threads 3 +34% at high load; head matmul batch 4 ≈ 4× batch 1 (kernel quirk).
- Ch 10 numbers: 10-token request ≈ 0.29 s → capacity ≈ 3.4 req/s (1 slot); Little's Law within 1% (L 0.29…13.2); at λ=6: bounded Q=8 → 57 ok / 36 rejected / 0 expired, unbounded → 36 ok / 53 expired. Port 50127 is taken by VS Code on this laptop.
- Tooling: Windows Smart App Control blocks grpc_tools protoc → regenerate with `wsl -d Ubuntu -- bash -lc "cd /mnt/d/Assignments/Distributed_Systems/final && ~/protoc-venv/bin/python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. src/proto/inference.proto"`.
- Ch 9 numbers: cached step flat ≈ 23 ms (12.3 → 43.5 tok/s over 200 tokens); distributed 49 → 27 ms/step (A 8 + B 16 + overhead 3.0); overhead is per call (no-compute Forward ≈ 0.8 ms); KV 36,864 B/token; best split moved 4/2 → 5/1 (1.17×); cached logits differ ≤ 8e-5, tokens identical.
- Ch 8 baseline (no cache, star, split 3): step ≈ 49 ms = A 19 + B 27 + overhead 3.4; block ≈ 6 ms, head ≈ 8.5 ms; best static split 4/2 (1.21×). Workers 242.6 + 239.5 MB (wte duplicated).
- Ch 7 facts: 1-token message 3,083 B, codec 0.027 ms; protobuf path ≈ 1.7 ms/MB (5 copies), torch.save faster above ~1 MB; split at layer 3 across the wire is bit-identical.
- Ch 6 facts for later: gRPC unary p50 ≈ 440 µs (6× raw socket); default max message 4 MB (matters for Ch 15 KV migration); reconnect to dead server fails after ≈ 2 s.
- Ch 5 facts for later: localhost RTT p50 ≈ 70 µs; refused connect takes ≈ 2 s on Windows (matters for Ch 12 restarts).
- Baseline numbers for Ch 9 (no KV-cache): decode step 24 ms at seq 1 → 120 ms at seq 204; 12 tokens/s over 200 tokens.
