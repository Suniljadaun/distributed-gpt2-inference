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
- Done: plan, textbook outline, Chapter 0 written, scope report.
- **Next: you do Chapter 0** (`book/ch00_setup.md`) and send me the output of `python src\check_env.py`. Then I write Chapter 1.
