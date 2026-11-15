# Log-Structured Distributed LLM Inference Engine with Dynamic Stage Rebalancing

![tests](https://github.com/Suniljadaun/distributed-gpt2-inference/actions/workflows/tests.yml/badge.svg)

Distributed Systems, Monsoon 2026, Project 55. Sunil Kumar, Roll No. 2025201099.

GPT-2 (distilgpt2) written from scratch in PyTorch and split layer-wise across two worker processes that exchange
tensors over gRPC. A coordinator admits requests through a bounded queue, drives token-by-token generation with a
distributed KV-cache, and appends every token to an append-only log. A crashed worker is restarted and its cache is
rebuilt by replaying the log; stale replies are fenced off by generation numbers. When one stage gets slower, the
coordinator moves layers (and their KV-cache) between the workers without stopping.

The design, the measurements and the discussion are in **[report/Final_Report.pdf](report/Final_Report.pdf)**.

| Result | |
|---|---|
| Output vs single-process inference | identical, token for token, in every configuration |
| KV-cache | 11.0 → 40.8 tokens/s in one process, 8.8 → 33.1 through the pipeline (200-token generations) |
| Per-token latency, single process → pipeline | p50 22.5 → 27.2 ms, p99 27.8 → 35.6 ms |
| Continuous batching | 39.0 → 147.9 tokens/s (3.8×) |
| Worker crash | detected in 0.25 s, restarted in 2.6 s, request finishes with identical output |
| Dynamic rebalancing vs static split | +17% steady, +45% with one slow worker |

## Requirements

- Python 3.14, CPU only (no GPU needed), about 4 GB of RAM
- Windows or Linux (tested on Windows 11 and on Ubuntu in GitHub Actions)
- Internet once, to download the model (≈ 330 MB); afterwards everything runs offline
- Optional: Docker Desktop / Docker Engine with Compose, for the container deployment

## Setup

```bash
git clone https://github.com/Suniljadaun/distributed-gpt2-inference.git
cd distributed-gpt2-inference
python -m venv .venv
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The model is downloaded automatically the first time it is needed. To do it up front (Linux, or if a download fails
with a 404 from Hugging Face's Xet endpoint, set `HF_HUB_DISABLE_XET=1` first):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('distilgpt2', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
```

After that, `HF_HUB_OFFLINE=1` (PowerShell: `$env:HF_HUB_OFFLINE=1`) skips the online checks.

## Run

### Tests (106, about 4 minutes)
```bash
python -m pytest -q
```

### The engine, and a prompt of your own
```bash
# terminal 1: coordinator on port 50060; it starts and supervises worker A and worker B itself
python -m src.coordinator.server --port 50060 --slots 4 --log logs/token_log.jsonl

# terminal 2
python -m src.client "The capital of France is" --tokens 30
python -m src.client "Once upon a time" --tokens 30 --request-id demo-1   # same id again: answered from the log
```

Coordinator options: `--slots` (requests in flight), `--queue-size` (bounded queue), `--max-batch N` (continuous
batching; not combined with `--log`), `--log` (token log, enables replay after a worker crash), `--rebalance` (move
layers between the workers when one stage is the bottleneck). With `--log` and `--rebalance` every layout is written
to the log, each token records the layout epoch it was computed under, and a restarted coordinator brings its workers
up in the last logged layout.

### Live demo (about 10 minutes, all goals on one screen)
```bash
python -m playground.ch18.demo            # press Enter between the steps
python -m playground.ch18.demo --auto     # no pauses, about 2 minutes
```
Steps: correctness against single-process inference → steady load with the rebalancer → worker B killed in the middle
of a request → worker A slowed down (layers move away, then back) → duplicate requests. Close other programs
first: the workers are pinned to cores 0–3 and the demo to cores 4–5.

### Benchmarks (about 11 minutes)
```bash
python -m src.bench.run_all                  # all experiments → results/bench/*.json, results/charts/, results/summary.md
python -m src.bench.run_all --quick          # shorter runs
python -m src.bench.run_all --only wire,load # some of them
python -m src.bench.run_all --from-saved     # redraw charts and summary from the saved data, run nothing
```

### Docker Compose (one container per process)
```bash
docker compose up -d --build                 # image with the model inside (~2.6 GB); worker-a, worker-b, coordinator
python -m src.client "The capital of France is" --tokens 30      # the coordinator is published on port 50060
python -m playground.ch17.compose_check      # correctness + kill a worker inside its container, check recovery
docker compose logs coordinator
docker compose down -v
```
Docker's restart policy restarts a crashed worker container; the coordinator recognises the new incarnation by its
generation and replays the log. The rebalancer is not enabled in this mode (Docker restarts a worker with its
original layer range).

### The report
```bash
pip install markdown          # already in requirements.txt
python report/build_pdf.py    # report/Final_Report_draft.md → report/Final_Report.pdf (uses Microsoft Edge, Windows)
```

## Continuous integration

`.github/workflows/tests.yml` runs on every push:
- **test**: installs everything on Ubuntu with Python 3.14, runs the 106 tests and a short benchmark (charts kept as
  a downloadable artifact);
- **compose**: builds the image, starts the three containers and runs `playground/ch17/compose_check.py`.

## Project layout

| Path | Contents |
|---|---|
| `src/model/` | GPT-2 layers, weight loader, tokenizer, KV-cache, generation (single process) |
| `src/proto/` | `inference.proto`, generated gRPC code, tensor codec |
| `src/worker/` | a pipeline stage (range of layers; layers can be added and dropped) and its gRPC server |
| `src/coordinator/` | pipeline client, schedulers, coordinator server, supervisor, remote health checks, token log, replay, rebalancer |
| `src/bench/` | experiments (including per-token latency, the pipeline bubble and the KV-cache migrated per layer move), `run_all`, NumPy analysis, Matplotlib and SVG charts |
| `src/client.py` | command-line client |
| `tests/` | 106 tests (`test_chNN.py`, one file per stage of the build) |
| `playground/` | small experiments used while building; `ch18/demo.py` is the live demo, `ch17/compose_check.py` the container check |
| `results/` | `reference.json` (single-process outputs for the correctness tests), benchmark data, charts, `summary.md` |
| `report/` | the final report (Markdown source, PDF, figure, build script) and the scope report |
| `Dockerfile`, `docker-compose.yml` | container deployment |

## Troubleshooting

- **`address already in use`**: another run is still going, or another program uses the port. The local coordinator
  uses 50060 and starts its workers on 50061–50062; the demo uses 50501–50503.
- **Windows: `grpc_tools.protoc` is blocked** (Smart App Control): the generated code is committed, so this only
  matters if you change `inference.proto`. Regenerate it under WSL:
  `python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. src/proto/inference.proto`.
- **Numbers differ from the report**: they depend on the machine and its load. Plug in the laptop and close other
  programs; differences of about ±5% between runs are normal.
