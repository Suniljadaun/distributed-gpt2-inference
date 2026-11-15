"""
experiments.py - Chapter 16
Every measurement the report needs, as functions that RETURN data (no printing, no charts).
Most reuse the experiment code written in earlier chapters' playground scripts.
"""
import json
import statistics
import tempfile
import threading
import time
from pathlib import Path

import grpc
import torch

from src.coordinator.cluster import WorkerSpec, running_cluster, supervised_cluster, two_stage_layout
from src.coordinator.pipeline import generate
from src.coordinator.replay import generate_logged
from src.coordinator.server import start_coordinator
from src.coordinator.token_log import TokenLog
from src.model.generate import cached_step, next_token
from src.model.kv_cache import KVCache
from src.model.weights import load_pretrained
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE

PROMPT = [464, 3139, 286, 4881, 318]          # "The capital of France is"


def kv_cache(steps: int) -> dict:
    """Ch 9: time of every decode step, with and without the KV-cache, in one process."""
    model = load_pretrained()
    rows = {}
    for name, use_cache in (("no cache", False), ("KV-cache", True)):
        ids, cache, feed, times = list(PROMPT), KVCache(), list(PROMPT), []
        for _ in range(steps):
            t0 = time.perf_counter()
            tok = cached_step(model, cache, feed) if use_cache else next_token(model, ids)
            times.append((time.perf_counter() - t0) * 1000)
            ids.append(tok)
            feed = [tok]
        rows[name] = times
    return {"step_ms": rows,
            "tokens_per_s": {k: round(1000 * len(v) / sum(v), 1) for k, v in rows.items()}}


def split_sweep(base_port: int) -> list[dict]:
    """Ch 8-9: per-stage compute and overhead per decode step for every split point."""
    out = []
    for split in range(1, 6):
        with running_cluster(two_stage_layout(base_port=base_port + 2 * split, split=split)) as pipe:
            _, timings = generate(pipe, f"sweep-{split}", PROMPT, 20)
        decode = timings[1:]                                       # skip the prefill step
        out.append({"split": f"{split}/{6 - split}",
                    "A_ms": round(statistics.median(t.compute_ms[0] for t in decode), 2),
                    "B_ms": round(statistics.median(t.compute_ms[1] for t in decode), 2),
                    "overhead_ms": round(statistics.median(t.overhead_ms for t in decode), 2),
                    "total_ms": round(statistics.median(t.total_ms for t in decode), 2)})
    return out


def bubble_model(m: int, p: int = 2) -> float:
    """Textbook pipeline bubble: idle fraction of p equal stages processing a burst of m micro-batches."""
    return (p - 1) / (m + p - 1)


def bubble(base_port: int, sizes=(1, 2, 4, 8, 16), repeats: int = 5, prompt_len: int = 32) -> list[dict]:
    """Ch 18 (G4): measured idle time of each stage for a burst of m independent micro-batches (prefills),
    vs the bubble model. Each worker is a true stage (--serial: one step at a time) on cores of its own.
    Idle fraction of a stage = 1 - (time it spent computing) / (burst duration, first send to last reply)."""
    prompt = (PROMPT * (prompt_len // len(PROMPT) + 1))[:prompt_len]
    a, b = two_stage_layout(base_port=base_port)
    layout = [WorkerSpec(a.worker_id, a.port, a.first, a.last, embed=True,
                         extra_args=("--serial", "--threads", "2", "--cpus", "0,1")),
              WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True,
                         extra_args=("--serial", "--threads", "2", "--cpus", "2,3"))]
    out = []
    with running_cluster(layout) as pipe:
        for k in range(3):                                          # warm-up
            pipe.step(f"warm-{k}", 0, prompt, 0)
            pipe.release(f"warm-{k}")
        single = None
        for m in sizes:
            runs = []
            for r in range(repeats):
                rids = [f"bubble-{m}-{r}-{i}" for i in range(m)]
                timings = [None] * m
                gate = threading.Barrier(m + 1)

                def one(i: int) -> None:
                    gate.wait()
                    timings[i] = pipe.step(rids[i], 0, prompt, 0)[1]

                threads = [threading.Thread(target=one, args=(i,)) for i in range(m)]
                for t in threads:
                    t.start()
                gate.wait()
                start = time.perf_counter()
                for t in threads:
                    t.join()
                makespan = (time.perf_counter() - start) * 1000
                for rid in rids:
                    pipe.release(rid)
                busy = [sum(t.compute_ms[k] for t in timings) for k in range(2)]
                runs.append({"makespan_ms": makespan, "idle": [1 - x / makespan for x in busy],
                             "stage_ms": [x / m for x in busy],
                             "overhead_ms": statistics.median(t.overhead_ms for t in timings)})
            med = lambda f: statistics.median(f(x) for x in runs)
            row = {"m": m, "makespan_ms": round(med(lambda x: x["makespan_ms"]), 1),
                   "A_idle": round(med(lambda x: x["idle"][0]), 3), "B_idle": round(med(lambda x: x["idle"][1]), 3),
                   "A_ms": round(med(lambda x: x["stage_ms"][0]), 2), "B_ms": round(med(lambda x: x["stage_ms"][1]), 2)}
            row["measured_idle"] = round((row["A_idle"] + row["B_idle"]) / 2, 3)
            row["model_idle"] = round(bubble_model(m), 3)
            if single is None:                                      # m = 1: the per-step network/RPC overhead
                single = med(lambda x: x["overhead_ms"])
            # the same reasoning with THIS burst's stage times (they grow with m: queued requests are deserialised
            # on the stage's own cores) and the overhead: fill + drain, then one item per bottleneck-stage time
            t_a, t_b = row["A_ms"], row["B_ms"]
            predicted = t_a + t_b + single + (m - 1) * max(t_a, t_b)
            row["model_measured_stages_idle"] = round(1 - m * (t_a + t_b) / 2 / predicted, 3)
            out.append(row)
    return out


def wire() -> list[dict]:
    """Ch 7: bytes on the wire and encode+decode time per payload, protobuf vs torch.save vs JSON."""
    from playground.ch07.serialize_cost import PAYLOADS, time_it, via_json, via_proto, via_torch_save
    out = []
    torch.manual_seed(0)
    for label, shape in PAYLOADS:
        t = torch.randn(shape)
        row = {"payload": label, "raw_bytes": t.nelement() * 4}
        for name, fn in (("protobuf", via_proto), ("torch.save", via_torch_save), ("json", via_json)):
            if name == "json" and row["raw_bytes"] > 2_000_000:
                continue
            size, sec, exact = time_it(fn, t, 20 if row["raw_bytes"] < 1_000_000 else 3)
            row[f"{name}_bytes"], row[f"{name}_ms"], row[f"{name}_exact"] = size, round(sec * 1000, 3), exact
        out.append(row)
    return out


def throughput(base_port: int) -> list[dict]:
    """Ch 11: tokens/s for a burst of 24 requests, one at a time vs slots vs micro-batching."""
    from playground.ch11.throughput import BURST, TOKENS, burst
    configs = [("1 at a time", 1, 0), ("4 slots", 4, 0), ("batch 8", 1, 8), ("batch 24", 1, 24),
               ("2 x batch 12", 2, 12)]
    a, b = two_stage_layout(base_port=base_port)
    layout = [WorkerSpec(a.worker_id, a.port, a.first, a.last, embed=True, extra_args=("--threads", "3")),
              WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, extra_args=("--threads", "3"))]
    out = []
    with running_cluster(layout) as pipe:
        for k, (label, slots, max_batch) in enumerate(configs):
            server, sched = start_coordinator(pipe, base_port + 2 + k, queue_size=64, slots=slots, max_batch=max_batch)
            channel = grpc.insecure_channel(f"127.0.0.1:{base_port + 2 + k}")
            elapsed, latencies, ok = burst(pb_grpc.CoordinatorStub(channel))
            channel.close()
            server.stop(grace=None)
            sched.stop()
            out.append({"config": label, "tokens_per_s": round(BURST * TOKENS / elapsed, 1),
                        "mean_latency_s": round(statistics.mean(latencies), 2), "all_match_reference": ok})
    return out


def load(base_port: int, rates: list[float], seconds: float) -> list[dict]:
    """Ch 10: open-loop load on a bounded (8) and an unbounded queue; Little's Law check."""
    from playground.ch10.load_test import one_request, run_load
    out = []
    with running_cluster(two_stage_layout(base_port=base_port)) as pipe:
        servers = [start_coordinator(pipe, base_port + 2, queue_size=8),
                   start_coordinator(pipe, base_port + 3, queue_size=1000)]
        channels = [grpc.insecure_channel(f"127.0.0.1:{base_port + p}") for p in (2, 3)]
        bounded, unbounded = (pb_grpc.CoordinatorStub(c) for c in channels)
        one_request(bounded, "warm-up")
        for rate in rates:
            out.append({"queue": "bounded 8", **run_load(bounded, rate, seconds, f"b{rate}")})
        out.append({"queue": "unbounded", **run_load(unbounded, max(rates), seconds, "u")})
        for c in channels:
            c.close()
        for server, sched in servers:
            server.stop(grace=None)
            sched.stop()
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in out]


def recovery(base_port: int) -> dict:
    """Ch 12-13: detection and restart times, replay vs regeneration, and a crash inside a request."""
    from playground.ch12.recovery import crash_and_hang
    cd, cr, hd, hr = crash_and_hang(0.25, base_port)
    result = {"heartbeat_interval_s": 0.25, "crash_detect_s": round(cd, 2), "crash_restart_s": round(cr, 2),
              "hang_detect_s": round(hd, 2), "hang_restart_s": round(hr, 2), "replay": []}
    with running_cluster(two_stage_layout(base_port=base_port + 4)) as pipe:
        tok, _ = pipe.step("probe", 0, PROMPT, 0)
        decode = statistics.median(pipe.step("probe", s, [262], 4 + s)[1].total_ms for s in range(1, 21))
        pipe.release("probe")
        for length in (25, 100, 400):
            prefix = [1000 + i for i in range(length)]
            replay = statistics.median(pipe.step(f"r{length}-{k}", 0, prefix, 0)[1].total_ms for k in range(3))
            for k in range(3):
                pipe.release(f"r{length}-{k}")
            result["replay"].append({"tokens": length, "replay_ms": round(replay, 1),
                                     "regenerate_ms": round(length * decode, 1)})
    with tempfile.TemporaryDirectory() as d, supervised_cluster(two_stage_layout(base_port=base_port + 6)) as (pipe, sup):
        log = TokenLog(Path(d) / "log.jsonl")
        t0 = time.monotonic()
        calm = generate_logged(pipe, log, "calm", PROMPT, 100, health=sup)
        calm_s = time.monotonic() - t0
        out = {}
        t0 = time.monotonic()
        t = threading.Thread(target=lambda: out.setdefault("r", generate_logged(pipe, log, "storm", PROMPT, 100, health=sup)))
        t.start()
        time.sleep(1.0)
        sup.kill(1)
        t.join()
        storm_s = time.monotonic() - t0
        log.close()
    result.update({"request_100_tokens_s": round(calm_s, 2), "with_worker_crash_s": round(storm_s, 2),
                   "replays": out["r"].replays, "identical_output": out["r"].new_ids == calm.new_ids})
    return result


def rebalance(quick: bool) -> dict:
    """Ch 15: tokens/s over time, static split vs dynamic rebalancing, with worker A slowing down."""
    import playground.ch15.rebalance_demo as demo
    if quick:
        demo.RUN_S, demo.SLOW_AT_S = 32, 12
    rows = {name: demo.run(flag, port) for name, flag, port in
            (("static", False, 50341), ("dynamic", True, 50345))}
    return {"slow_at_s": demo.SLOW_AT_S,
            "timeline": {name: [{"t": t, "tokens_per_s": round(r, 1), "detail": info} for t, r, info in data]
                         for name, data in rows.items()}}


def reference_check(base_port: int) -> dict:
    """G1: the distributed engine still reproduces results/reference.json token for token."""
    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    with running_cluster(two_stage_layout(base_port=base_port)) as pipe:
        matches = [generate(pipe, f"ref-{i}", e["prompt_ids"], reference["max_new_tokens"])[0] == e["new_ids"]
                   for i, e in enumerate(reference["entries"])]
    return {"prompts": len(matches), "matching": sum(matches)}
