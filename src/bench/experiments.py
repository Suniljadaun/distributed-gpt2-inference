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


def kv_cache(steps: int, base_port: int | None = None) -> dict:
    """Ch 9: time of every decode step, with and without the KV-cache, in one process; with base_port, also through
    the 2-worker pipeline. Pipeline without cache = Chapter 8's design: every step sends the WHOLE sequence (a fresh
    request id at step 0, released afterwards), so A ships seq x 768 floats to B each time."""
    model = load_pretrained()
    rows, wire = {}, {}
    for name, use_cache in (("no cache", False), ("KV-cache", True)):
        ids, cache, feed, times = list(PROMPT), KVCache(), list(PROMPT), []
        for _ in range(steps):
            t0 = time.perf_counter()
            tok = cached_step(model, cache, feed) if use_cache else next_token(model, ids)
            times.append((time.perf_counter() - t0) * 1000)
            ids.append(tok)
            feed = [tok]
        rows[name] = times
    if base_port is not None:
        with running_cluster(two_stage_layout(base_port=base_port)) as pipe:
            for k in range(3):                                      # warm-up
                generate(pipe, f"warm-{k}", PROMPT, 5)
            ids, times = list(PROMPT), []
            for i in range(steps):                                  # no cache: whole sequence, every step
                tok, timing = pipe.step(f"nocache-{i}", 0, ids, 0)
                pipe.release(f"nocache-{i}")
                times.append(timing.total_ms)
                ids.append(tok)
            rows["pipeline, no cache"] = times
            new_ids, timings = generate(pipe, "cached", PROMPT, steps)
            assert new_ids == ids[len(PROMPT):], "pipeline with and without cache disagree"
            rows["pipeline, KV-cache"] = [t.total_ms for t in timings]
        hidden = 768 * 4                                            # one position's hidden state, float32
        wire = {"no cache": (len(PROMPT) + steps - 1) * hidden, "KV-cache": hidden}   # A -> B, last step
    return {"step_ms": rows, "a_to_b_bytes_last_step": wire,
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


def _percentiles(first: list[float], rest: list[float]) -> dict:
    import numpy as np
    p = lambda xs, q: round(float(np.percentile(xs, q)), 2)
    return {"ttft_p50_ms": p(first, 50), "ttft_p99_ms": p(first, 99),
            "itl_p50_ms": p(rest, 50), "itl_p99_ms": p(rest, 99), "itl_mean_ms": round(statistics.mean(rest), 2),
            "tokens_per_s_per_request": round(1000 / statistics.mean(rest), 1), "samples": len(rest)}


def token_latency(base_port: int, runs: int = 10, tokens: int = 60, concurrent: int = 4) -> dict:
    """Per-token latency, p50 / p99: time to the first token (prefill) and between tokens (one decode step).
    Single process vs the 2-worker pipeline (one request at a time, then `concurrent` at once). The same reference
    prompts each way; pipeline times are measured by the coordinator around each step (both hops included)."""
    prompts = [e["prompt_ids"] for e in json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))["entries"]]
    rows, samples = [], {}

    model = load_pretrained()
    first, rest = [], []
    for r in range(runs + 1):                                       # run 0 = warm-up, not counted
        cache, feed, times = KVCache(), list(prompts[r % len(prompts)]), []
        for _ in range(tokens):
            t0 = time.perf_counter()
            with torch.no_grad():
                tok = cached_step(model, cache, feed)
            times.append((time.perf_counter() - t0) * 1000)
            feed = [tok]
        if r:
            first.append(times[0])
            rest += times[1:]
    rows.append({"config": "single process", **_percentiles(first, rest)})
    samples["single process"] = rest
    del model

    with running_cluster(two_stage_layout(base_port=base_port)) as pipe:
        first, rest = [], []
        for r in range(runs + 1):
            _, timings = generate(pipe, f"lat-{r}", prompts[r % len(prompts)], tokens)
            if r:
                first.append(timings[0].total_ms)
                rest += [t.total_ms for t in timings[1:]]
        rows.append({"config": "pipeline, 2 workers", **_percentiles(first, rest)})
        samples["pipeline, 2 workers"] = rest

        first, rest, lock = [], [], threading.Lock()

        def one(c: int, r: int) -> None:
            _, timings = generate(pipe, f"lat-c{c}-{r}", prompts[(r + c) % len(prompts)], tokens)
            with lock:
                first.append(timings[0].total_ms)
                rest.extend(t.total_ms for t in timings[1:])

        for r in range(max(1, runs // 2)):                          # runs/2 rounds of `concurrent` requests
            threads = [threading.Thread(target=one, args=(c, r)) for c in range(concurrent)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        label = f"pipeline, {concurrent} at once"
        rows.append({"config": label, **_percentiles(first, rest)})
        samples[label] = rest
    return {"rows": rows, "itl_ms": {k: [round(x, 2) for x in v] for k, v in samples.items()}}


KV_BYTES_PER_TOKEN = 2 * 768 * 4             # one layer: K and V, 768 float32 values each, per cached position


def kv_migration(base_port: int, requests=(1, 4, 16), contexts=(64, 256, 512), repeats: int = 3) -> list[dict]:
    """G6: what one layer move costs. R requests of L cached tokens each are in flight; layer 2 moves A -> B and
    back `repeats` times. Per configuration: bytes of KV-cache migrated per move (vs R x L x 6,144), the pause
    (all steps blocked: export + import + commits) and the preparation (weights loaded, no pause)."""
    from src.coordinator.rebalancer import Rebalancer
    out = []
    with running_cluster(two_stage_layout(base_port=base_port)) as pipe:
        reb = Rebalancer(pipe)
        reb.move(0, 1)                                              # warm-up: first move loads code paths
        reb.move(1, 0)
        for r in requests:
            for ctx in contexts:
                prompt = (PROMPT * (ctx // len(PROMPT) + 1))[:ctx]
                rids = [f"kv-{r}-{ctx}-{i}" for i in range(r)]
                for rid in rids:
                    pipe.step(rid, 0, prompt, 0)                    # prefill: every stage caches ctx positions
                moves = []
                for _ in range(repeats):
                    moves += [reb.move(0, 1), reb.move(1, 0)]
                for rid in rids:
                    pipe.release(rid)
                assert all(m.kv_requests == r for m in moves)
                kv = moves[0].kv_bytes
                out.append({"requests": r, "context": ctx, "kv_MB": round(kv / 1e6, 3),
                            "expected_MB": round(r * ctx * KV_BYTES_PER_TOKEN / 1e6, 3),
                            "bytes_per_token": round(kv / (r * ctx)),
                            "pause_ms": round(statistics.median(m.pause_ms for m in moves), 1),
                            "pause_max_ms": round(max(m.pause_ms for m in moves), 1),
                            "prepare_ms": round(statistics.median(m.prepare_ms for m in moves), 1),
                            "moves": len(moves)})
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
    """Ch 15: tokens/s over time with worker A slowing down: static split, the model policy ("dynamic") and
    the scope report's imbalance-threshold policy ("threshold")."""
    import playground.ch15.rebalance_demo as demo
    if quick:
        demo.RUN_S, demo.SLOW_AT_S = 32, 12
    runs = {name: demo.run(policy, port) for name, policy, port in
            (("static", None, 50341), ("dynamic", "model", 50345), ("threshold", "threshold", 50348))}
    return {"slow_at_s": demo.SLOW_AT_S,
            "timeline": {name: [{"t": t, "tokens_per_s": round(r, 1), "detail": info} for t, r, info in rows]
                         for name, (rows, _) in runs.items()},
            "moves": {name: [{"t": t, "event": text} for t, text in moves] for name, (_, moves) in runs.items()}}


def stages(base_port: int, tokens: int = 60, runs: int = 3, clients: int = 8, client_tokens: int = 30,
           rebalance_s: float = 25.0) -> list[dict]:
    """Ch 18 (scope: "extendable to 3 workers"): the same 6 cores as 1 process, 2 workers (3 cores each) or
    3 workers (2 cores each). Each worker is a serial stage on cores of its own; the coordinator and its clients
    run on the other cores. Per config: one request at a time (latency) and `clients` requests at once
    (pipelining). Every output is checked against the single-process one. Then the rebalancer runs on the
    3 workers under that load for `rebalance_s` seconds, and the layout it settles on is measured again."""
    import os
    from src.coordinator.cluster import n_stage_layout
    from src.worker.server import pin_to_cpus
    n_cpu = os.cpu_count() or 1
    pin = n_cpu >= 8                                                # else: no pinning (e.g. a 4-core CI runner)
    prompts = [PROMPT, [464, 2068, 7586, 21831, 18045], [40, 1842, 284, 2193, 546]]
    model = load_pretrained()
    threads_before = torch.get_num_threads()
    if pin:
        pin_to_cpus(list(range(6)))
    torch.set_num_threads(6)

    def single(ids: list[int], n: int) -> tuple[list[int], list[float]]:
        cache, feed, out, ms = KVCache(), list(ids), [], []
        for _ in range(n):
            t0 = time.perf_counter()
            tok = cached_step(model, cache, feed)
            ms.append((time.perf_counter() - t0) * 1000)
            out.append(tok)
            feed = [tok]
        return out, ms

    single(PROMPT, 5)                                               # warm-up
    expected = {i: single(pr, max(tokens, client_tokens))[0] for i, pr in enumerate(prompts)}
    step = [statistics.median(single(PROMPT, tokens)[1][1:]) for _ in range(runs)]
    rows = [{"config": "1 process", "cores": "6", "layers": "0-5", "decode_ms": round(statistics.median(step), 1),
             "tokens_per_s": round(1000 / statistics.median(step), 1), "overhead_ms": 0.0,
             "stage_ms": [round(statistics.median(step), 1)], "concurrent_tokens_per_s": None, "outputs_match": True}]
    torch.set_num_threads(threads_before)
    if pin:
        pin_to_cpus(list(range(6, n_cpu)))                          # coordinator + clients: not on the workers' cores

    def measure(pipe, config: str, cores: str, layers: list[tuple[int, int]]) -> dict:
        ok, step, over, comp, rates = True, [], [], [], []
        for r in range(runs):
            out, timings = generate(pipe, f"{config}-single-{r}", PROMPT, tokens)
            ok &= out == expected[0][:tokens]
            step.append(statistics.median(t.total_ms for t in timings[1:]))
            over.append(statistics.median(t.overhead_ms for t in timings[1:]))
            comp.append([statistics.median(t.compute_ms[k] for t in timings[1:]) for k in range(len(layers))])
        for r in range(runs):
            outs = [None] * clients

            def client(i: int) -> None:
                outs[i] = generate(pipe, f"{config}-c{r}-{i}", prompts[i % 3], client_tokens)[0]

            threads = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            rates.append(clients * client_tokens / (time.perf_counter() - t0))
            ok &= all(o == expected[i % 3][:client_tokens] for i, o in enumerate(outs))
        return {"config": config, "cores": cores, "layers": " | ".join(f"{f}-{l - 1}" for f, l in layers),
                "decode_ms": round(statistics.median(step), 1), "tokens_per_s": round(1000 / statistics.median(step), 1),
                "overhead_ms": round(statistics.median(over), 1),
                "stage_ms": [round(statistics.median(c[k] for c in comp), 1) for k in range(len(layers))],
                "concurrent_tokens_per_s": round(statistics.median(rates), 1), "outputs_match": ok}

    try:
        for n_stages, port in ((2, base_port), (3, base_port + 3)):
            per = 6 // n_stages
            cores = " + ".join([str(per)] * n_stages)
            layout = [WorkerSpec(s.worker_id, s.port, s.first, s.last, embed=s.embed, head=s.head,
                                 extra_args=("--serial", "--threads", str(per))
                                 + (("--cpus", ",".join(str(c) for c in range(k * per, (k + 1) * per))) if pin else ()))
                      for k, s in enumerate(n_stage_layout(n_stages, base_port=port))]
            with running_cluster(layout) as pipe:
                for k in range(3):
                    generate(pipe, f"warm-{k}", PROMPT, 5)
                rows.append(measure(pipe, f"{n_stages} workers", cores, [(s.first, s.last) for s in layout]))
                if n_stages == 3:                                   # let the rebalancer find a better split
                    from src.coordinator.rebalancer import Rebalancer
                    reb = Rebalancer(pipe, period_s=0.5, window_s=2.0, cooldown_s=2.5, min_samples=15)
                    reb.start()
                    stop = time.monotonic() + rebalance_s

                    def load(i: int) -> None:
                        k = 0
                        while time.monotonic() < stop:
                            generate(pipe, f"reb-{i}-{k}", prompts[i % 3], client_tokens)
                            k += 1

                    threads = [threading.Thread(target=load, args=(i,)) for i in range(clients)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    reb.stop()
                    row = measure(pipe, "3 workers, rebalanced", cores, reb.layout)
                    row["moves"] = [f"layer {m.layer}: {'ABC'[m.src]} -> {'ABC'[m.dst]}" for m in reb.moves]
                    rows.append(row)
    finally:
        if pin:
            pin_to_cpus(list(range(n_cpu)))
    return rows


def reference_check(base_port: int) -> dict:
    """G1: the distributed engine still reproduces results/reference.json token for token."""
    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    with running_cluster(two_stage_layout(base_port=base_port)) as pipe:
        matches = [generate(pipe, f"ref-{i}", e["prompt_ids"], reference["max_new_tokens"])[0] == e["new_ids"]
                   for i, e in enumerate(reference["entries"])]
    return {"prompts": len(matches), "matching": sum(matches)}
