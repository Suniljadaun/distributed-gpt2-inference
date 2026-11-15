"""
run_all.py - Chapter 16
One command for every measurement in the report. Writes, under results/:
  bench/<experiment>.json   raw data (the source of truth)
  charts/<chart>.png        Matplotlib charts made from that data (plots.py), for the report
  charts/<chart>.svg        the same charts from the dependency-free SVG writer (svg.py)
  summary.md                environment + one table per experiment, ready to paste into the report,
                            plus NumPy analysis (stats.py): growth fits and bootstrap 95% CIs
Run:  python -m src.bench.run_all                  (all experiments, about 7 minutes)
      python -m src.bench.run_all --quick          (shorter runs, about 4 minutes)
      python -m src.bench.run_all --only wire,load (just some)
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import grpc
import torch

from src.bench import experiments as ex
from src.bench import plots, stats
from src.bench.svg import bar_chart, line_chart

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ALL = ["reference", "kv_cache", "split_sweep", "wire", "throughput", "bubble", "load", "recovery", "rebalance"]


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
    except OSError:
        commit = "unknown"
    return {"date": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
            "machine": platform.processor(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__, "grpc": grpc.__version__}


def run_experiment(name: str, quick: bool):
    if name == "reference":
        return ex.reference_check(50401)
    if name == "kv_cache":
        return ex.kv_cache(60 if quick else 200)
    if name == "split_sweep":
        return ex.split_sweep(50410)
    if name == "wire":
        return ex.wire()
    if name == "throughput":
        return ex.throughput(50430)
    if name == "bubble":
        return ex.bubble(50440, repeats=3 if quick else 5)
    if name == "load":
        return ex.load(50450, [1.0, 3.0, 6.0], 6.0 if quick else 12.0)
    if name == "recovery":
        return ex.recovery(50460)
    if name == "rebalance":
        return ex.rebalance(quick)
    raise ValueError(f"unknown experiment {name!r}")


def charts(name: str, data, out: Path) -> None:
    if name == "kv_cache":
        line_chart(out / "kv_cache.svg", "Decode step time: no cache vs KV-cache (1 process)", "token number",
                   "ms per step", {k: list(enumerate(v, 1)) for k, v in data["step_ms"].items()})
    elif name == "split_sweep":
        bar_chart(out / "split_sweep.svg", "Per-stage compute per decode step, by split (A layers / B layers)",
                  "ms", [r["split"] for r in data],
                  {"A": [r["A_ms"] for r in data], "B": [r["B_ms"] for r in data],
                   "overhead": [r["overhead_ms"] for r in data]})
    elif name == "throughput":
        bar_chart(out / "throughput.svg", "Throughput for a burst of 24 requests (20 tokens each)", "tokens/s",
                  [r["config"] for r in data], {"tokens/s": [r["tokens_per_s"] for r in data]})
    elif name == "load":
        bounded = [r for r in data if r["queue"] == "bounded 8"]
        line_chart(out / "load_latency.svg", "Latency vs arrival rate (bounded queue of 8)", "requests/s offered",
                   "seconds", {"p50": [(r["rate"], r["p50"]) for r in bounded],
                               "p99": [(r["rate"], r["p99"]) for r in bounded]})
    elif name == "rebalance":
        line_chart(out / "rebalance.svg", f"Throughput over time; worker A slows down at ~{data['slow_at_s']} s",
                   "seconds", "tokens/s", {k: [(p["t"], p["tokens_per_s"]) for p in v]
                                           for k, v in data["timeline"].items()})


def _pct(g: dict) -> str:
    return (f"**{g['gain'] * 100:+.0f}%** ({g['baseline_mean']:.1f} → {g['treatment_mean']:.1f} tok/s; "
            f"95% CI {g['ci'][0] * 100:+.0f}% … {g['ci'][1] * 100:+.0f}%, {g['n'][0]} buckets each)")


def analysis(name: str, data) -> str:
    """Ch 18: NumPy analysis lines under the table (empty if there is nothing to add)."""
    if name == "kv_cache":
        fits = {k: stats.growth(v) for k, v in data["step_ms"].items()}
        return "Least-squares fit of step time vs token number: " + "; ".join(
            f"{k} **{f['slope']:+.3f} ms per token** (R² {f['r2']:.2f})" for k, f in fits.items()) + "."
    if name == "rebalance":
        an = stats.rebalance_analysis(data)
        lines = [f"Worker A is clearly slow from the t = {an['slow_start_t']} bucket. Dynamic vs static:",
                 f"- steady: {_pct(an['steady'])}"]
        if "slowed" in an:
            lines.append(f"- A slowed: {_pct(an['slowed'])}")
        return "\n".join(lines)
    return ""


def table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) if k in r else "" for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def summary_section(name: str, data) -> str:
    if name == "reference":
        return f"**{data['matching']} of {data['prompts']}** reference prompts reproduced token for token by 2 workers."
    if name == "kv_cache":
        no, yes = data["step_ms"]["no cache"], data["step_ms"]["KV-cache"]
        return (f"{len(no)} tokens: **{data['tokens_per_s']['no cache']} vs {data['tokens_per_s']['KV-cache']} tokens/s**; "
                f"last step {no[-1]:.1f} ms without cache vs {yes[-1]:.1f} ms with it. Chart: `charts/kv_cache.svg`")
    if name == "recovery":
        replay = table(data["replay"])
        rest = {k: v for k, v in data.items() if k != "replay"}
        return table([rest]) + "\n\n" + replay
    if name == "rebalance":
        static, dynamic = data["timeline"]["static"], data["timeline"]["dynamic"]
        return table([{"t": s["t"], "static tok/s": s["tokens_per_s"], "static": s["detail"].replace("|", "/"),
                       "dynamic tok/s": d["tokens_per_s"], "dynamic": d["detail"].replace("|", "/")}
                      for s, d in zip(static, dynamic)])
    return table(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(ALL))
    parser.add_argument("--from-saved", action="store_true",
                        help="run nothing: rebuild charts and summary.md from results/bench/*.json")
    args = parser.parse_args()
    names = args.only.split(",") if args.only else ALL

    (RESULTS / "bench").mkdir(parents=True, exist_ok=True)
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    env = environment()
    sections = []
    for name in names:
        if args.from_saved:
            saved = json.loads((RESULTS / "bench" / f"{name}.json").read_text(encoding="utf-8"))
            data, env, args.quick = saved["data"], saved["environment"], saved["quick"]   # the run's own env
            took = saved["seconds"]
        else:
            print(f"[bench] {name} ...", flush=True)
            start = time.monotonic()
            data = run_experiment(name, args.quick)
            took = time.monotonic() - start
            (RESULTS / "bench" / f"{name}.json").write_text(
                json.dumps({"environment": env, "quick": args.quick, "seconds": round(took, 1), "data": data},
                           indent=2), encoding="utf-8")
        charts(name, data, RESULTS / "charts")
        plots.draw(name, data, RESULTS / "charts")
        extra = analysis(name, data)
        sections.append(f"## {name}\n\n{summary_section(name, data)}\n" + (f"\n{extra}\n" if extra else ""))
        print(f"[bench] {name} done in {took:.0f} s", flush=True)

    header = ["# Benchmark summary", "", f"Generated by `python -m src.bench.run_all{' --quick' if args.quick else ''}`"
              f"{' --only ' + args.only if args.only else ''}", "", table([env]), ""]
    (RESULTS / "summary.md").write_text("\n".join(header + sections), encoding="utf-8")
    print(f"[bench] wrote {RESULTS / 'summary.md'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
