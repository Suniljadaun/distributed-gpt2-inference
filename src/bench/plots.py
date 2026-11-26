"""
plots.py - Matplotlib charts for the benchmark results (PNG, for the report).
Called by run_all after each experiment, or on its own to redraw from saved data without rerunning:
    python -m src.bench.plots            (reads results/bench/*.json, writes results/charts/*.png)
The dependency-free SVG writer (svg.py) still produces the .svg versions.
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")                     # no window: files only (works in CI too)
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

from src.bench import stats               # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
BLUE, RED, GREEN, GREY = "#1f6fb4", "#d62728", "#2ca02c", "#7f7f7f"
plt.rcParams.update({"figure.figsize": (7.2, 3.8), "figure.dpi": 100, "savefig.dpi": 160, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e5", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def kv_cache(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    styles = {"no cache": (RED, "-"), "KV-cache": (BLUE, "-"),
              "pipeline, no cache": ("#f59e0b", "-"), "pipeline, KV-cache": (GREEN, "-")}
    for name, ms in data["step_ms"].items():
        colour, ls = styles.get(name, (GREY, "-"))
        x = np.arange(1, len(ms) + 1)
        fit = stats.growth(ms[1:])                                     # step 1 is the prefill: not in the fit
        label = name if name.startswith("pipeline") else f"1 process, {name}"
        ax.plot(x, ms, color=colour, lw=1.1, ls=ls,
                label=f"{label}: {data['tokens_per_s'][name]} tok/s, {fit['slope']:+.2f} ms/token")
        ax.plot(x[1:], fit["intercept"] + fit["slope"] * x[1:], color=colour, lw=0.9, ls="--")
    title = "Decode step time: no cache vs KV-cache" + ("" if len(data["step_ms"]) > 2 else " (1 process)")
    ax.set(title=title, xlabel="token number", ylabel="ms per step")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, out / "kv_cache.png")


def _grouped_bars(ax, labels: list[str], series: dict[str, list[float]], colours) -> None:
    width = 0.8 / len(series)
    x = np.arange(len(labels))
    for k, ((name, values), colour) in enumerate(zip(series.items(), colours)):
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * width, values, width, label=name, color=colour)
        ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
    ax.set_xticks(x, labels)


def split_sweep(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots()
    _grouped_bars(ax, [r["split"] for r in data], {"stage A": [r["A_ms"] for r in data],
                                                   "stage B": [r["B_ms"] for r in data],
                                                   "network + serialisation": [r["overhead_ms"] for r in data]},
                  (BLUE, RED, GREY))
    ax.set(title="Per-stage time per decode step, by split (A blocks / B blocks)", xlabel="split", ylabel="ms")
    ax.legend(loc="upper center", ncol=3)
    ax.set_ylim(0, max(max(r["A_ms"], r["B_ms"]) for r in data) * 1.25)
    _save(fig, out / "split_sweep.png")


def throughput(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots()
    labels = [r["config"] for r in data]
    rates = [r["tokens_per_s"] for r in data]
    bars = ax.bar(labels, rates, color=BLUE, width=0.6)
    ax.bar_label(bars, labels=[f"{r:.1f}\n({r / rates[0]:.1f}×)" for r in rates], fontsize=8, padding=2)
    ax.set(title="Throughput for a burst of 24 requests (20 tokens each)", ylabel="tokens/s")
    ax.set_ylim(0, max(rates) * 1.25)
    _save(fig, out / "throughput.png")


def load(data: list[dict], out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    bounded = [r for r in data if r["queue"] == "bounded 8"]
    rate = [r["rate"] for r in bounded]
    ax.plot(rate, [r["p50"] for r in bounded], "o-", color=BLUE, label="p50")
    ax.plot(rate, [r["p99"] for r in bounded], "s-", color=RED, label="p99")
    ax.set(title="Latency, bounded queue (8)", xlabel="requests/s offered", ylabel="seconds")
    ax.set_ylim(bottom=0)
    ax.legend()
    top = max(r["rate"] for r in data)
    pair = [r for r in data if r["rate"] == top]                     # bounded vs unbounded at the highest rate
    names = [r["queue"] for r in pair]
    bottom = np.zeros(len(pair))
    for key, colour in (("ok", GREEN), ("rejected", GREY), ("expired", RED)):
        values = np.array([r[key] for r in pair], dtype=float)
        ax2.bar(names, values, bottom=bottom, color=colour, label=key, width=0.55)
        bottom += values
    ax2.set(title=f"Outcomes at {top:g} req/s", ylabel="requests")
    ax2.set_ylim(0, bottom.max() * 1.3)
    ax2.legend(fontsize=8, loc="upper center", ncol=3, handlelength=1, columnspacing=0.8)
    _save(fig, out / "load_latency.png")


def wire(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots()
    rows = [r for r in data if r.get("json_ms") not in (None, "")]
    formats = {"protobuf (raw bytes)": "protobuf_ms", "torch.save": "torch.save_ms", "JSON": "json_ms"}
    _grouped_bars(ax, [r["payload"] for r in rows], {k: [r[v] for r in rows] for k, v in formats.items()},
                  (BLUE, GREEN, RED))
    ax.set_yscale("log")
    ax.set(title="Encode + decode time per message (log scale)", ylabel="ms")
    ax.legend(loc="upper left")
    _save(fig, out / "wire.png")


def rebalance(data: dict, out: Path) -> None:
    static, dynamic = data["timeline"]["static"], data["timeline"]["dynamic"]
    an = stats.rebalance_analysis(data)
    fig, ax = plt.subplots()
    lines = [("static 3/3", static, BLUE, "o-"), ("rebalancer: model", dynamic, RED, "o-")]
    if "threshold" in data["timeline"]:
        lines.append(("rebalancer: threshold 1.25", data["timeline"]["threshold"], GREEN, "s--"))
    for name, tl, colour, style in lines:
        ax.plot([p["t"] for p in tl], [p["tokens_per_s"] for p in tl], style, color=colour, ms=4, label=name)
    if an["slow_start_t"] is not None:
        start = an["slow_start_t"] - an["bucket_s"]
        ax.axvspan(start, static[-1]["t"] + an["bucket_s"], color="#fde68a", alpha=0.45, lw=0)
        ax.text(start + 0.5, 4, "worker A +8 ms per block", fontsize=8, color="#92400e")
    for key, label in (("steady", "steady"), ("slowed", "A slowed")):
        if key in an:
            g = an[key]
            print_ci = f"model, {label}: {g['gain'] * 100:+.0f}% (95% CI {g['ci'][0] * 100:+.0f}…{g['ci'][1] * 100:+.0f}%)"
            ax.annotate(print_ci, xy=(0.99, 0.97 if key == "steady" else 0.89), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8.5)
    ax.set(title="Throughput over time, 4 requests in flight (4 s buckets)", xlabel="seconds", ylabel="tokens/s")
    ax.set_ylim(0, max(p["tokens_per_s"] for _, tl, _, _ in lines for p in tl) * 1.3)
    ax.legend(loc="lower left")
    _save(fig, out / "rebalance.png")


def bubble(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots()
    m = [r["m"] for r in data]
    ax.plot(m, [r["model_idle"] for r in data], "--", color=GREY, lw=1.5, label="model (p−1)/(m+p−1), equal stages")
    ax.plot(m, [r["model_measured_stages_idle"] for r in data], ":", color="black", lw=1.5,
            label="model with measured stage times + overhead")
    ax.plot(m, [r["measured_idle"] for r in data], "o-", color=RED, lw=2, label="measured, mean of both stages")
    ax.plot(m, [r["A_idle"] for r in data], "^", color=BLUE, ms=6, label="measured, stage A (faster)")
    ax.plot(m, [r["B_idle"] for r in data], "v", color=GREEN, ms=6, label="measured, stage B (slower)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m, [str(x) for x in m])
    ax.set(title="Pipeline bubble: stage idle time for a burst of m micro-batches (p = 2)",
           xlabel="micro-batches in the burst (m)", ylabel="idle fraction")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8)
    _save(fig, out / "bubble.png")


def kv_migration(data: list[dict], out: Path) -> None:
    fig, ax = plt.subplots()
    fit = stats.linear_fit([r["kv_MB"] for r in data], [r["pause_ms"] for r in data])
    for r_count, colour, marker in zip(sorted({r["requests"] for r in data}), (BLUE, GREEN, RED, GREY), "os^v"):
        rows = [r for r in data if r["requests"] == r_count]
        ax.plot([r["kv_MB"] for r in rows], [r["pause_ms"] for r in rows], marker, color=colour, ms=7,
                label=f"{r_count} request{'s' if r_count > 1 else ''} in flight")
        for r in rows:                                      # 4 requests: label above-left, so it never hides another
            ax.annotate(f"{r['requests']}×{r['context']} tok", (r["kv_MB"], r["pause_ms"]), textcoords="offset points",
                        xytext=(-8, 7) if r_count == 4 else (6, -11), ha="right" if r_count == 4 else "left",
                        fontsize=7.5, color=colour)
    sizes = [r["kv_MB"] for r in data]
    x = np.geomspace(min(sizes) * 0.8, max(sizes) * 1.25, 60)
    ax.plot(x, fit["intercept"] + fit["slope"] * x, "--", color="black", lw=1,
            label=f"fit: {fit['intercept']:.1f} ms + {fit['slope']:.1f} ms/MB (R² {fit['r2']:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    pauses = [r["pause_ms"] for r in data]
    ax.set_yticks([t for t in (10, 20, 50, 100, 200, 500, 1000, 2000) if min(pauses) / 2 <= t <= max(pauses) * 2])
    ax.set_ylim(min(pauses) / 1.6, max(pauses) * 1.6)
    ax.set(title="One layer move: pause vs KV-cache migrated (log–log)", xlabel="KV-cache migrated (MB)",
           ylabel="pause, median (ms)")
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, out / "kv_migration.png")


def token_latency(data: dict, out: Path) -> None:
    fig, ax = plt.subplots()
    for (name, values), colour in zip(data["itl_ms"].items(), (GREY, BLUE, RED)):
        x = np.sort(np.asarray(values))
        p50, p99 = (float(np.percentile(x, q)) for q in (50, 99))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=colour, lw=1.8,
                label=f"{name}: p50 {p50:.1f}, p99 {p99:.1f} ms")
        ax.plot([p50, p99], [0.5, 0.99], "o", color=colour, ms=5)      # the two percentiles on the curve
    ax.set(title="Time between tokens (one decode step): distribution", xlabel="ms per token",
           ylabel="fraction of tokens ≤ x")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(max(v) for v in data["itl_ms"].values()) * 1.45)   # room on the right for the legend
    ax.legend(loc="center right", fontsize=8.5)
    _save(fig, out / "token_latency.png")


def stages(data: list[dict], out: Path) -> None:
    """Ch 18: 1 process vs 2 vs 3 workers on the same 6 cores; right: where each decode step's time goes."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = [r["config"].replace(", ", ",\n") for r in data]
    x = np.arange(len(data))
    for k, (key, name, colour) in enumerate((("tokens_per_s", "1 request", BLUE),
                                            ("concurrent_tokens_per_s", "8 at once", GREEN))):
        values = [r[key] or 0 for r in data]
        bars = left.bar(x + (k - 0.5) * 0.38, values, 0.38, label=name, color=colour)
        left.bar_label(bars, labels=[f"{v:.1f}" if v else "–" for v in values], fontsize=8, padding=2)
    left.set_xticks(x, labels, fontsize=8.5)
    left.set(title="Tokens/s on the same 6 cores", ylabel="tokens/s")
    left.set_ylim(0, max(r["concurrent_tokens_per_s"] or r["tokens_per_s"] for r in data) * 1.22)
    left.legend(loc="upper left", fontsize=8.5)
    colours = (BLUE, RED, "#f59e0b")
    for i, r in enumerate(data):
        bottom = 0.0
        ranges = r["layers"].split(" | ")
        for k, ms in enumerate(r["stage_ms"]):
            right.bar(i, ms, 0.6, bottom=bottom, color=colours[k] if len(r["stage_ms"]) > 1 else GREY,
                      edgecolor="white", linewidth=0.8)
            name = "ABC"[k] if len(r["stage_ms"]) > 1 else "all"
            right.text(i, bottom + ms / 2, f"{name} {ranges[k]}\n{ms:.1f} ms", ha="center", va="center",
                       fontsize=7, color="white")
            bottom += ms
        if r["overhead_ms"]:
            right.bar(i, r["overhead_ms"], 0.6, bottom=bottom, color="#d0d0d0", edgecolor="white", linewidth=0.8)
            right.text(i, bottom + r["overhead_ms"] + 0.4, f"+{r['overhead_ms']:.1f} ms\nnetwork", ha="center",
                       va="bottom", fontsize=7)
    right.set_xticks(x, labels, fontsize=8.5)
    right.set(title="One decode step: compute per stage (layers)", ylabel="ms")
    right.set_ylim(0, max(sum(r["stage_ms"]) + r["overhead_ms"] for r in data) * 1.25)
    _save(fig, out / "stages.png")


DRAW = {"kv_cache": kv_cache, "split_sweep": split_sweep, "throughput": throughput, "load": load,
        "wire": wire, "rebalance": rebalance, "bubble": bubble, "kv_migration": kv_migration,
        "token_latency": token_latency, "stages": stages}
FILE = {"load": "load_latency"}                     # experiment name -> chart file name, where they differ


def draw(name: str, data, out: Path) -> Path | None:
    if name not in DRAW:
        return None
    DRAW[name](data, out)
    return out / f"{FILE.get(name, name)}.png"


def main() -> None:
    out = RESULTS / "charts"
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted((RESULTS / "bench").glob("*.json")):
        written = draw(path.stem, json.loads(path.read_text(encoding="utf-8"))["data"], out)
        if written:
            print(f"[plots] {written}")


if __name__ == "__main__":
    sys.exit(main())
