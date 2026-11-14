"""
stats.py - NumPy analysis for the benchmark results (used by run_all and plots).
- growth(): least-squares line through step time vs token number -> "ms more per token"
  (the KV-cache claim, as a slope: without the cache every token makes the next step slower).
- bootstrap_ci(): 95% confidence interval of a statistic by resampling (no normality assumption,
  which matters for a handful of noisy throughput buckets).
- gain(): relative gain of one configuration over another, with its bootstrap 95% CI.
"""
import numpy as np

RNG_SEED = 0              # fixed: the same data always gives the same interval


def growth(values: list[float]) -> dict:
    """Fit values[i] ~ intercept + slope * i. Returns the slope (per step) and the fit quality."""
    y = np.asarray(values, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0}


def bootstrap_ci(values: list[float], stat=np.mean, n: int = 10_000, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap: resample with replacement n times, take the middle `level` of the statistic."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    samples = stat(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def gain(baseline: list[float], treatment: list[float], n: int = 10_000, level: float = 0.95) -> dict:
    """mean(treatment) / mean(baseline) - 1, with a bootstrap CI (both groups resampled independently)."""
    a, b = np.asarray(baseline, dtype=float), np.asarray(treatment, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    ra = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    rb = b[rng.integers(0, len(b), size=(n, len(b)))].mean(axis=1)
    lo, hi = np.percentile(rb / ra - 1, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"baseline_mean": float(a.mean()), "treatment_mean": float(b.mean()),
            "gain": float(b.mean() / a.mean() - 1), "ci": (float(lo), float(hi)), "n": (len(a), len(b))}


def slow_start(timeline: list[dict], factor: float = 1.8) -> int | None:
    """First bucket time t where stage A's median step time is `factor` x its value at the start (the slowdown)."""
    a_ms = [float(p["detail"].split()[1]) for p in timeline]        # detail: "A 10.1 B 19.3 ms  0-2 | 3-5"
    base = float(np.median(a_ms[1:4])) if len(a_ms) > 4 else a_ms[0]
    for p, a in zip(timeline, a_ms):
        if a > factor * base:
            return int(p["t"])
    return None


def rebalance_analysis(data: dict) -> dict:
    """Static vs dynamic throughput, before and after the slowdown, with CIs.
    Bucket 0 (warm-up, first move) and the bucket in which the slowdown starts are left out."""
    static, dynamic = data["timeline"]["static"], data["timeline"]["dynamic"]
    slow = slow_start(static)
    bucket = static[1]["t"] - static[0]["t"]
    steady = [i for i, p in enumerate(static) if p["t"] > 0 and (slow is None or p["t"] + bucket <= slow - bucket)]
    slowed = [i for i, p in enumerate(static) if slow is not None and p["t"] >= slow + bucket]
    rate = lambda tl, idx: [tl[i]["tokens_per_s"] for i in idx]
    out = {"slow_start_t": slow, "bucket_s": bucket,
           "steady": gain(rate(static, steady), rate(dynamic, steady))}
    if slowed:
        out["slowed"] = gain(rate(static, slowed), rate(dynamic, slowed))
    return out
