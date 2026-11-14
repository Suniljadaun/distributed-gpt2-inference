"""
svg.py - Chapter 16
Tiny dependency-free chart writer: line charts and grouped bar charts as SVG files.
(SVG is plain text XML: it opens in any browser or VS Code, and Word can insert it.)
"""
from html import escape
from pathlib import Path

W, H = 720, 420
LEFT, RIGHT, TOP, BOTTOM = 70, 170, 50, 60          # margins; the legend lives on the right
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def _nice_max(v: float) -> float:
    """Round an axis maximum up to 1, 2 or 5 x a power of ten."""
    if v <= 0:
        return 1.0
    p = 10 ** len(str(int(v))) / 10
    for m in (1, 2, 5, 10):
        if m * p >= v:
            return m * p
    return 10 * p


def _frame(title: str, xlabel: str, ylabel: str, ymax: float) -> list[str]:
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W / 2}" y="24" text-anchor="middle" font-size="15" font-weight="bold">{escape(title)}</text>',
           f'<text x="{LEFT + pw / 2}" y="{H - 15}" text-anchor="middle">{escape(xlabel)}</text>',
           f'<text x="18" y="{TOP + ph / 2}" text-anchor="middle" transform="rotate(-90 18 {TOP + ph / 2})">'
           f'{escape(ylabel)}</text>']
    for k in range(6):                                                   # horizontal grid + y labels
        y = TOP + ph - ph * k / 5
        out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT + pw}" y2="{y}" stroke="#ddd"/>')
        out.append(f'<text x="{LEFT - 6}" y="{y + 4}" text-anchor="end">{ymax * k / 5:g}</text>')
    out.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + ph}" stroke="black"/>')
    out.append(f'<line x1="{LEFT}" y1="{TOP + ph}" x2="{LEFT + pw}" y2="{TOP + ph}" stroke="black"/>')
    return out


def _legend(names: list[str]) -> list[str]:
    out = []
    for i, name in enumerate(names):
        y = TOP + 10 + 20 * i
        out.append(f'<rect x="{W - RIGHT + 15}" y="{y - 9}" width="12" height="12" fill="{COLOURS[i % len(COLOURS)]}"/>')
        out.append(f'<text x="{W - RIGHT + 32}" y="{y + 2}">{escape(name)}</text>')
    return out


def line_chart(path: Path, title: str, xlabel: str, ylabel: str, series: dict[str, list[tuple[float, float]]]) -> None:
    """series: name -> [(x, y), ...]. Axes start at 0, so differences are not exaggerated."""
    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts]
    xmax, ymax = _nice_max(max(xs)), _nice_max(max(ys))
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    out = _frame(title, xlabel, ylabel, ymax)
    for k in range(6):                                                   # x labels
        x = LEFT + pw * k / 5
        out.append(f'<text x="{x}" y="{TOP + ph + 18}" text-anchor="middle">{xmax * k / 5:g}</text>')
    for i, (name, pts) in enumerate(series.items()):
        coords = " ".join(f"{LEFT + pw * x / xmax:.1f},{TOP + ph - ph * y / ymax:.1f}" for x, y in pts)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{COLOURS[i % len(COLOURS)]}" stroke-width="2"/>')
    out += _legend(list(series)) + ["</svg>"]
    path.write_text("\n".join(out), encoding="utf-8")


def bar_chart(path: Path, title: str, ylabel: str, categories: list[str], groups: dict[str, list[float]]) -> None:
    """One cluster of bars per category; groups: name -> one value per category."""
    ymax = _nice_max(max(v for vals in groups.values() for v in vals))
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    out = _frame(title, "", ylabel, ymax)
    slot = pw / len(categories)
    bar = slot * 0.8 / len(groups)
    for c, cat in enumerate(categories):
        x0 = LEFT + slot * c + slot * 0.1
        out.append(f'<text x="{LEFT + slot * (c + 0.5)}" y="{TOP + ph + 18}" text-anchor="middle" font-size="11">'
                   f'{escape(cat)}</text>')
        for g, vals in enumerate(groups.values()):
            h = ph * vals[c] / ymax
            out.append(f'<rect x="{x0 + g * bar:.1f}" y="{TOP + ph - h:.1f}" width="{bar - 2:.1f}" height="{h:.1f}" '
                       f'fill="{COLOURS[g % len(COLOURS)]}"/>')
            out.append(f'<text x="{x0 + g * bar + bar / 2:.1f}" y="{TOP + ph - h - 4:.1f}" text-anchor="middle" '
                       f'font-size="10">{vals[c]:.3g}</text>')
    out += _legend(list(groups)) + ["</svg>"]
    path.write_text("\n".join(out), encoding="utf-8")
