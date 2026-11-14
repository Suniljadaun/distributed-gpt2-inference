"""
test_ch16.py - Chapter 16
The reporting tools are code too: charts must be valid SVG with one mark per data point,
and summary tables must be valid Markdown. (The experiments themselves are tested in Ch 3-15.)
Run:  python -m pytest tests/test_ch16.py -v
"""
import xml.etree.ElementTree as ET

from src.bench.run_all import table
from src.bench.svg import _nice_max, bar_chart, line_chart

NS = "{http://www.w3.org/2000/svg}"


def test_axis_maximum_is_rounded_up_nicely() -> None:
    assert [_nice_max(v) for v in (0.7, 3, 17, 48, 130, 0)] == [1, 5, 20, 50, 200, 1]


def test_line_chart_has_one_line_per_series(tmp_path) -> None:
    path = tmp_path / "line.svg"
    line_chart(path, "title", "x", "y", {"a": [(0, 1), (1, 2), (2, 3)], "b": [(0, 3), (2, 1)]})
    root = ET.parse(path).getroot()                                  # parses = well-formed XML
    lines = root.findall(f"{NS}polyline")
    assert len(lines) == 2 and len(lines[0].get("points").split()) == 3


def test_bar_chart_has_one_bar_per_value(tmp_path) -> None:
    path = tmp_path / "bars.svg"
    bar_chart(path, "title", "ms", ["x", "y", "z"], {"A": [1, 2, 3], "B": [3, 2, 1]})
    root = ET.parse(path).getroot()
    bars = [r for r in root.findall(f"{NS}rect") if r.get("width") not in ("720", "12")]   # not background/legend
    assert len(bars) == 6
    assert all(float(r.get("height")) > 0 for r in bars)


def test_markdown_table() -> None:
    md = table([{"split": "3/3", "A_ms": 8.0}, {"split": "4/2", "A_ms": 10.4}])
    assert md.splitlines() == ["| split | A_ms |", "|---|---|", "| 3/3 | 8.0 |", "| 4/2 | 10.4 |"]
