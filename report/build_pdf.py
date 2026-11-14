"""
build_pdf.py - turn Final_Report_draft.md into Final_Report.pdf (A4).
Markdown -> HTML (the `markdown` package) with the SVG figures inlined and made scalable,
then printed to PDF by Microsoft Edge in headless mode (no LaTeX, no pandoc needed).
Run from the project root:   python report/build_pdf.py
Needs:  pip install markdown   (pure Python)
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Final_Report_draft.md"
HTML = HERE / "Final_Report.html"
PDF = HERE / "Final_Report.pdf"
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.3pt; line-height: 1.42; color: #111; }
h1 { font-size: 19pt; margin: 0 0 4pt 0; line-height: 1.2; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt 0; border-bottom: 1px solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11.2pt; margin: 11pt 0 3pt 0; break-after: avoid; }
p { margin: 4pt 0 6pt 0; text-align: justify; }
ul, ol { margin: 3pt 0 6pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
hr { border: 0; border-top: 1px solid #bbb; margin: 8pt 0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f3f4f6; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f3f4f6; padding: 6pt 8pt; border-radius: 3pt; font-size: 8.6pt; line-height: 1.3; break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 5pt 0 8pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #c7c7c7; padding: 2.5pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef2f7; }
.figure { margin: 8pt 0 2pt 0; text-align: center; break-inside: avoid; }
.figure svg { width: 100%; height: auto; max-height: 105mm; }
.figure img { width: 88%; height: auto; }
.figure + p em, p.caption { font-size: 9pt; color: #333; }
"""


def inline_svg(path: Path) -> str:
    """The SVG's markup, scalable: viewBox from its width/height, fixed size removed (CSS sizes it)."""
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    head = re.search(r"<svg\b[^>]*>", svg).group(0)
    new = head
    if "viewBox" not in head:
        w = re.search(r'\bwidth="([\d.]+)', head).group(1)
        h = re.search(r'\bheight="([\d.]+)', head).group(1)
        new = new.replace("<svg", f'<svg viewBox="0 0 {w} {h}"', 1)
    new = re.sub(r'\s(width|height)="[^"]*"', "", new)
    return svg.replace(head, new, 1)


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def github_style_lists(text: str) -> str:
    """GitHub starts a list right after a paragraph line; Python-Markdown needs a blank line first."""
    out, in_code = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        prev = out[-1] if out else ""
        if (not in_code and LIST_ITEM.match(line) and prev.strip() and not LIST_ITEM.match(prev)
                and not prev.startswith(" ")):
            out.append("")
        out.append(line)
    return "\n".join(out)


def build_html() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    text = github_style_lists(text)
    text = re.sub(r"^(\*\*Final Report[^\n]*\*\*)\n", r"\1<br>\n", text, flags=re.M)   # title / author on 2 lines
    text = re.sub(r"^\*\(If the course requires it.*\)\*\s*$", "", text, flags=re.M)   # drop the placeholder note

    def figure(m: re.Match) -> str:
        path = (HERE / m.group(2)).resolve()
        if path.suffix == ".svg":
            return f'\n<div class="figure">{inline_svg(path)}</div>\n'
        return f'\n<div class="figure"><img src="{path.as_uri()}" alt="{m.group(1)}"></div>\n'   # PNG (Matplotlib)

    text = re.sub(r"^!\[([^\]]*)\]\(([^)]+\.(?:svg|png))\)\s*$", figure, text, flags=re.M)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>Final Report - Project 55</title><style>{CSS}</style></head><body>{body}</body></html>")


def main() -> None:
    HTML.write_text(build_html(), encoding="utf-8")
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        sys.exit(f"Microsoft Edge not found; open {HTML} in a browser and print it to PDF")
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
