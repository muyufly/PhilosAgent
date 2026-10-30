"""Build the reviewed stage report as a readable, locally linked PDF.

Reads the Markdown source afresh on every invocation. No model calls are made.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/PhiloAgent阶段性报告-20261030.md"
OUTPUT = ROOT / "output/pdf/PhiloAgent阶段性报告-20261030.pdf"
RENDER = ROOT / "tmp/pdfs/stage-report-20261030"
POPPLER = Path(
    r"C:\Users\muyufly\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\native\poppler\Library\bin\pdftoppm.exe"
)

FONT = "StageReportCJK"
BOLD = "StageReportCJKBold"
PAGE_W, PAGE_H = A4
LEFT = RIGHT = 21 * mm
TOP = 21 * mm
BOTTOM = 20 * mm
WIDTH = PAGE_W - LEFT - RIGHT
INK = colors.HexColor("#161616")
MUTED = colors.HexColor("#666666")
RULE = colors.HexColor("#D8D8D8")


def normalize_dashes(text: str) -> str:
    return re.sub("[\u2010-\u2015\u2212]", "-", text)


def link_target(target: str) -> str:
    if re.match(r"^[a-zA-Z][\w+.-]*:", target):
        return target
    return (SOURCE.parent / target).resolve().as_uri()


INLINE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`")


def inline(text: str) -> str:
    text = normalize_dashes(text)
    result = []
    position = 0
    for match in INLINE.finditer(text):
        result.append(html.escape(text[position:match.start()]))
        label, target, strong, code = match.groups()
        if label is not None:
            destination = html.escape(link_target(target), quote=True)
            result.append(f'<link href="{destination}" color="#202020"><u>{inline(label)}</u></link>')
        elif strong is not None:
            result.append(f"<b>{inline(strong)}</b>")
        else:
            result.append(html.escape(code))
        position = match.end()
    result.append(html.escape(text[position:]))
    return "".join(result)


def plain(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return normalize_dashes(text.replace("**", "").replace("`", ""))


def styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName=FONT, textColor=INK, wordWrap="CJK", allowWidows=0, allowOrphans=0)
    return {
        "title": ParagraphStyle("ReportTitle", **base, fontSize=22, leading=30,
                                fontNameOverride=BOLD, spaceAfter=7, keepWithNext=True),
        "subtitle": ParagraphStyle("ReportSubtitle", **base, fontSize=11.6, leading=18,
                                   spaceAfter=12, keepWithNext=True),
        "metadata": ParagraphStyle("ReportMetadata", **base, fontSize=9, leading=14,
                                   textColorOverride=MUTED, spaceAfter=2, keepWithNext=True),
        "h2": ParagraphStyle("ReportSection", **base, fontSize=14.2, leading=21,
                             fontNameOverride=BOLD, spaceBefore=14, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("ReportSubsection", **base, fontSize=11.4, leading=18,
                             fontNameOverride=BOLD, spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("ReportBody", **base, fontSize=11, leading=17,
                               alignment=TA_LEFT, spaceAfter=6),
        "table": ParagraphStyle("ReportTable", **base, fontSize=9.8, leading=15.1,
                                alignment=TA_LEFT, spaceAfter=0),
        "table_number": ParagraphStyle("ReportTableNumber", **base, fontSize=9.8, leading=15.1,
                                       alignment=TA_RIGHT, spaceAfter=0),
        "table_header": ParagraphStyle("ReportTableHeader", **base, fontSize=9.6, leading=14.6,
                                       fontNameOverride=BOLD, spaceAfter=0),
        "reference": ParagraphStyle("ReportReference", **base, fontSize=9.2, leading=14.8,
                                    spaceAfter=7, leftIndent=15, firstLineIndent=-15),
    }


class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.saved_pages = []

    def showPage(self):
        self.saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.saved_pages)
        for state in self.saved_pages:
            self.__dict__.update(state)
            self.saveState()
            self.setStrokeColor(RULE)
            self.setLineWidth(0.4)
            self.line(LEFT, PAGE_H - 14.5 * mm, PAGE_W - RIGHT, PAGE_H - 14.5 * mm)
            self.setFillColor(MUTED)
            self.setFont(FONT, 8)
            self.drawString(LEFT, PAGE_H - 12 * mm, "PhiloAgent  |  阶段性研究报告")
            self.drawRightString(PAGE_W - RIGHT, PAGE_H - 12 * mm, "2026.10.30")
            self.drawString(LEFT, 12 * mm, "方法与系统原型 · 阶段性实验结果")
            self.drawRightString(PAGE_W - RIGHT, 12 * mm, f"{self._pageNumber} / {total}")
            self.restoreState()
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


def table_widths(header: list[str]) -> list[float]:
    if header[0] == "组别":
        fractions = (0.25, 0.30, 0.23, 0.22)
    elif len(header) == 5 and header[0] == "领域与组别":
        fractions = (0.26, 0.185, 0.185, 0.185, 0.185)
    elif len(header) == 5 and header[0] == "领域":
        fractions = (0.14, 0.26, 0.20, 0.20, 0.20)
    elif len(header) == 4 and header[0] == "领域":
        fractions = (0.16, 0.26, 0.34, 0.24)
    elif header[0] == "项目":
        fractions = (0.40, 0.23, 0.37)
    elif len(header) == 4:
        fractions = (0.46, 0.18, 0.18, 0.18)
    elif header[0] == "人文质量维度":
        fractions = (0.50, 0.25, 0.25)
    elif len(header) == 3:
        fractions = (0.50, 0.25, 0.25)
    else:
        fractions = tuple(1 / len(header) for _ in header)
    return [WIDTH * item for item in fractions]


def make_story(source: str):
    ss = styles()
    for style in ss.values():
        if getattr(style, "fontNameOverride", None):
            style.fontName = style.fontNameOverride
        if getattr(style, "textColorOverride", None):
            style.textColor = style.textColorOverride
    story, checked_fragments = [], []
    lines = source.splitlines()
    index = 0
    appendix = False
    prelude = True

    def paragraph(value: str, name: str):
        story.append(Paragraph(inline(value), ss[name]))
        checked_fragments.append(plain(value))

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            paragraph(line[2:], "title")
            index += 1
            continue
        if line.startswith("## "):
            prelude = False
            appendix = line.startswith("## 附录")
            if appendix:
                story.append(PageBreak())
            paragraph(line[3:], "h2")
            index += 1
            continue
        if line.startswith("### "):
            paragraph(line[4:], "h3")
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [item.strip() for item in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", item) for item in cells):
                    rows.append(cells)
                index += 1
            cells = []
            for row_index, row in enumerate(rows):
                cell_row = []
                for col_index, value in enumerate(row):
                    style = "table_header" if row_index == 0 else (
                        "table_number" if col_index and rows[0][0] != "组别" else "table")
                    cell_row.append(Paragraph(inline(value), ss[style]))
                    checked_fragments.append(plain(value))
                cells.append(cell_row)
            table = Table(cells, colWidths=table_widths(rows[0]), repeatRows=1, splitByRow=0,
                          hAlign="LEFT", spaceBefore=4, spaceAfter=11)
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F1F1")),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
                ("LINEBELOW", (0, -1), (-1, -1), 0.65, INK),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            if story and isinstance(story[-1], Paragraph):
                story[-1].keepWithNext = True
            story.append(table)
            continue
        if prelude:
            name = "metadata" if line.startswith(("报告日期：", "数据截止：")) else "subtitle"
            paragraph(line, name)
            index += 1
            continue
        block = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].lstrip().startswith(("#", "|")):
            if re.match(r"\d+\. ", lines[index].strip()):
                break
            block.append(lines[index].strip())
            index += 1
        paragraph(" ".join(block), "reference" if appendix else "body")
    return story, checked_fragments


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", normalize_dashes(text))


def main():
    raw = SOURCE.read_bytes()
    source = raw.decode("utf-8-sig")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RENDER.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont(FONT, r"C:\Windows\Fonts\msyh.ttc"))
    pdfmetrics.registerFont(TTFont(BOLD, r"C:\Windows\Fonts\msyhbd.ttc"))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=BOLD, italic=FONT, boldItalic=BOLD)
    story, fragments = make_story(source)
    document = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                               topMargin=TOP, bottomMargin=BOTTOM,
                               title="PhiloAgent 阶段性研究报告", author="PhiloAgent 项目组",
                               subject="可修订哲学状态与开放任务实验，2026年10月30日")
    frame = Frame(LEFT, BOTTOM, WIDTH, PAGE_H - TOP - BOTTOM,
                  leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
    document.addPageTemplates(PageTemplate(id="report", frames=frame))
    document.build(story, canvasmaker=ReportCanvas)
    reader = PdfReader(str(OUTPUT))
    extracted = "\n".join((page.extract_text() or "").rsplit("PhiloAgent  |  阶段性研究报告", 1)[0]
                           for page in reader.pages)
    text = normalized(extracted)
    missing = [fragment for fragment in fragments if normalized(fragment) not in text]
    if missing:
        raise RuntimeError("PDF text validation failed: " + repr(missing[:5]))
    # Remove only this renderer's old page images before Poppler regenerates them.
    for old_page in RENDER.glob("page-*.png"):
        if old_page.parent.resolve() == RENDER.resolve() and re.fullmatch(r"page-\d+\.png", old_page.name):
            old_page.unlink()
    subprocess.run([str(POPPLER), "-r", "125", "-png", str(OUTPUT), str(RENDER / "page")], check=True)
    digits = len(str(len(reader.pages)))
    pages = [str(RENDER / f"page-{index:0{digits}d}.png") for index in range(1, len(reader.pages) + 1)]
    if not all(Path(path).is_file() for path in pages):
        raise RuntimeError("Expected rendered page files were not produced")
    annotations = sum(len(page.get("/Annots", [])) for page in reader.pages)
    manifest = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "pdf": str(OUTPUT), "pdf_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "page_count": len(reader.pages), "checked_text_fragments": len(fragments),
        "missing_text_fragments": missing, "link_annotations": annotations,
        "rendered_pages": pages,
        "typography": {"body_points": 11, "body_leading": 17, "reference_points": 9.2},
    }
    (RENDER / "render-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
