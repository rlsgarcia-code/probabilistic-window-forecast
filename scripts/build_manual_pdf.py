"""Gera o manual matematico em PDF a partir de docs/manual_detalhado.md.

Dependencias de documentacao: reportlab e matplotlib.
O PDF final e gravado em output/pdf/.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "manual_detalhado.md"
TEMP = ROOT / "tmp" / "pdfs"
OUTPUT = ROOT / "output" / "pdf" / "manual_detalhado_modelagem_probabilistica.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1677A5")
TEAL = colors.HexColor("#0E8A80")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#EAF4F8")
PALE_TEAL = colors.HexColor("#E7F6F4")
LINE = colors.HexColor("#CBD5E1")
WHITE = colors.white


class ManualDocTemplate(BaseDocTemplate):
    """Documento com sumario automatico e cabecalho discreto."""

    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="content",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="manual", frames=frame, onPage=self._page))

    def _page(self, canvas, doc) -> None:
        page = canvas.getPageNumber()
        canvas.saveState()
        if page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
            canvas.setFillColor(TEAL)
            canvas.rect(0, A4[1] - 1.1 * cm, A4[0], 1.1 * cm, fill=1, stroke=0)
        else:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(self.leftMargin, A4[1] - 1.25 * cm, A4[0] - self.rightMargin, A4[1] - 1.25 * cm)
            canvas.setFillColor(MUTED)
            canvas.setFont("Helvetica", 8)
            canvas.drawString(self.leftMargin, A4[1] - 0.95 * cm, "PROBABILISTIC WINDOW FORECAST")
            canvas.drawRightString(A4[0] - self.rightMargin, 0.75 * cm, f"{page}")
            canvas.setStrokeColor(TEAL)
            canvas.setLineWidth(1.4)
            canvas.line(self.leftMargin, 1.02 * cm, self.leftMargin + 1.6 * cm, 1.02 * cm)
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in {"Heading1Manual", "Heading2Manual"}:
                level = 0 if style_name == "Heading1Manual" else 1
                text = flowable.getPlainText()
                key = f"heading-{self.seq.nextf('heading')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=34,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=16,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#D9EAF1"),
            spaceAfter=16,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#A8DADC"),
        ),
        "h1": ParagraphStyle(
            "Heading1Manual",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=15,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2Manual",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=BLUE,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyManual",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=14.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "quote": ParagraphStyle(
            "QuoteManual",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.1,
            leading=14,
            textColor=NAVY,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=0,
        ),
        "code": ParagraphStyle(
            "CodeManual",
            fontName="Courier",
            fontSize=7.5,
            leading=10.1,
            textColor=colors.HexColor("#16324F"),
            backColor=colors.HexColor("#EDF6FA"),
            borderPadding=9,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "caption": ParagraphStyle(
            "CaptionManual",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=9,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=NAVY,
            spaceAfter=14,
        ),
    }


def inline_markup(text: str) -> str:
    value = html.escape(text.strip())
    value = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#0E7490">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


def _mathtext_formula(formula: str) -> str:
    value = " ".join(line.strip() for line in formula.splitlines() if line.strip())
    value = value.replace("\\mathrm{mistura\\ preditiva}", r"\mathrm{mistura\ preditiva}")
    value = value.replace("\\mathrm{revisao\\ se\\ material}", r"\mathrm{revisao\ se\ material}")
    value = re.sub(r"\\mathbf\s*([A-Za-z])", r"\\mathbf{\1}", value)
    value = re.sub(r"\\boldsymbol\s*(\\[A-Za-z]+|[A-Za-z])", r"\\boldsymbol{\1}", value)
    return value


def _plain_formula(formula: str) -> str:
    value = formula
    replacements = {
        r"\alpha": "α",
        r"\beta": "β",
        r"\mu": "μ",
        r"\tau": "τ",
        r"\pi": "π",
        r"\sigma": "σ",
        r"\kappa": "κ",
        r"\Gamma": "Γ",
        r"\widehat": "",
        r"\hat": "",
        r"\boldsymbol": "",
        r"\mathbf": "",
        r"\operatorname": "",
        r"\mathrm": "",
        r"\left": "",
        r"\right": "",
        r"\quad": "   ",
        r"\ldots": "...",
        r"\varnothing": "vazio",
        r"\rightarrow": " -> ",
        r"\mid": " | ",
        r"\sim": " ~ ",
        r"\geq": " >= ",
        r"\leq": " <= ",
        r"\in": " em ",
        r"\frac": "frac",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("^T", "^(T)")
    return " ".join(value.split())


def _render_array_formula(formula: str, path: Path) -> None:
    pattern = re.compile(r"\\left\[\\begin\{array\}\{[^}]+\}(.*?)\\end\{array\}\\right\]", re.S)
    match = pattern.search(formula)
    if not match:
        raise ValueError("array nao reconhecido")
    prefix_raw = formula[: match.start()].strip()
    suffix_raw = formula[match.end() :].strip()
    prefix = _mathtext_formula(prefix_raw) if prefix_raw else ""
    suffix = _mathtext_formula(suffix_raw) if suffix_raw else ""
    rows = [row.strip() for row in re.split(r"\\\\", match.group(1)) if row.strip()]
    matrix = [[cell.strip() for cell in row.split("&")] for row in rows]
    cols = max(len(row) for row in matrix)
    fig_w = min(10.5, 2.4 + cols * 0.75 + len(prefix) * 0.10)
    fig_h = max(0.8, 0.38 * len(matrix) + 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    left = 0.20 + min(0.22, len(prefix) * 0.008)
    right = min(0.92, left + 0.11 * cols + 0.06)
    if prefix:
        ax.text(left - 0.04, 0.5, f"${prefix}$", ha="right", va="center", fontsize=15)
    if suffix:
        ax.text(right + 0.04, 0.5, f"${suffix}$", ha="left", va="center", fontsize=15)
    for r, row in enumerate(matrix):
        y = 1 - (r + 0.5) / len(matrix)
        for c, cell in enumerate(row):
            x = left + (c + 0.5) * (right - left) / cols
            ax.text(x, y, _plain_formula(cell), ha="center", va="center", fontsize=14, family="STIXGeneral")
    hook = 0.014
    ax.plot([left, left, left + hook], [0.94, 0.06, 0.06], color="#243B53", lw=1.3)
    ax.plot([left, left + hook], [0.94, 0.94], color="#243B53", lw=1.3)
    ax.plot([right, right, right - hook], [0.94, 0.06, 0.06], color="#243B53", lw=1.3)
    ax.plot([right, right - hook], [0.94, 0.94], color="#243B53", lw=1.3)
    fig.savefig(path, dpi=210, bbox_inches="tight", transparent=True, pad_inches=0.05)
    plt.close(fig)


def render_equation(formula: str, index: int) -> Path:
    path = TEMP / f"equation_{index:03d}.png"
    if "\\begin{array}" in formula:
        _render_array_formula(formula, path)
        return path
    transformed = _mathtext_formula(formula)
    source_lines = [
        _mathtext_formula(line).replace(r"\left", "").replace(r"\right", "")
        for line in formula.splitlines()
        if line.strip()
    ]
    if len(transformed) > 105 and len(source_lines) > 1:
        pieces: list[str] = []
        for line in source_lines:
            if line in {"(", "[", r"\{"} and pieces:
                pieces[-1] += line
            elif line in {")", "]", r"\}"} and pieces:
                pieces[-1] += line
            else:
                pieces.append(line)
        logical_lines: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > 92:
                logical_lines.append(current)
                current = piece
            else:
                current = candidate
        if current:
            logical_lines.append(current)
        fig, ax = plt.subplots(figsize=(9.4, max(0.72, 0.42 * len(logical_lines))))
        ax.set_axis_off()
        for row, line in enumerate(logical_lines):
            y = 1.0 - (row + 0.5) / len(logical_lines)
            try:
                ax.text(0.5, y, f"${line}$", ha="center", va="center", fontsize=13.2, color="#102A43")
            except Exception:
                ax.text(0.5, y, _plain_formula(line), ha="center", va="center", fontsize=11.5, family="STIXGeneral", color="#102A43")
        try:
            fig.canvas.draw()
        except Exception:
            plt.close(fig)
        else:
            fig.savefig(path, dpi=210, bbox_inches="tight", transparent=True, pad_inches=0.06)
            plt.close(fig)
            return path
    fontsize = 14 if len(transformed) < 105 else 11.5
    width = min(11.0, max(4.0, 1.2 + 0.070 * len(transformed)))
    fig, ax = plt.subplots(figsize=(width, 0.58 if len(transformed) < 130 else 0.72))
    ax.set_axis_off()
    try:
        ax.text(
            0.5,
            0.5,
            f"${transformed}$",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#102A43",
        )
        fig.canvas.draw()
    except Exception:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(width, 0.68))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            _plain_formula(formula),
            ha="center",
            va="center",
            fontsize=11.5,
            family="STIXGeneral",
            color="#102A43",
            wrap=True,
        )
    fig.savefig(path, dpi=210, bbox_inches="tight", transparent=True, pad_inches=0.06)
    plt.close(fig)
    return path


def equation_flowable(path: Path, max_width: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def make_figures() -> None:
    TEMP.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.2, 2.4))
    ax.set_axis_off()
    labels = ["Série\nmensal", "Hipóteses\nde janela", "Modelos\nlocais", "Posterior e\nevidência", "Mistura\npreditiva", "Revisão\nse material"]
    xs = np.linspace(0.07, 0.93, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        color = "#0E8A80" if i in {0, 5} else "#1677A5"
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.65", facecolor=color, edgecolor="none"),
        )
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.075, 0.5), xytext=(x + 0.075, 0.5), arrowprops=dict(arrowstyle="->", color="#627D98", lw=1.6))
    fig.savefig(TEMP / "manual_fluxo.png", dpi=210, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)

    rng = np.random.default_rng(42)
    t = np.arange(48)
    y = np.r_[40 + 0.5 * np.arange(15), 53 - 0.25 * np.arange(17), 47 + 0.85 * np.arange(16)]
    y = y + rng.normal(0, 0.75, len(y))
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.plot(t, y, color="#102A43", lw=1.5, marker="o", ms=3.2, label="observado")
    ax.axvspan(32, 47.6, color="#A8DADC", alpha=0.34, label="regime terminal")
    ax.axvline(15, color="#94A3B8", ls="--", lw=1)
    ax.axvline(32, color="#0E8A80", ls="--", lw=1.6)
    ax.text(32.4, min(y) + 0.5, "início candidato do regime atual", color="#0E8A80", fontsize=9)
    ax.set_xlabel("mês dentro do lookback")
    ax.set_ylabel("valor")
    ax.grid(axis="y", color="#E2E8F0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(TEMP / "manual_regimes.png", dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    horizons = np.arange(1, 7)
    within = np.array([1.0, 1.15, 1.35, 1.55, 1.82, 2.10])
    between = np.array([0.25, 0.38, 0.58, 0.83, 1.12, 1.48])
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    ax.bar(horizons, within, color="#1677A5", label="dentro: ruído + coeficientes")
    ax.bar(horizons, between, bottom=within, color="#0E8A80", label="entre: janela + modelo")
    ax.set_xlabel("horizonte mensal")
    ax.set_ylabel("variância total")
    ax.set_xticks(horizons)
    ax.grid(axis="y", color="#E2E8F0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(TEMP / "manual_incerteza.png", dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_table(lines: list[str], style_map: dict) -> Table:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append([Paragraph(inline_markup(cell), style_map["body"]) for cell in cells])
    cols = len(rows[0])
    widths = [17.0 * cm / cols] * cols
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_story(source: str, style_map: dict, doc_width: float):
    lines = source.splitlines()
    story = []

    title = lines[0].removeprefix("# ").strip()
    subtitle = lines[2].removeprefix("## ").strip()
    version = lines[4].strip()
    story.extend(
        [
            Spacer(1, 5.3 * cm),
            Paragraph(inline_markup(title), style_map["cover_title"]),
            Paragraph(inline_markup(subtitle), style_map["cover_subtitle"]),
            Spacer(1, 0.6 * cm),
            Table([[" "]], colWidths=[2.1 * cm], rowHeights=[0.08 * cm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEAL)])),
            Spacer(1, 0.55 * cm),
            Paragraph(f"{inline_markup(version)}<br/>Manual técnico e conceitual<br/>probabilistic-window-forecast", style_map["cover_meta"]),
            PageBreak(),
            Paragraph("Sumário", style_map["toc_title"]),
        ]
    )
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOC1", fontName="Helvetica-Bold", fontSize=9.4, leading=13, leftIndent=0, firstLineIndent=0, textColor=NAVY, spaceBefore=3),
        ParagraphStyle(name="TOC2", fontName="Helvetica", fontSize=8.5, leading=11.5, leftIndent=14, firstLineIndent=0, textColor=INK),
    ]
    story.extend([toc, PageBreak()])

    i = 5
    equation_index = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(item.strip() for item in paragraph).strip()
            if text:
                story.append(Paragraph(inline_markup(text), style_map["body"]))
            paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped == "$$":
            flush_paragraph()
            formula_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                formula_lines.append(lines[i])
                i += 1
            formula = "\n".join(formula_lines)
            path = render_equation(formula, equation_index)
            equation_index += 1
            story.extend([Spacer(1, 2), equation_flowable(path, doc_width * 0.94), Spacer(1, 6)])
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code_lines), style_map["code"], maxLineLength=100))
            i += 1
            continue

        if stripped.startswith("!["):
            flush_paragraph()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                caption, raw_path = match.groups()
                image_path = (SOURCE.parent / raw_path).resolve()
                image = Image(str(image_path))
                scale = min(doc_width / image.imageWidth, 1.0)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                image.hAlign = "CENTER"
                story.append(KeepTogether([image, Paragraph(inline_markup(caption), style_map["caption"])]))
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, style_map))
            story.append(Spacer(1, 8))
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().removeprefix(">").strip())
                i += 1
            quote = Paragraph(inline_markup(" ".join(quote_lines)), style_map["quote"])
            box = Table([[quote]], colWidths=[doc_width - 10], hAlign="LEFT")
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL), ("BOX", (0, 0), (-1, -1), 0.7, TEAL), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
            story.extend([box, Spacer(1, 7)])
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), style_map["h1"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), style_map["h2"]))
            i += 1
            continue

        if re.match(r"^-\s+", stripped):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r"^-\s+", lines[i].strip()):
                item = re.sub(r"^-\s+", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_markup(item), style_map["body"]), leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=18, bulletFontName="Helvetica", bulletColor=TEAL, spaceAfter=5))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_markup(item), style_map["body"]), leftIndent=14))
                i += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=22, bulletColor=BLUE, spaceAfter=5))
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return story


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    make_figures()
    doc = ManualDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.45 * cm,
        title="Manual detalhado da modelagem probabilística de janela e forecast",
        author="probabilistic-window-forecast",
        subject="Derivação matemática, algoritmo, incerteza e uso do pacote Python",
    )
    style_map = styles()
    story = build_story(SOURCE.read_text(encoding="utf-8"), style_map, doc.width)
    doc.multiBuild(story)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
