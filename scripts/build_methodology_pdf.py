from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem,
    PageTemplate, Paragraph, PageBreak, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "docs" / "Mickey_Mouse_Scientific_Methodology.html"
PDF_PATH = ROOT / "docs" / "Mickey_Mouse_Scientific_Methodology.pdf"
OVERVIEW = ROOT / "docs" / "assets" / "heldout_varied_seeds_overview.png"


class ReportParser(HTMLParser):
    BLOCKS = {"h1", "h2", "h3", "p", "li", "td", "th"}

    def __init__(self):
        super().__init__()
        self.blocks = []
        self.tag = None
        self.text = []
        self.list_type = None
        self.table = None
        self.row = None
        self.in_figure = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.BLOCKS:
            self.tag = tag
            self.text = []
        if tag in ("ul", "ol"):
            self.list_type = tag
            self.blocks.append(("list_start", tag))
        elif tag == "table":
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag == "div" and "page-break" in attrs.get("class", ""):
            self.blocks.append(("page_break", ""))
        elif tag == "div" and "figure" in attrs.get("class", ""):
            self.in_figure = True
        elif tag == "img" and self.in_figure:
            self.blocks.append(("figure", attrs.get("alt", "Results overview")))
        elif tag == "a":
            self.text.append(f'<a href="{attrs.get("href", "")}">')
        elif tag == "code":
            self.text.append("<font name=\"Courier\">")
        elif tag in ("strong", "b"):
            self.text.append("<b>")
        elif tag in ("em", "i"):
            self.text.append("<i>")

    def handle_endtag(self, tag):
        if tag in self.BLOCKS and self.tag == tag:
            value = " ".join("".join(self.text).split())
            if tag in ("td", "th") and self.row is not None:
                self.row.append((value, tag == "th"))
            elif tag == "li":
                self.blocks.append(("list_item", value))
            elif value:
                self.blocks.append((tag, value))
            self.tag = None
            self.text = []
        if tag in ("ul", "ol"):
            self.blocks.append(("list_end", tag))
            self.list_type = None
        elif tag == "tr" and self.table is not None and self.row:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table:
            self.blocks.append(("table", self.table))
            self.table = None
        elif tag == "div" and self.in_figure:
            self.in_figure = False
        elif tag == "a":
            self.text.append("</a>")
        elif tag == "code":
            self.text.append("</font>")
        elif tag in ("strong", "b"):
            self.text.append("</b>")
        elif tag in ("em", "i"):
            self.text.append("</i>")

    def handle_data(self, data):
        if self.tag:
            self.text.append(unescape(data))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.65 * inch, 0.38 * inch, "Context-Conditioned Audio-to-Album-Cover Generation")
    canvas.drawRightString(letter[0] - 0.65 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    parser = ReportParser()
    parser.feed(HTML_PATH.read_text(encoding="utf-8"))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=7))
    styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontSize=15, leading=18, spaceBefore=13, spaceAfter=6, textColor=colors.HexColor("#1f2933")))
    styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=9.3, leading=13, spaceAfter=5))
    styles.add(ParagraphStyle(name="CaptionX", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#555555"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="CellX", parent=styles["BodyText"], fontSize=8.2, leading=10))

    doc = BaseDocTemplate(str(PDF_PATH), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.62 * inch, bottomMargin=0.58 * inch, title="Context-Conditioned Audio-to-Album-Cover Generation")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=footer))

    story = []
    current_list = None
    list_kind = None

    def flush_list():
        nonlocal current_list, list_kind
        if current_list:
            bullet = "1" if list_kind == "ol" else "bullet"
            story.append(ListFlowable(current_list, bulletType=bullet, leftIndent=18, bulletFontSize=8, spaceAfter=6))
        current_list = None
        list_kind = None

    for kind, value in parser.blocks:
        if kind == "list_start":
            flush_list()
            current_list = []
            list_kind = value
        elif kind == "list_item":
            if current_list is None:
                current_list = []
                list_kind = "ul"
            current_list.append(ListItem(Paragraph(value, styles["BodyX"]), leftIndent=8))
        elif kind == "list_end":
            flush_list()
        elif kind == "h1":
            flush_list()
            story.append(Paragraph(value, styles["TitleX"]))
        elif kind == "h2":
            flush_list()
            story.append(Paragraph(value, styles["H1X"]))
        elif kind == "h3":
            flush_list()
            story.append(Paragraph(value, styles["H2X"]))
        elif kind == "p":
            flush_list()
            story.append(Paragraph(value, styles["BodyX"]))
        elif kind == "table":
            flush_list()
            rows = []
            for row in value:
                rows.append([Paragraph(cell, styles["CellX"]) for cell, _ in row])
            table = Table(rows, colWidths=[doc.width / len(rows[0])] * len(rows[0]), repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa3ab")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 6)])
        elif kind == "page_break":
            flush_list()
            story.append(PageBreak())
        elif kind == "figure":
            flush_list()
            image = Image(str(OVERVIEW))
            image.drawWidth = doc.width
            image.drawHeight = doc.width * 1050 / 2304
            story.append(KeepTogether([image, Spacer(1, 4), Paragraph("Figure 1. Held-out songs generated with a distinct seed per song. Within each song, style, biography, and combined modes use the same seed.", styles["CaptionX"])]))
    flush_list()
    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    build()
