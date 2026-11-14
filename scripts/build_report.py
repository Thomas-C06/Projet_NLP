"""Render the project documentation as a simple PDF report."""

from html import escape
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

from backend.config import ROOT


def main():
    output = ROOT / "output/pdf/project-documentation.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=23,
                              leading=28, textColor=colors.black, spaceAfter=20))
    styles.add(ParagraphStyle(name="ReportHeading", fontName="Helvetica-Bold", fontSize=12,
                              leading=16, textColor=colors.black, spaceBefore=15, spaceAfter=7,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=10,
                              leading=15, alignment=TA_LEFT, spaceAfter=9))
    story = []
    paragraphs = (ROOT / "docs/report.md").read_text(encoding="utf-8").split("\n\n")
    for paragraph in paragraphs:
        text = paragraph.strip()
        if not text:
            continue
        if text == "---":
            story.append(PageBreak())
            continue
        style = "ReportTitle" if text.startswith("# ") else "ReportHeading" if text.startswith("## ") else "ReportBody"
        text = re.sub(r"^#{1,2} ", "", text)
        # Escape source characters before passing the text to the PDF paragraph parser.
        text = escape(" ".join(text.splitlines()))
        story.append(Paragraph(text, styles[style]))

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667066"))
        canvas.drawString(48, 28, "Movie Match | ING5 NLP")
        canvas.drawRightString(A4[0] - 48, 28, str(document.page))

    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
                                 topMargin=45, bottomMargin=48, title="Movie Match project documentation",
                                 author="Project group")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()
