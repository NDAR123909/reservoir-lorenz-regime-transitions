"""Render paper/06_progress_log.md to a PDF, embedding Figures 2-4."""
import os, re, glob
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
FONTD = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONTD, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONTD, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(FONTD, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="DejaVu", fontSize=9.3,
                      leading=13.6, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DejaVu-Bold", fontSize=14,
                    leading=17, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2c3e50"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=11.3,
                    leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#16403a"))
title = ParagraphStyle("title", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=17,
                       leading=21, spaceAfter=8, textColor=colors.HexColor("#2c3e50"))
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)


def inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r'<font name="DejaVu-Bold">\1</font>', s)
    s = re.sub(r"`(.+?)`", r'<font name="DejaVuMono" size="8.4">\1</font>', s)
    return s


FIG_AFTER = {  # insert a figure once we pass the table under each contribution
    "C2": ("fig2_c2_range.png", "Figure 2. C2 — extrapolation distance vs. training-range width."),
    "C3": ("fig3_c3_density.png", "Figure 3. C3 — extrapolation distance vs. sample density."),
    "C4": ("fig4_c4_position.png", "Figure 4. C4 — across-Hopf behaviour vs. window position."),
}
cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10,
                     textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=10,
                     alignment=1)


def fig_flowables(png, caption, maxw):
    path = os.path.join(ROOT, "figures", png)
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = maxw; h = w * ih / iw
    return [Spacer(1, 4), Image(path, width=w, height=h), Paragraph(caption, cap)]


def build():
    md = open(os.path.join(ROOT, "paper", "06_progress_log.md")).read().splitlines()
    doc = SimpleDocTemplate(os.path.join(ROOT, "paper", "06_progress_log.pdf"),
                            pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch,
                            leftMargin=0.8*inch, rightMargin=0.8*inch,
                            title="Progress log - Session 7", author="Noah Riego")
    maxw = letter[0] - 1.6*inch
    story = []
    i = 0
    pending_fig = None
    while i < len(md):
        ln = md[i].rstrip()
        if not ln.strip():
            i += 1; continue
        # table block
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(md) and md[i].lstrip().startswith("|"):
                cells = [c.strip() for c in md[i].strip().strip("|").split("|")]
                if not re.match(r"^[-:\s|]+$", md[i].strip().strip("|")):
                    rows.append(cells)
                i += 1
            data = [[Paragraph(inline(c), ParagraphStyle("tc", parent=body, fontSize=8.2,
                     leading=10.5, spaceAfter=0)) for c in r] for r in rows]
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0,0), (-1,-1), "DejaVu"),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eef2f1")),
                ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#16403a")),
                ("LINEBELOW", (0,1), (-1,-1), 0.25, colors.HexColor("#cccccc")),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ]))
            story.append(tbl); story.append(Spacer(1, 6))
            if pending_fig:
                story += fig_flowables(*pending_fig, maxw); pending_fig = None
            continue
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), title))
        elif ln.startswith("## "):
            htxt = ln[3:]
            story.append(Paragraph(inline(htxt), h1))
            for key, (png, capt) in FIG_AFTER.items():
                if htxt.startswith(key + " ") or htxt.startswith(key + "\u2014") or f"## {key} " in "## "+htxt:
                    pending_fig = (png, capt)
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), h2))
        elif ln.startswith("- "):
            story.append(Paragraph(inline(ln[2:]), bullet, bulletText="\u2022"))
        else:
            story.append(Paragraph(inline(ln), body))
        i += 1
    doc.build(story)
    print("wrote paper/06_progress_log.pdf")


if __name__ == "__main__":
    build()
