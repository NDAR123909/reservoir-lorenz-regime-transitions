"""Render paper/07_progress_log.md to a PDF, embedding Figure 1 (C1 reproduction)."""
import os, re
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
cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10,
                     textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=10,
                     alignment=1)


def inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r'<font name="DejaVu-Bold">\1</font>', s)
    s = re.sub(r"`(.+?)`", r'<font name="DejaVuMono" size="8.4">\1</font>', s)
    return s


def fig_flowables(png, caption, maxw):
    from PIL import Image as PILImage
    path = os.path.join(ROOT, "figures", png)
    iw, ih = PILImage.open(path).size
    w = maxw; h = w * ih / iw
    return [Spacer(1, 4), Image(path, width=w, height=h), Paragraph(caption, cap)]


# drop Figure 1 in once we pass the verification heading
FIG_AFTER_HEADING = "Verification"
FIG = ("fig1_c1_v2_bifurcation.png",
       "Figure 1. C1 reproduction under the v2 criterion, regenerated through the "
       "Session-8 determinism fix. Pass at 2.29% z-maxima RMSE on the chaotic band.")


def build():
    md = open(os.path.join(ROOT, "paper", "07_progress_log.md")).read().splitlines()
    doc = SimpleDocTemplate(os.path.join(ROOT, "paper", "07_progress_log.pdf"),
                            pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch,
                            leftMargin=0.8*inch, rightMargin=0.8*inch,
                            title="Progress log - Session 8", author="Noah Riego")
    maxw = letter[0] - 1.6*inch
    story = []
    i = 0
    fig_done = False
    while i < len(md):
        ln = md[i].rstrip()
        if not ln.strip():
            i += 1; continue
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(md) and md[i].lstrip().startswith("|"):
                if not re.match(r"^[-:\s|]+$", md[i].strip().strip("|")):
                    rows.append([c.strip() for c in md[i].strip().strip("|").split("|")])
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
            continue
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), title))
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), h1))
            if not fig_done and FIG_AFTER_HEADING.lower() in ln[3:].lower():
                story += fig_flowables(*FIG, maxw); fig_done = True
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), h2))
        elif ln.startswith("- "):
            story.append(Paragraph(inline(ln[2:]), bullet, bulletText="\u2022"))
        else:
            story.append(Paragraph(inline(ln), body))
        i += 1
    doc.build(story)
    print("wrote paper/07_progress_log.pdf")


if __name__ == "__main__":
    build()
