#!/usr/bin/env python3
"""Export one shared vector scene to editable SVG and font-embedded PDF.

Requires reportlab; optional verification uses pypdf. No data/model inputs.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "descriptor_selection"
W, H = 9 * 72, 5 * 72
INK, MAIN, SECONDARY = "#20262B", "#24556A", "#72777B"
SCENE = []


def register_fonts():
    candidates = [Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts")]
    for directory in candidates:
        if (directory / "Arial.ttf").exists() and (directory / "Arial Bold.ttf").exists():
            pdfmetrics.registerFont(TTFont("Arial", str(directory / "Arial.ttf")))
            pdfmetrics.registerFont(TTFont("Arial-Bold", str(directory / "Arial Bold.ttf")))
            return
    raise FileNotFoundError("Arial.ttf and Arial Bold.ttf are required for matched SVG/PDF typography")


def label(cx, y, text, size=9.3, bold=False, color=INK):
    # Runs are (text, relative size, baseline offset); positive offset lowers text.
    runs = [(text, 1, 0)] if isinstance(text, str) else text
    font = "Arial-Bold" if bold else "Arial"
    width = sum(pdfmetrics.stringWidth(t, font, size * scale) for t, scale, dy in runs)
    x = cx - width / 2
    positioned = []
    for t, scale, dy in runs:
        positioned.append((x, y + dy, t, size * scale))
        x += pdfmetrics.stringWidth(t, font, size * scale)
    top = min(yy - pdfmetrics.getAscent(font) * ss / 1000 for xx, yy, tt, ss in positioned)
    bottom = max(yy - pdfmetrics.getDescent(font) * ss / 1000 for xx, yy, tt, ss in positioned)
    SCENE.append(dict(kind="text", font=font, color=color, runs=positioned,
                      bbox=[cx - width / 2, top, cx + width / 2, bottom]))


def line(x1, y1, x2, y2, color=MAIN, width=.85, dashed=False):
    SCENE.append(dict(kind="line", xy=[x1, y1, x2, y2], color=color, width=width, dashed=dashed))


def arrow(x1, y1, x2, y2, color=MAIN, width=.85, dashed=False, head=3):
    import math
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    bx, by = x2 - head * ux, y2 - head * uy
    line(x1, y1, bx, by, color, width, dashed)
    SCENE.append(dict(kind="polygon", color=color,
                      points=[(x2, y2), (bx - uy * head * .45, by + ux * head * .45),
                              (bx + uy * head * .45, by - ux * head * .45)]))


def metric(cx, y, value, color=SECONDARY):
    label(cx, y, [("R", 1, 0), ("2", .68, -3.5), (f" = {value}", 1, 0)], size=9.3, color=color)


def subscript(text):
    return (text, .72, 2.3)


def build_scene():
    SCENE.clear()
    # Global selection occupies the upper row. Only the selected baseline has
    # a downward connector, exactly aligned with the first local-context stage.
    label(76, 24, "Global descriptors", size=9, bold=True, color=SECONDARY)
    line(24, 31, 624, 31, color="#D4D8DA", width=.5)
    bulk, baseline, scd = 60, 205, 465
    for cx, rows in [(bulk, ("Bulk", "electrostatics")),
                     (baseline, ("+ hydropathy /", "stickiness"))]:
        for y, text in zip([56, 69], rows):
            label(cx, y, text, size=11, bold=True)
    label(bulk, 96, "−NCP", size=12)
    metric(bulk, 123, "0.9195")
    arrow(108, 92, 130, 92, width=1.2, head=4.5)
    label(baseline, 91, [("mean λ", 1, 0), subscript("RG/RGG"), (", mean λ", 1, 0), subscript("AHD")])
    label(baseline, 106, [("SHD", 1, 0), subscript("RG/RGG"), (", SHD", 1, 0), subscript("AHD")])
    metric(baseline, 123, "0.9334")
    label(baseline, 144, "Selected global baseline", size=9.3, bold=True, color=MAIN)
    label(baseline, 158, "5 descriptors", size=9.3, color=MAIN)

    # Thin, dashed side comparison: boxed separately, with no onward arrow.
    arrow(284, 88, 377, 76, color=SECONDARY, width=.6, dashed=True, head=3.5)
    SCENE.append(dict(kind="rect", x=382, y=48, w=166, h=79, color="#AEB4B8", width=.6))
    label(scd, 69, "+ full-sequence SCD", size=10, bold=True, color=SECONDARY)
    label(scd, 92, [("SCD", 1, 0), subscript("RG/RGG"), (" + SCD", 1, 0), subscript("AHD")],
          size=9, color=SECONDARY)
    metric(scd, 114, "0.9373")
    label(scd, 144, "modest additional gain", size=9, color=SECONDARY)
    label(scd, 158, "tested; not retained", size=8, color=SECONDARY)

    label(84, 181, "Local sequence context", size=9, bold=True, color=SECONDARY)
    line(24, 188, 179, 188, color="#D4D8DA", width=.5)
    line(230, 188, 624, 188, color="#D4D8DA", width=.5)
    arrow(baseline, 169, baseline, 225, width=1.2, head=4.5)

    # The lower row uses compact text groups, not equal-sized cards.
    centers = [baseline, 349, 484]
    headings = [("+ local electrostatics",), ("+ local patterning /", "composition"),
                ("+ local interaction", "propensity")]
    for cx, rows in zip(centers, headings):
        for y, text in zip([241, 254], rows):
            label(cx, y, text, size=10.2, bold=True)
    label(baseline, 278, "variation in −NCP across", size=9)
    label(baseline, 291, "5-residue sequence windows", size=9)
    for y, text in zip([276, 289, 302], ["local SCD", "RG / aromatic content", "local SHD"]):
        label(349, y, text, size=9)
    label(484, 278, "Ashbaugh–Hatch", size=9)
    label(484, 291, "heterogeneity", size=9)
    for cx, value in zip(centers, ["0.9472", "0.9492", "0.9507"]):
        metric(cx, 320, value)
    for left, right in zip(centers, centers[1:] + [551]):
        arrow(left + 2, 334, right - 4 if right != 551 else right, 334, width=1.2, head=4.5)
    for cx in centers:
        SCENE.append(dict(kind="circle", x=cx, y=334, radius=1.7, color=MAIN))
    SCENE.append(dict(kind="rect", x=551, y=266, w=85, h=81, color=MAIN, width=1.1))
    label(593.5, 284, "FINAL", size=9, bold=True, color=MAIN)
    label(593.5, 309, "13 descriptors", size=11.2, bold=True)
    label(593.5, 326, "5 global + 8 local", size=8.5)


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Fig4_descriptor_selection"
    pdf = canvas.Canvas(str(stem.with_suffix(".pdf")), pagesize=(W, H), pageCompression=1,
                        initialFontName="Arial")
    pdf.setTitle("AHD–RG/RGG descriptor selection")
    pdf.setAuthor("Figure 5 revision")
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.rect(0, 0, W, H, fill=1, stroke=0)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="9in" height="5in" viewBox="0 0 {W:.8f} {H:.8f}">',
           '<rect width="100%" height="100%" fill="white"/>']
    for obj in SCENE:
        kind, color = obj["kind"], obj["color"]
        pdf.setFillColor(HexColor(color))
        pdf.setStrokeColor(HexColor(color))
        if kind == "text":
            weight = "700" if obj["font"] == "Arial-Bold" else "400"
            svg.append(f'<text font-family="Arial" font-weight="{weight}" fill="{color}" xml:space="preserve">')
            for x, y, text, size in obj["runs"]:
                pdf.setFont(obj["font"], size)
                pdf.drawString(x, H - y, text)
                svg.append(f'<tspan x="{x:.6f}" y="{y:.6f}" font-size="{size:.6f}">{escape(text)}</tspan>')
            svg.append('</text>')
        elif kind == "line":
            x1, y1, x2, y2 = obj["xy"]
            pdf.setLineWidth(obj["width"])
            pdf.setDash([2, 1.7] if obj["dashed"] else [])
            pdf.line(x1, H - y1, x2, H - y2)
            dash = ' stroke-dasharray="2 1.7"' if obj["dashed"] else ''
            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{obj["width"]}"{dash}/>')
            pdf.setDash([])
        elif kind == "polygon":
            path = pdf.beginPath()
            for i, (x, y) in enumerate(obj["points"]):
                (path.moveTo if i == 0 else path.lineTo)(x, H - y)
            path.close()
            pdf.drawPath(path, fill=1, stroke=0)
            points = ' '.join(f'{x},{y}' for x, y in obj["points"])
            svg.append(f'<polygon points="{points}" fill="{color}"/>')
        elif kind == "circle":
            pdf.circle(obj["x"], H - obj["y"], obj["radius"], stroke=0, fill=1)
            svg.append(f'<circle cx="{obj["x"]}" cy="{obj["y"]}" r="{obj["radius"]}" fill="{color}"/>')
        elif kind == "rect":
            pdf.setLineWidth(obj["width"])
            pdf.rect(obj["x"], H - obj["y"] - obj["h"], obj["w"], obj["h"], stroke=1, fill=0)
            svg.append(f'<rect x="{obj["x"]}" y="{obj["y"]}" width="{obj["w"]}" height="{obj["h"]}" fill="none" stroke="{color}" stroke-width="{obj["width"]}"/>')
    pdf.showPage()
    pdf.save()
    svg.append('</svg>')
    stem.with_suffix('.svg').write_text('\n'.join(svg), encoding='utf-8')
    # Text bounding boxes provide a reproducible geometric audit.
    bounds = [o for o in SCENE if o['kind'] == 'text']
    for obj in bounds:
        x1, y1, x2, y2 = obj['bbox']
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, obj
    for i, obj in enumerate(bounds):
        x1, y1, x2, y2 = obj['bbox']
        for other in bounds[i + 1:]:
            u1, v1, u2, v2 = other['bbox']
            assert not (min(x2, u2) > max(x1, u1) and min(y2, v2) > max(y1, v1)), (obj, other)
    (OUT / "layout_audit.json").write_text(json.dumps({"width_in": 9, "height_in": 5,
        "shared_scene": True, "text_objects": len(bounds), "text_bounds": bounds}, indent=2))
    print(stem.with_suffix('.svg'))
    print(stem.with_suffix('.pdf'))


if __name__ == "__main__":
    register_fonts()
    build_scene()
    export()
