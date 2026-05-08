"""Render text prompts into page images at controlled DPI settings.

This lightweight renderer follows the visual-text rendering setup used by
Glyph (https://github.com/thu-coai/Glyph): vector typesetting with ReportLab
followed by rasterization with pdf2image. The implementation here is trimmed
for ACZ-Jailbreak's release scripts.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

from pdf2image import convert_from_bytes
from PIL import Image, ImageChops
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, A3, LETTER
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

PAGE_SIZE_MAP = {
    "A4": A4,
    "A3": (16.5 * inch, 11.7 * inch),
    "LETTER": LETTER,
}

ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}


def find_default_font() -> str:
    """Return a cross-platform font path that works for most releases."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No default font found. Pass --font-path to scripts/generate_images.py."
    )


def _crop_whitespace(image: Image.Image, background: str = "#FFFFFF") -> Image.Image:
    bg = Image.new(image.mode, image.size, background)
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image


def _safe_font_name(font_path: str) -> str:
    stem = Path(font_path).stem
    return re.sub(r"[^A-Za-z0-9_]+", "_", stem)


def convert_text_to_images(
    text: str,
    output_path: str | Path,
    font_path: str | None = None,
    font_size: int = 9,
    dpi: int = 300,
    page_size: str = "A4",
    margin: int = 5,
    text_color: str = "#000000",
    bg_color: str = "#FFFFFF",
    alignment: str = "JUSTIFY",
    auto_crop_to_content: bool = False,
) -> list[str]:
    """Convert a text string into one or more PNG page images.

    If output_path ends with .png, page suffixes are inserted when the text spans
    multiple pages. Otherwise output_path is treated as a directory.
    """
    font_path = font_path or find_default_font()
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    output_path = Path(output_path)
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        output_dir = output_path.parent or Path(".")
        base_name = output_path.stem
    else:
        output_dir = output_path
        base_name = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)

    page_dimensions = PAGE_SIZE_MAP.get(page_size.upper())
    if page_dimensions is None:
        raise ValueError(f"Unsupported page size: {page_size}")

    align = ALIGN_MAP.get(alignment.upper())
    if align is None:
        raise ValueError(f"Unsupported alignment: {alignment}")

    font_name = _safe_font_name(font_path)
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception:
        pass

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_dimensions,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    styles = getSampleStyleSheet()
    paragraph_style = ParagraphStyle(
        name="ACZText",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.HexColor(text_color),
        backColor=colors.HexColor(bg_color),
        alignment=align,
        wordWrap="CJK" if re.search(r"[\u4E00-\u9FFF]", text) else None,
    )

    clean_text = text.replace("\xad", "").replace("\u200b", "")
    clean_text = re.sub(r" {2,}", lambda match: "&nbsp;" * len(match.group()), escape(clean_text))
    parts = clean_text.split("\n")
    story = [Paragraph("<br/>".join(parts[i : i + 30]), paragraph_style) for i in range(0, len(parts), 30)]
    doc.build(story)

    images = convert_from_bytes(buffer.getvalue(), dpi=dpi, fmt="png")
    saved_paths: list[str] = []
    for page_index, image in enumerate(images, start=1):
        if auto_crop_to_content:
            image = _crop_whitespace(image, bg_color)
        suffix = f"_{page_index:03d}" if len(images) > 1 or output_path.suffix else ""
        out_file = output_dir / f"{base_name}{suffix}.png"
        image.save(out_file)
        saved_paths.append(str(out_file))
    return saved_paths
