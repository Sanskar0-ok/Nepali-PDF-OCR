# ============================================================
# WEB APP - FAST SELECTABLE PDF VIEWER
# ============================================================
from fastapi.middleware.cors import CORSMiddleware

import traceback
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pymupdf
import torch

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ------------------------------------------------------------
# FAST / LOW-END PC OCR CONFIG
# ------------------------------------------------------------

# Lower DPI = dramatically less image data for OCR.
# 120 is a good compromise for ordinary Nepali documents.
# ============================================================
# NEPALI PDF OCR CONFIG
# ============================================================
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
WEB_OCR_DPI_CPU = 150
WEB_OCR_DPI_GPU = 170
WEB_VIEW_DPI = 120
WEB_OCR_CANVAS_CPU = 1920
WEB_OCR_CANVAS_GPU = 2200

# CPU: keep at 1 for low-memory machines.
# GPU: batching gives better throughput.
WEB_OCR_BATCH_CPU = 1
WEB_OCR_BATCH_GPU = 8

WEB_OCR_ENGINE = None

LINE_Y_TOLERANCE = 0.55

MIN_WEB_FONT_PX = 8
MAX_WEB_FONT_PX = 24
# OCR text should normally be around this size.
OCR_FONT_SCALE = 0.72
# ------------------------------------------------------------
# BASIC TEXT HELPERS
# ------------------------------------------------------------

def normalize_text(text):
    """Normalize extracted/OCR text without damaging Nepali."""

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def suspicious_legacy_text(text):
    """
    Detect corrupted/legacy Nepali text.

    Important:
    Some broken Nepali PDFs contain mostly valid Devanagari
    with only a few corrupted Latin/extended-Unicode characters.

    Example:
        गणेशटारकृȤष तथा पशुपंÿी कृषक समुह

    This must NOT be allowed to bypass OCR.
    """

    if not text or not text.strip():
        return True

    cleaned = text.strip()

    # --------------------------------------------------------
    # 1. Known legacy/corruption characters
    # --------------------------------------------------------

    known_bad_chars = (
        "ĴľĂČȥàÙĆìÝðĨŢŠž"
        "Ȥÿ"
    )

    known_bad_count = sum(
        cleaned.count(char)
        for char in known_bad_chars
    )

    if known_bad_count > 0:
        return True

    # --------------------------------------------------------
    # 2. Devanagari detection
    # --------------------------------------------------------

    devanagari_chars = re.findall(
        r"[\u0900-\u097F]",
        cleaned
    )

    devanagari_count = len(
        devanagari_chars
    )

    # --------------------------------------------------------
    # 3. Extended Latin characters
    #
    # Characters such as:
    #   Ȥ
    #   ÿ
    #   Ĵ
    #   ľ
    #
    # are extremely suspicious when they appear inside
    # predominantly Nepali text.
    # --------------------------------------------------------

    extended_latin_chars = re.findall(
        r"[\u0080-\u024F]",
        cleaned
    )

    extended_latin_count = len(
        extended_latin_chars
    )

    # If the page contains substantial Devanagari and even
    # a small amount of extended Latin, treat it as corrupted.
    if (
        devanagari_count >= 10
        and extended_latin_count >= 1
    ):
        return True

    # --------------------------------------------------------
    # 4. Detect suspicious mixed Unicode words
    # --------------------------------------------------------

    mixed_words = re.findall(
        r"[\u0900-\u097F]+[\u0080-\u024F]+|"
        r"[\u0080-\u024F]+[\u0900-\u097F]+",
        cleaned
    )

    if mixed_words:
        return True

    # --------------------------------------------------------
    # 5. Existing ratio-based detection
    # --------------------------------------------------------

    total = max(
        len(cleaned),
        1
    )

    suspicious_ratio = (
        known_bad_count / total
    )

    if suspicious_ratio > 0.005:
        return True

    # --------------------------------------------------------
    # 6. Lots of text but no Devanagari
    # --------------------------------------------------------

    if (
        len(cleaned) >= 25
        and devanagari_count == 0
    ):
        # Only consider this suspicious if it contains
        # unusual extended characters.
        if extended_latin_count > 0:
            return True

    return False


# ------------------------------------------------------------
# PDF FONT HELPERS
# ------------------------------------------------------------

def get_page_font_names(page):
    """Return font names embedded/referenced by a PDF page."""

    fonts = []

    try:

        for font in page.get_fonts(
            full=True
        ):

            if len(font) >= 4:

                name = font[3]

                if name:
                    fonts.append(name)

    except Exception:
        pass

    return fonts


def clean_pdf_font_name(font_name):

    if not font_name:
        return "Noto Sans Devanagari"

    font_name = str(font_name)

    # Remove PDF subset prefix:
    # ABCDEF+Kalimati -> Kalimati
    font_name = re.sub(
        r"^[A-Z]{6}\+",
        "",
        font_name
    )

    lower = font_name.lower()

    if any(
        technical in lower
        for technical in [
            "identity",
            "cidfont",
            "symbol",
            "embedded",
        ]
    ):
        return "Noto Sans Devanagari"

    return font_name


def choose_font(font_names):

    if not font_names:
        return "Noto Sans Devanagari"

    preferred = [
        "Kalimati",
        "Mangal",
        "Kokila",
        "Noto Sans Devanagari",
        "Noto Serif Devanagari",
        "Lohit Devanagari",
        "Utsaah",
    ]

    cleaned = [
        clean_pdf_font_name(name)
        for name in font_names
    ]

    for preferred_font in preferred:

        preferred_lower = preferred_font.lower()

        for font in cleaned:

            if preferred_lower in font.lower():
                return font

    return cleaned[0]


def choose_web_font(font_name):

    font_name = clean_pdf_font_name(
        font_name
    )

    lower = font_name.lower()

    if "kalimati" in lower:
        return "Kalimati"

    if "mangal" in lower:
        return "Mangal"

    if "kokila" in lower:
        return "Kokila"

    if "noto serif" in lower:
        return "Noto Serif Devanagari"

    if "noto sans" in lower:
        return "Noto Sans Devanagari"

    if "lohit" in lower:
        return "Lohit Devanagari"

    if "utsaah" in lower:
        return "Utsaah"

    return font_name


# ------------------------------------------------------------
# PDF FONT STYLE HELPERS
# ------------------------------------------------------------

def span_is_bold(flags):
    """
    PyMuPDF font flags:
    bit 4 = bold
    """

    return bool(
        int(flags) & 16
    )


def span_is_italic(flags):
    """
    PyMuPDF font flags:
    bit 1 = italic.
    """

    return bool(
        int(flags) & 2
    )


# ------------------------------------------------------------
# OCR ENGINE
# ------------------------------------------------------------

class WebOCRProcessor:

    def __init__(self):

        use_gpu = torch.cuda.is_available()

        print(
            "[WEB] Initializing EasyOCR..."
        )

        print(
            f"[WEB] GPU available: {use_gpu}"
        )

        self.reader = easyocr.Reader(
            ["ne", "en"],
            gpu=use_gpu,
            model_storage_directory=str(
                Path(__file__).resolve().parent /
                "models"
            ),
            download_enabled=False,
            verbose=False
        )

        print(
            "[WEB] EasyOCR ready."
        )


def get_web_ocr_engine():

    global WEB_OCR_ENGINE

    if WEB_OCR_ENGINE is None:

        WEB_OCR_ENGINE = (
            WebOCRProcessor()
        )

    return WEB_OCR_ENGINE


# ------------------------------------------------------------
# PAGE RENDERING FOR OCR
# ------------------------------------------------------------

def web_render_page(page, dpi):
    pix = page.get_pixmap(
        dpi=dpi,
        alpha=False,
        colorspace=pymupdf.csRGB
    )

    image = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    if pix.n == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    return image

# ------------------------------------------------------------
# OCR FONT SIZE
# ------------------------------------------------------------

def estimate_web_ocr_font_size(line_height, scale=1.0):
    """
    Convert OCR bounding-box height into a normal browser font size.

    OCR box height is NOT the same thing as CSS font size, so
    don't use the raw height directly.
    """

    if not line_height or line_height <= 0:
        return 14

    font_size = float(line_height) * OCR_FONT_SCALE * scale

    return max(
        MIN_WEB_FONT_PX,
        min(MAX_WEB_FONT_PX, font_size)
    )

# ------------------------------------------------------------
# OCR LINE GROUPING
# ------------------------------------------------------------

def group_web_ocr_lines(items):

    if not items:
        return []

    items = sorted(
        items,
        key=lambda item: (
            item["y"] +
            item["height"] / 2,
            item["x"]
        )
    )

    lines = []

    for item in items:

        item_top = item["y"]
        item_bottom = (
            item["y"] +
            item["height"]
        )

        item_center = (
            item_top +
            item["height"] / 2
        )

        best_line = None
        best_distance = None

        for line in lines:

            line_top = line["y"]
            line_bottom = line["bottom"]

            min_height = min(
                item["height"],
                line["height"]
            )

            if min_height <= 0:
                continue

            overlap = max(
                0,
                min(
                    item_bottom,
                    line_bottom
                )
                -
                max(
                    item_top,
                    line_top
                )
            )

            overlap_ratio = (
                overlap / min_height
            )

            distance = abs(
                item_center -
                line["center_y"]
            )

            tolerance = max(
                5,
                min_height *
                LINE_Y_TOLERANCE
            )

            if (
                overlap_ratio >= 0.30
                or distance <= tolerance
            ):

                if (
                    best_distance is None
                    or distance < best_distance
                ):

                    best_line = line
                    best_distance = distance

        if best_line is None:

            lines.append({

                "items": [item],

                "y": item_top,

                "bottom": item_bottom,

                "height": item["height"],

                "center_y": item_center,

            })

        else:

            best_line["items"].append(
                item
            )

            top_values = [
                x["y"]
                for x in best_line["items"]
            ]

            bottom_values = [
                x["y"] + x["height"]
                for x in best_line["items"]
            ]

            best_line["y"] = min(
                top_values
            )

            best_line["bottom"] = max(
                bottom_values
            )

            best_line["height"] = (
                best_line["bottom"] -
                best_line["y"]
            )

            best_line["center_y"] = (
                best_line["y"] +
                best_line["height"] / 2
            )

    for line in lines:

        line["items"].sort(
            key=lambda item: item["x"]
        )

    lines.sort(
        key=lambda line: line["y"]
    )

    return lines


# ------------------------------------------------------------
# OCR LINE RECONSTRUCTION
# ------------------------------------------------------------

def build_ocr_lines(
    items,
    page_width,
    page_height,
    font_family="Kalimati"
):

    grouped = group_web_ocr_lines(
        items
    )

    output = []

    for line in grouped:

        line_items = [
            item
            for item in line["items"]
            if item["text"].strip()
        ]

        if not line_items:
            continue

        x1 = min(
            item["x"]
            for item in line_items
        )

        x2 = max(
            item["x"] +
            item["width"]
            for item in line_items
        )

        y1 = min(
            item["y"]
            for item in line_items
        )

        y2 = max(
            item["y"] +
            item["height"]
            for item in line_items
        )

        # ----------------------------------------------------
        # Reconstruct text left -> right.
        # ----------------------------------------------------

        parts = []

        previous = None

        for item in line_items:

            text = item["text"].strip()

            if not text:
                continue

            if previous is not None:

                gap = (
                    item["x"]
                    -
                    (
                        previous["x"] +
                        previous["width"]
                    )
                )

                average_height = (
                    previous["height"] +
                    item["height"]
                ) / 2

                # EasyOCR normally returns word boxes.
                # Add a space only between clearly
                # separated words.
                if (
                    gap >
                    average_height * 0.25
                ):
                    parts.append(" ")

            parts.append(text)

            previous = item

        text = "".join(
            parts
        ).strip()

        if not text:
            continue

        font_size = estimate_web_ocr_font_size(
            y2 - y1
        )

        confidence_values = [
            item["confidence"]
            for item in line_items
        ]

        confidence = (
            sum(confidence_values)
            /
            len(confidence_values)
        )

        output.append({

            "text": text,

            "x": round(
                x1,
                2
            ),

            "y": round(
                y1,
                2
            ),

            "width": round(
                max(
                    1,
                    x2 - x1
                ),
                2
            ),

            "height": round(
                max(
                    1,
                    y2 - y1
                ),
                2
            ),

            "font_size": font_size,

            "font_family":
                font_family,

            "bold": False,

            "italic": False,

            "confidence": round(
                confidence,
                3
            ),

        })

    return output

# ------------------------------------------------------------
# PARAGRAPH GROUPING
# ------------------------------------------------------------

def group_lines_into_paragraphs(
    lines,
    page_width
):
    """
    Combine consecutive visual lines that belong to the
    same paragraph.

    The browser can then wrap the paragraph naturally instead
    of treating every PDF/OCR line as an isolated sentence.
    """

    if not lines:
        return []

    lines = sorted(
        lines,
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    paragraphs = []

    current = None

    for line in lines:

        x = float(line["x"])
        y = float(line["y"])
        width = float(line["width"])
        height = float(line["height"])

        bottom = y + height

        if current is None:

            current = {
                "lines": [line],

                "x": x,
                "y": y,

                "right": x + width,
                "bottom": bottom,

                "height": height,

                "font_size":
                    line["font_size"],

                "font_family":
                    line["font_family"],

                "bold":
                    line.get(
                        "bold",
                        False
                    ),

                "italic":
                    line.get(
                        "italic",
                        False
                    ),
            }

            continue


        previous = current["lines"][-1]

        previous_bottom = (
            previous["y"] +
            previous["height"]
        )

        previous_height = max(
            1,
            previous["height"]
        )


        vertical_gap = (
            y -
            previous_bottom
        )


        
        indent_difference = abs(
            x -
            current["x"]
        )


        
        normal_vertical_gap = (
            vertical_gap
            <=
            previous_height * 0.90
        )


        
        same_left_area = (
            indent_difference
            <=
            max(
                25,
                previous_height * 2.0
            )
        )


        
        previous_text = (
            previous.get(
                "text",
                ""
            ).strip()
        )

        short_previous_line = (
            len(previous_text) < 15
        )


        should_join = (
            normal_vertical_gap
            and
            same_left_area
            and
            not short_previous_line
        )


        if should_join:

            current["lines"].append(
                line
            )

            current["right"] = max(
                current["right"],
                x + width
            )

            current["bottom"] = max(
                current["bottom"],
                bottom
            )

            current["height"] = (
                current["bottom"]
                -
                current["y"]
            )

        else:

            paragraphs.append(
                current
            )

            current = {
                "lines": [line],

                "x": x,
                "y": y,

                "right": x + width,
                "bottom": bottom,

                "height": height,

                "font_size":
                    line["font_size"],

                "font_family":
                    line["font_family"],

                "bold":
                    line.get(
                        "bold",
                        False
                    ),

                "italic":
                    line.get(
                        "italic",
                        False
                    ),
            }


    if current is not None:

        paragraphs.append(
            current
        )


    # --------------------------------------------------------
    # Convert paragraph groups into browser blocks
    # --------------------------------------------------------

    output = []

    for paragraph in paragraphs:

        paragraph_lines = (
            paragraph["lines"]
        )

        texts = []

        for index, line in enumerate(
            paragraph_lines
        ):

            text = line.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            if texts:
                texts.append(" ")

            texts.append(text)


        text = "".join(
            texts
        ).strip()


        if not text:
            continue


        first_line = (
            paragraph_lines[0]
        )


        # Estimate how many visual lines this
        # paragraph originally occupied.
        original_line_count = len(
            paragraph_lines
        )


        average_font_size = sum(
            float(
                item.get(
                    "font_size",
                    12
                )
            )
            for item in paragraph_lines
        ) / max(
            1,
            original_line_count
        )


        output.append({

            "text":
                text,

            "x":
                round(
                    paragraph["x"],
                    2
                ),

            "y":
                round(
                    paragraph["y"],
                    2
                ),

            
            "width":
                round(
                    max(
                        1,
                        page_width -
                        paragraph["x"]
                    ),
                    2
                ),

            
            "height":
                round(
                    max(
                        paragraph["height"],
                        average_font_size *
                        1.35 *
                        original_line_count
                    ),
                    2
                ),

            "font_size":
                round(
                    average_font_size,
                    2
                ),

            "font_family":
                first_line.get(
                    "font_family",
                    "Kalimati"
                ),

            "bold":
                first_line.get(
                    "bold",
                    False
                ),

            "italic":
                first_line.get(
                    "italic",
                    False
                ),

            "original_line_count":
                original_line_count,

        })


    return output
# ------------------------------------------------------------
# OCR PAGE
# ------------------------------------------------------------

def web_ocr_page(engine, page):

    gpu = torch.cuda.is_available()

    if gpu:
        dpi = WEB_OCR_DPI_GPU
        batch_size = WEB_OCR_BATCH_GPU
        canvas_size = WEB_OCR_CANVAS_GPU
    else:
        dpi = WEB_OCR_DPI_CPU
        batch_size = WEB_OCR_BATCH_CPU
        canvas_size = WEB_OCR_CANVAS_CPU

    print(
        f"[WEB OCR] "
        f"{'GPU' if gpu else 'CPU'} | "
        f"DPI={dpi} | "
        f"canvas={canvas_size}"
    )

    # --------------------------------------------------------
    # Render page for OCR
    # --------------------------------------------------------

    image = web_render_page(
        page,
        dpi
    )

    # OCR coordinates are produced at OCR DPI.
    # Convert them to the browser's fixed display DPI.
    scale = WEB_VIEW_DPI / dpi

    # --------------------------------------------------------
    # Run OCR
    # --------------------------------------------------------

    results = engine.reader.readtext(
        image,

        decoder="greedy",
        paragraph=False,
        detail=1,

        batch_size=batch_size,
        workers=0,

        canvas_size=canvas_size,
        mag_ratio=1.0,

        text_threshold=0.65,
        low_text=0.35,
        link_threshold=0.35,

        contrast_ths=0.05,
        adjust_contrast=0.5,

        min_size=8,
    )

    # --------------------------------------------------------
    # Convert OCR detections into browser coordinates
    # --------------------------------------------------------

    items = []

    for result in results:

        if len(result) != 3:
            continue

        bbox, text, confidence = result

        confidence = float(
            confidence
        )

        # Ignore very weak detections.
        if confidence < 0.25:
            continue

        text = normalize_text(
            text
        )

        if not text.strip():
            continue

        xs = [
            float(point[0])
            for point in bbox
        ]

        ys = [
            float(point[1])
            for point in bbox
        ]

        # OCR coordinates -> browser coordinates
        x1 = min(xs) * scale
        y1 = min(ys) * scale
        x2 = max(xs) * scale
        y2 = max(ys) * scale

        width = max(
            1,
            x2 - x1
        )

        height = max(
            1,
            y2 - y1
        )

        items.append({

            "text": text,

            "x": x1,
            "y": y1,

            "width": width,
            "height": height,

            "confidence": confidence,

        })

    # --------------------------------------------------------
    # Get the PDF's font information
    # --------------------------------------------------------

    pdf_fonts = get_page_font_names(
        page
    )

    source_font = choose_font(
        pdf_fonts
    )

    web_font = choose_web_font(
        source_font
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Browser page dimensions come from the REAL PDF page,
    # not from the OCR image.
    # --------------------------------------------------------

    page_rect = page.rect

    page_width = (
        page_rect.width *
        WEB_VIEW_DPI /
        72.0
    )

    page_height = (
        page_rect.height *
        WEB_VIEW_DPI /
        72.0
    )

    # --------------------------------------------------------
    # Reconstruct OCR detections into text lines
    # --------------------------------------------------------

    lines = build_ocr_lines(
        items,
        page_width,
        page_height,
        font_family=web_font
    )
    lines = group_lines_into_paragraphs(
        lines,
        page_width
    )

    return (
        round(page_width, 2),
        round(page_height, 2),
        lines
    )
# ------------------------------------------------------------
# DIRECT PDF TEXT
# ------------------------------------------------------------

def group_direct_spans_into_lines(
    page
):

    data = page.get_text(
        "dict"
    )

    output = []

    fonts = get_page_font_names(
        page
    )

    fallback_font = choose_font(
        fonts
    )

    scale = WEB_VIEW_DPI / 72.0

    for block in data.get(
        "blocks",
        []
    ):

        if block.get(
            "type"
        ) != 0:
            continue

        for pdf_line in block.get(
            "lines",
            []
        ):

            spans = []

            for span in pdf_line.get(
                "spans",
                []
            ):

                text = span.get(
                    "text",
                    ""
                )

                if not text.strip():
                    continue

                bbox = span.get(
                    "bbox",
                    (0, 0, 0, 0)
                )

                font = choose_web_font(
                    span.get(
                        "font",
                        fallback_font
                    )
                )

                size_pt = float(
                    span.get(
                        "size",
                        11
                    )
                )

                # PDF points -> browser pixels
                font_px = (
                    size_pt *
                    scale
                )

                # Prevent broken PDFs from
                # generating absurd text.
                font_px = max(
                    8,
                    min(
                        28,
                        font_px
                    )
                )

                spans.append({

                    "text": text,

                    "x":
                        bbox[0] * scale,

                    "y":
                        bbox[1] * scale,

                    "width": max(
                        1,
                        (
                            bbox[2] -
                            bbox[0]
                        ) * scale
                    ),

                    "height": max(
                        1,
                        (
                            bbox[3] -
                            bbox[1]
                        ) * scale
                    ),

                    "font_size":
                        font_px,

                    "font_family":
                        font,

                    "bold":
                        span_is_bold(
                            span.get(
                                "flags",
                                0
                            )
                        ),

                    "italic":
                        span_is_italic(
                            span.get(
                                "flags",
                                0
                            )
                        ),

                })

            if not spans:
                continue

            x1 = min(
                span["x"]
                for span in spans
            )

            x2 = max(
                span["x"] +
                span["width"]
                for span in spans
            )

            y1 = min(
                span["y"]
                for span in spans
            )

            y2 = max(
                span["y"] +
                span["height"]
                for span in spans
            )

            segments = []

            for span in spans:

                segments.append({

                    "text":
                        span["text"],

                    "font_size":
                        span["font_size"],

                    "font_family":
                        span["font_family"],

                    "bold":
                        span["bold"],

                    "italic":
                        span["italic"],

                })

            primary = spans[0]

            output.append({

                "x": round(
                    x1,
                    2
                ),

                "y": round(
                    y1,
                    2
                ),

                "width": round(
                    max(
                        1,
                        x2 - x1
                    ),
                    2
                ),

                "height": round(
                    max(
                        1,
                        y2 - y1
                    ),
                    2
                ),

                "font_size":
                    primary["font_size"],

                "font_family":
                    primary["font_family"],

                "bold":
                    primary["bold"],

                "italic":
                    primary["italic"],

                "text": "".join(
                    span["text"]
                    for span in spans
                ),

                "segments":
                    segments,

                "confidence": 1.0,

            })

    output.sort(
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    return output


def web_direct_text_page(
    page
):

    scale = WEB_VIEW_DPI / 72.0

    rect = page.rect

    lines = (
        group_direct_spans_into_lines(
            page
        )
    )
    page_width = (
        rect.width *
        scale
    )

    lines = group_lines_into_paragraphs(
        lines,
        page_width
    )

    return (
        page_width,
        

        rect.height * scale,

        lines

    )


# ------------------------------------------------------------
# PAGE PROCESSING
# ------------------------------------------------------------

def process_page_for_web(
    page,
    page_number,
    total_pages
):

    raw_text = normalize_text(
        page.get_text(
            "text"
        )
    )

    if (
        raw_text
        and not suspicious_legacy_text(
            raw_text
        )
    ):

        print(
            f"[PAGE {page_number}/{total_pages}] "
            "DIRECT TEXT - OCR SKIPPED"
        )

        width, height, lines = (
            web_direct_text_page(
                page
            )
        )

        mode = "text"

    else:

        print(
            f"[PAGE {page_number}/{total_pages}] "
            "OCR REQUIRED"
        )

        width, height, lines = (
            web_ocr_page(
                get_web_ocr_engine(),
                page
            )
        )

        mode = "ocr"

    return {

        "page":
            page_number,

        "width":
            width,

        "height":
            height,

        "mode":
            mode,

        "lines":
            lines,

    }


# ------------------------------------------------------------
# FASTAPI
# ------------------------------------------------------------

app = FastAPI(
    title="Nepali PDF Viewer"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


FRONTEND_DIR = (
    BASE_DIR /
    "frontend"
)


app.mount(
    "/static",
    StaticFiles(
        directory=str(
            FRONTEND_DIR
        )
    ),
    name="static"
)


@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    index_file = (
        FRONTEND_DIR /
        "index.html"
    )

    with open(
        index_file,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


@app.post(
    "/convert"
)
async def convert_pdf(
    file: UploadFile = File(...)
):

    if not file.filename:

        return {
            "error":
                "No file selected."
        }

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        return {
            "error":
                "Please upload a PDF file."
        }

    temp_dir = tempfile.mkdtemp(
        prefix="pdftoword_"
    )

    safe_name = Path(
        file.filename
    ).name

    pdf_path = os.path.join(
        temp_dir,
        safe_name
    )

    try:

        with open(pdf_path, "wb") as buffer:
            total_size = 0

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break
                

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    return {
                        "error": "PDF is too large. Maximum size is 50 MB."
                    }
                buffer.write(chunk)
        pdf = pymupdf.open(pdf_path)

        try:
            total_pages = len(pdf)

            print("=" * 60)
            print(
                f"[WEB] Processing "
                f"{safe_name} "
                f"({total_pages} pages)"
            )
            print("=" * 60)

            pages = []

            for page_number, page in enumerate(pdf, start=1):
                pages.append(
                    process_page_for_web(
                        page,
                        page_number,
                        total_pages
                    )
                )

            print("[WEB] Processing complete.")

            return {
                "filename": safe_name,
                "pages": pages,
            }

        finally:
            pdf.close()

        print(
            "[WEB] Processing complete."
        )

        return {

            "filename":
                safe_name,

            "pages":
                pages,

        }

    except Exception as exc:

        print(
            f"[WEB ERROR] {exc}"
        )

        return {
            "error":
                str(exc)
        }

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )