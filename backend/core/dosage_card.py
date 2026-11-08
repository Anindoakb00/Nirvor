"""Deterministic local renderer for the patient dosage card.

Meta's image model refuses medication/dosage prompts by content policy, so the
dosage card is drawn locally with PIL: exact drug names, no network, no cost.
English headers: prescriptions in Bangladesh are written in English, and PIL
has no Bangla shaping (unshaped Bangla would render as broken glyphs).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

CARD_WIDTH = 900
MARGIN = 28
TITLE_HEIGHT = 72
HEADER_HEIGHT = 52
ROW_HEIGHT = 52
FOOTER_HEIGHT = 48

# (header, width) pairs: widths sum to CARD_WIDTH - 2 * MARGIN.
COLUMNS: Tuple[Tuple[str, int], ...] = (
    ("Medicine", 364),
    ("Morning", 100),
    ("Noon", 100),
    ("Night", 100),
    ("Schedule", 180),
)

TITLE_BG = (13, 71, 83)
HEADER_BG = (238, 242, 245)
GRID = (200, 208, 212)
TEXT = (33, 37, 41)
MUTED = (140, 149, 156)
ZEBRA = (247, 250, 252)
FOOTER_TEXT = "Take medicines exactly as prescribed. Ask your doctor or pharmacist if unsure."

_ITEM_RE = re.compile(r"^(.*)\(([^()]*)\)\s*$")
_SCHEDULE_RE = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*$")

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
)


@dataclass(frozen=True)
class CardRow:
    medicine: str
    counts: Optional[Tuple[int, int, int]]
    note: str


@lru_cache(maxsize=16)
def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int):
    size = start_size
    while size > 10:
        font = _load_font(size)
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        if right - left <= max_width:
            return font
        size -= 2
    return _load_font(10)


def _split_items(summary: str) -> List[str]:
    """Splits on top-level commas so names containing commas stay intact.

    Real items always carry " (frequency)", so a paren-less fragment must be
    a broken piece of a comma-bearing name: merge it into the next item.
    """
    raw, buf, depth = [], "", 0
    for ch in summary:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            if buf.strip():
                raw.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        raw.append(buf.strip())
    merged, pending = [], []
    for part in raw:
        if "(" not in part and ")" not in part:
            pending.append(part)
        else:
            pending.append(part)
            merged.append(", ".join(pending))
            pending = []
    if pending:
        merged.append(", ".join(pending))
    return merged


def parse_dosage_summary(summary: str) -> List[CardRow]:
    """Parses 'Name (1+0+1), ...' into card rows. Never raises."""
    rows = []
    for item in _split_items(summary or ""):
        match = _ITEM_RE.match(item)
        if match:
            name, freq = match.group(1).strip(), match.group(2).strip()
        else:
            name, freq = item.strip(), ""
        if not name or name == "None":
            name = "Unspecified"
        counts = None
        sched = _SCHEDULE_RE.match(freq or "")
        if sched:
            counts = (int(sched.group(1)), int(sched.group(2)), int(sched.group(3)))
        note = freq if freq and freq != "None" else "\u2013"
        rows.append(CardRow(medicine=name, counts=counts, note=note))
    return rows


def _column_edges() -> List[int]:
    edges, x = [MARGIN], MARGIN
    for _, width in COLUMNS:
        x += width
        edges.append(x)
    return edges


def render_dosage_card_png(rows: List[CardRow]) -> bytes:
    """Renders rows to PNG bytes. Same input always yields the same bytes."""
    if not rows:
        rows = [CardRow(medicine="No medications listed.", counts=None, note="\u2013")]
    n = len(rows)
    height = MARGIN + TITLE_HEIGHT + HEADER_HEIGHT + n * ROW_HEIGHT + FOOTER_HEIGHT + MARGIN
    img = Image.new("RGB", (CARD_WIDTH, height), "white")
    draw = ImageDraw.Draw(img)
    edges = _column_edges()
    y = MARGIN

    draw.rectangle([edges[0], y, edges[-1], y + TITLE_HEIGHT], fill=TITLE_BG)
    draw.text((edges[0] + 16, y + TITLE_HEIGHT // 2), "Patient Dosage Card",
              font=_load_font(30), fill="white", anchor="lm")
    draw.text((edges[-1] - 16, y + TITLE_HEIGHT // 2), "Nirvor",
              font=_load_font(20), fill=(178, 223, 219), anchor="rm")
    y += TITLE_HEIGHT

    draw.rectangle([edges[0], y, edges[-1], y + HEADER_HEIGHT], fill=HEADER_BG)
    header_font = _load_font(20)
    for i, (title, _) in enumerate(COLUMNS):
        cx = (edges[i] + edges[i + 1]) // 2
        draw.text((cx, y + HEADER_HEIGHT // 2), title, font=header_font,
                  fill=TEXT, anchor="mm")
    y += HEADER_HEIGHT

    for r, row in enumerate(rows):
        if r % 2 == 1:
            draw.rectangle([edges[0], y, edges[-1], y + ROW_HEIGHT], fill=ZEBRA)
        cy = y + ROW_HEIGHT // 2
        med_font = _fit_font(draw, row.medicine, COLUMNS[0][1] - 24, 20)
        draw.text((edges[0] + 12, cy), row.medicine, font=med_font, fill=TEXT, anchor="lm")
        cell_font = _load_font(20)
        for c in range(3):
            cx = (edges[c + 1] + edges[c + 2]) // 2
            if row.counts is None or row.counts[c] == 0:
                draw.text((cx, cy), "\u2013", font=cell_font, fill=MUTED, anchor="mm")
            else:
                draw.text((cx, cy), str(row.counts[c]), font=cell_font, fill=TEXT, anchor="mm")
        note_font = _fit_font(draw, row.note, COLUMNS[4][1] - 24, 20)
        nx = (edges[4] + edges[5]) // 2
        draw.text((nx, cy), row.note, font=note_font, fill=TEXT, anchor="mm")
        y += ROW_HEIGHT

    for x in edges:
        draw.line([x, MARGIN + TITLE_HEIGHT, x, y], fill=GRID, width=1)
    grid_y = MARGIN + TITLE_HEIGHT
    draw.line([edges[0], grid_y, edges[-1], grid_y], fill=GRID, width=1)
    grid_y += HEADER_HEIGHT
    for _ in rows:
        draw.line([edges[0], grid_y, edges[-1], grid_y], fill=GRID, width=1)
        grid_y += ROW_HEIGHT

    draw.text((CARD_WIDTH // 2, y + FOOTER_HEIGHT // 2), FOOTER_TEXT,
              font=_fit_font(draw, FOOTER_TEXT, CARD_WIDTH - 2 * MARGIN - 16, 16),
              fill=MUTED, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
