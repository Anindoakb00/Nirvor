import base64
import io

import pytest
from PIL import Image

from core.dosage_card import (
    CARD_WIDTH,
    FOOTER_HEIGHT,
    HEADER_HEIGHT,
    MARGIN,
    ROW_HEIGHT,
    TITLE_HEIGHT,
    parse_dosage_summary,
    render_dosage_card_png,
)
from core.meta_agent import generate_patient_dosage_card

ABDULLAH = "Penvik 250 (1+1+1), Furoclav 250 (1+0+1), Lasix 20mg (stat)"


def _expected_height(n_rows):
    return MARGIN + TITLE_HEIGHT + HEADER_HEIGHT + n_rows * ROW_HEIGHT + FOOTER_HEIGHT + MARGIN


def test_parse_three_part_schedules():
    rows = parse_dosage_summary("Penvik 250 (1+1+1), Furoclav 250 (1+0+1)")
    assert [(r.medicine, r.counts, r.note) for r in rows] == [
        ("Penvik 250", (1, 1, 1), "1+1+1"),
        ("Furoclav 250", (1, 0, 1), "1+0+1"),
    ]


def test_parse_handles_stat_missing_and_commas():
    rows = parse_dosage_summary("Lasix 20mg (stat), Tab. X, 500mg (1+0+1), Y (None), Zed")
    assert [(r.medicine, r.counts, r.note) for r in rows] == [
        ("Lasix 20mg", None, "stat"),
        ("Tab. X, 500mg", (1, 0, 1), "1+0+1"),
        ("Y", None, "\u2013"),
        ("Zed", None, "\u2013"),
    ]


def test_parse_empty_and_none_yield_no_rows():
    assert parse_dosage_summary("") == []
    assert parse_dosage_summary(None) == []


def test_render_png_dimensions_and_deterministic():
    rows = parse_dosage_summary(ABDULLAH)
    first, second = render_dosage_card_png(rows), render_dosage_card_png(rows)
    assert first == second
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(first))
    assert img.size == (CARD_WIDTH, _expected_height(3))


def test_render_empty_summary_shows_placeholder():
    png = render_dosage_card_png(parse_dosage_summary(""))
    img = Image.open(io.BytesIO(png))
    assert img.size == (CARD_WIDTH, _expected_height(1))


@pytest.mark.asyncio
async def test_generate_returns_png_data_uri():
    result = await generate_patient_dosage_card(ABDULLAH)
    assert result["provider_used"] == "local_renderer"
    assert result["url"].startswith("data:image/png;base64,")
    raw = base64.b64decode(result["url"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(raw)).size == (CARD_WIDTH, _expected_height(3))


@pytest.mark.asyncio
async def test_generate_handles_none_summary():
    result = await generate_patient_dosage_card(None)
    assert result["provider_used"] == "local_renderer"
    assert result["url"].startswith("data:image/png;base64,")
