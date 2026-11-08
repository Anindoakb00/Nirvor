import io
import json
import sys
from unittest.mock import patch

import pytest

from core import clinical_agent
from core.clinical_agent import SYSTEM_PROMPT
from core.math_engine import ClinicalMathEngine


def test_prompt_requires_bangla_in_both_views():
    assert "Write EVERY user-facing text field in Bangla" in SYSTEM_PROMPT
    assert "Bengali script" in SYSTEM_PROMPT
    assert "doctor_view" in SYSTEM_PROMPT and "patient_view" in SYSTEM_PROMPT


def test_prompt_bans_romanized_dialect():
    assert "Never use romanized or transliterated dialect text" in SYSTEM_PROMPT


def test_prompt_keeps_machine_readable_fields_stable():
    assert "Keep ICD-10 codes unchanged" in SYSTEM_PROMPT
    assert "Low, Medium, or High in English" in SYSTEM_PROMPT


def test_prompt_has_no_hardcoded_score_example():
    assert '"primary_vector_match_score": 0.9' not in SYSTEM_PROMPT
    assert "overwrites your values" in SYSTEM_PROMPT


def test_measured_telemetry_thresholds():
    build = ClinicalMathEngine.build_verification_telemetry
    high = build([{
        "phonetic_cosine_similarity": 0.83,
        "clinical_translation": "Acute gastritis",
    }])
    assert high == {
        "primary_vector_match_score": 0.83,
        "grounding_source_text": "Acute gastritis",
        "hallucination_risk_status": "Low",
    }
    medium = build([{
        "phonetic_cosine_similarity": 0.37,
        "matched_idiom": "ফেডত জ্বালা",
    }])
    assert medium["hallucination_risk_status"] == "Medium"
    assert medium["grounding_source_text"] == "ফেডত জ্বালা"
    below_floor = build([{
        "phonetic_cosine_similarity": 0.19,
        "clinical_translation": "Wrong term",
    }])
    assert below_floor["primary_vector_match_score"] == 0.19
    assert below_floor["grounding_source_text"] == ""
    assert below_floor["hallucination_risk_status"] == "High"


def test_measured_telemetry_empty_and_bad_inputs():
    build = ClinicalMathEngine.build_verification_telemetry
    empty = build([])
    assert empty["primary_vector_match_score"] == 0.0
    assert empty["grounding_source_text"] == ""
    assert empty["hallucination_risk_status"] == "High"
    bad = build([{"phonetic_cosine_similarity": "not-a-number"}])
    assert bad["primary_vector_match_score"] == 0.0
    assert bad["hallucination_risk_status"] == "High"
    nan = build([{"phonetic_cosine_similarity": float("nan")}])
    assert nan["primary_vector_match_score"] == 0.0
    assert nan["hallucination_risk_status"] == "High"


def test_local_align_fallback_scores_without_supabase(monkeypatch):
    engine = ClinicalMathEngine()
    engine.supabase = None
    assert engine.dialect_lexicon, "lexicon fixture is empty"
    phrase = engine.dialect_lexicon[0]
    query = phrase["dialect_phrase"] + " " + phrase["phonetic_roman"]
    matches = engine.align_regional_dialect_symptoms(query, top_k=1)
    assert len(matches) == 1
    top = matches[0]
    assert top["phonetic_cosine_similarity"] == pytest.approx(1.0)
    assert top["clinical_translation"] == phrase["clinical_term"]
    assert engine.align_regional_dialect_symptoms("", top_k=1) == []


BANGLA_CONTENT = json.dumps(
    {
        "doctor_view": {
            "clinical_summary": "রোগীর পেটে ব্যথা।",
            "dialect_icd10_alignment": ["R10.13 পেটে ব্যথা"],
            "active_triage_question": "ব্যথা কতক্ষণ ধরে?",
            "grounded_medications": [],
            "verification_telemetry": {
                "primary_vector_match_score": 0.9,
                "grounding_source_text": "তেল-মশলা থেকে ব্যথা।",
                "hallucination_risk_status": "Low",
            },
        },
        "patient_view": {
            "layman_explanation": "পেটে ব্যথা হয়েছে।",
            "next_steps": "ডাক্তার দেখান।",
        },
    },
    ensure_ascii=False,
)


class _FakeResp:
    status_code = 200
    text = BANGLA_CONTENT

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": BANGLA_CONTENT}}]}


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return _FakeResp()


@pytest.mark.asyncio
async def test_bangla_response_survives_cp1252_console(monkeypatch):
    monkeypatch.setattr(clinical_agent.AIConfig, "META_API_KEY", "test-key")
    buf = io.BytesIO()
    fake_stdout = io.TextIOWrapper(
        buf, encoding="cp1252", errors="strict", write_through=True
    )
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    with patch("core.clinical_agent.httpx.AsyncClient", _FakeClient):
        result = await clinical_agent.generate_clinical_insights("পেটে ব্যথা", {})
    assert result["doctor_view"]["clinical_summary"] == "রোগীর পেটে ব্যথা।"
    assert result["patient_view"]["layman_explanation"] == "পেটে ব্যথা হয়েছে।"
