import copy
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agents.orchestrator import IntakeOrchestrator
from api.routers.intake import orchestrator as route_orchestrator
from main import app

client = TestClient(app)

CLINICAL_OK = {
    "doctor_view": {
        "clinical_summary": "ব্যথা।",
        "dialect_icd10_alignment": ["R10.9 ব্যথা"],
        "active_triage_question": "কতদিন?",
        "grounded_medications": [],
        "verification_telemetry": {
            "primary_vector_match_score": 0.0,
            "grounding_source_text": "",
            "hallucination_risk_status": "Low",
        },
    },
    "patient_view": {"layman_explanation": "ব্যথা।", "next_steps": "ডাক্তার।"},
}

CLINICAL_ERROR = {
    "doctor_view": {
        "clinical_summary": "BACKEND ERROR boom",
        "dialect_icd10_alignment": [],
        "active_triage_question": "Check terminal.",
        "grounded_medications": [],
        "verification_telemetry": {
            "primary_vector_match_score": 0.0,
            "grounding_source_text": "",
            "hallucination_risk_status": "High",
        },
    },
    "patient_view": {"layman_explanation": "Error.", "next_steps": ""},
}


def _local_only(monkeypatch):
    monkeypatch.setattr(route_orchestrator.math_engine, "supabase", None)


def test_combined_empty_400():
    res = client.post("/api/v2/intake/combined")
    assert res.status_code == 400


def test_combined_voice_only_success(monkeypatch):
    _local_only(monkeypatch)
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "পেটে ব্যথা"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_OK))):
        res = client.post("/api/v2/intake/combined",
                          files={"audio": ("t.wav", b"fake-audio", "audio/wav")})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["voice"]["doctor_view"]["clinical_summary"] == "ব্যথা।"
    assert body["prescription"] is None and body["xray"] is None
    tele = body["voice"]["doctor_view"]["verification_telemetry"]
    assert tele["hallucination_risk_status"] in ("Low", "Medium", "High")
    assert 0.0 <= tele["primary_vector_match_score"] <= 1.0


def test_combined_partial_survives_bad_prescription(monkeypatch):
    _local_only(monkeypatch)
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "পেটে ব্যথা"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_OK))):
        res = client.post("/api/v2/intake/combined", files={
            "audio": ("t.wav", b"fake-audio", "audio/wav"),
            "prescription": ("rx.jpg", b"not-an-image", "image/jpeg"),
        })
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "partial_success"
    assert body["voice"] is not None
    assert body["prescription"] is None
    assert any("Prescription" in e for e in body["error_details"])


def test_voice_route_parity_after_refactor(monkeypatch):
    _local_only(monkeypatch)
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "পেটে ব্যথা"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_OK))):
        res = client.post("/api/v2/intake/voice",
                          files={"audio": ("t.wav", b"fake-audio", "audio/wav")})
    assert res.status_code == 200
    body = res.json()
    assert body["doctor_view"]["clinical_summary"] == "ব্যথা।"
    assert "verification_telemetry" in body["doctor_view"]


def test_voice_route_transcribe_failure_shape(monkeypatch):
    _local_only(monkeypatch)
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(side_effect=RuntimeError("Transcription failed: down"))):
        res = client.post("/api/v2/intake/voice",
                          files={"audio": ("t.wav", b"fake-audio", "audio/wav")})
    assert res.status_code == 200
    assert res.json()["error"].startswith("ROUTER CRASH: Transcription failed")


@pytest.mark.asyncio
async def test_branch_keeps_backend_error_telemetry():
    orch = IntakeOrchestrator()
    orch.math_engine.supabase = None
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "x"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_ERROR))):
        result, err = await orch.run_voice_branch(b"fake-audio")
    assert err is None
    tele = result["doctor_view"]["verification_telemetry"]
    assert tele == {"primary_vector_match_score": 0.0,
                    "grounding_source_text": "",
                    "hallucination_risk_status": "High"}


RX_EXTRACTION = {
    "medications": [{
        "raw_text": "Tab. Sarjel 20mg",
        "dosage": "20mg",
        "frequency": "1+0+1",
        "duration_days": 14,
    }],
    "patient_vitals": {"bp": "120/80", "pulse": "72", "temp": "98.6",
                       "creatinine_mg_dl": 1.0},
    "raw_symptoms": [],
}


def _white_jpeg_bytes():
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf.tobytes())


def _mock_rx_llm():
    return (
        patch("agents.orchestrator.extract_prescription_multimodal",
              AsyncMock(return_value=copy.deepcopy(RX_EXTRACTION))),
        patch("agents.orchestrator.generate_entropy_triage_question",
              AsyncMock(return_value={"question": "q?",
                                      "provider_used": "mock"})),
        patch("agents.orchestrator.generate_patient_dosage_card",
              AsyncMock(return_value={"url": "http://card",
                                      "provider_used": "mock"})),
    )


def test_single_prescription_defaults_unchanged(monkeypatch):
    """Single-rx route still grounds without voice context (cos_voice == 0)."""
    _local_only(monkeypatch)
    p_extract, p_triage, p_card = _mock_rx_llm()
    with p_extract, p_triage, p_card:
        res = client.post("/api/v2/intake/prescription", files={
            "prescription": ("rx.jpg", _white_jpeg_bytes(), "image/jpeg"),
        })
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    meds = body["grounded_medications"]
    assert len(meds) == 1
    assert meds[0]["matched_drug"]["brand_name"] == "Sergel 20"
    assert meds[0]["cos_voice"] == 0.0


def test_combined_voice_transcript_reaches_drug_grounding(monkeypatch):
    """Fusion: voice transcript flows into rx Bayesian grounding + OCR prompt."""
    _local_only(monkeypatch)
    p_extract, p_triage, p_card = _mock_rx_llm()
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "Sergel 20"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_OK))), \
         p_extract as mock_extract, p_triage, p_card:
        res = client.post("/api/v2/intake/combined", files={
            "audio": ("t.wav", b"fake-audio", "audio/wav"),
            "prescription": ("rx.jpg", _white_jpeg_bytes(), "image/jpeg"),
        })
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["voice"] is not None and body["prescription"] is not None
    meds = body["prescription"]["grounded_medications"]
    assert len(meds) == 1
    assert meds[0]["matched_drug"]["brand_name"] == "Sergel 20"
    assert meds[0]["cos_voice"] > 0.0
    _, dialect_ctx = mock_extract.call_args[0]
    assert isinstance(dialect_ctx, str) and len(dialect_ctx) > 0
    fusion = body["cross_modal_fusion"]
    assert fusion["modalities_present"] == ["voice", "prescription"]
    assert fusion["modalities_succeeded"] == ["voice", "prescription"]
    assert 0.0 <= fusion["fused_confidence"] <= 1.0


def test_combined_voice_only_reports_fusion(monkeypatch):
    """Fusion block exists for a single modality and mirrors its telemetry."""
    _local_only(monkeypatch)
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "পেটে ব্যথা"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_OK))):
        res = client.post("/api/v2/intake/combined",
                          files={"audio": ("t.wav", b"fake-audio",
                                           "audio/wav")})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    tele = body["voice"]["doctor_view"]["verification_telemetry"]
    fusion = body["cross_modal_fusion"]
    assert fusion["modalities_present"] == ["voice"]
    assert fusion["modalities_succeeded"] == ["voice"]
    assert fusion["voice_confidence"] == tele["primary_vector_match_score"]
    assert fusion["overall_risk"] == tele["hallucination_risk_status"]
    assert fusion["is_coherent"] is True


def test_combined_xray_biometrics_reach_voice_and_fusion(monkeypatch):
    """Fusion: measured X-ray biometrics flow into voice context + fusion."""
    _local_only(monkeypatch)
    thorax = np.zeros((20, 50), dtype=np.uint8)
    thorax[:, 5:45] = 1
    heart = np.zeros((20, 50), dtype=np.uint8)
    heart[:, 10:35] = 1
    lesion = np.zeros((20, 50), dtype=np.uint8)
    lesion[5:15, 20:30] = 1
    masks = {"heart_mask": heart, "thorax_mask": thorax,
             "lesion_mask": lesion}
    with patch("agents.orchestrator.transcribe_audio_meta",
               AsyncMock(return_value={"transcript": "বুকে চাপ"})), \
         patch("agents.orchestrator.generate_clinical_insights",
               AsyncMock(return_value=copy.deepcopy(CLINICAL_OK))) as mock_clinical, \
         patch("agents.orchestrator.segment_medical_scan_sam3",
               AsyncMock(return_value=masks)):
        res = client.post("/api/v2/intake/combined", files={
            "audio": ("t.wav", b"fake-audio", "audio/wav"),
            "xray_scan": ("x.jpg", b"fake-xray", "image/jpeg"),
        })
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    ctr = body["xray"]["scan_biometrics"]["cardiothoracic_ratio"]
    assert ctr["ctr"] == 0.6154
    assert ctr["cardiomegaly_flagged"] is True
    _, telemetry = mock_clinical.call_args[0]
    assert "CTR=0.6154" in telemetry["xray_biometrics"]
    fusion = body["cross_modal_fusion"]
    assert fusion["modalities_present"] == ["voice", "xray"]
    assert fusion["xray_biometrics_present"] is True


def test_coherence_aligns_cardiac_symptoms_with_cardiomegaly():
    """Coherence: cardiac symptoms + cardiomegaly flag count as agreement."""
    orch = IntakeOrchestrator()
    report = orch.math_engine.verify_symptom_medication_coherence(
        [{"icd10_code": "R07.9", "clinical_translation": "chest pain"}],
        [{"matched_drug": {"category": "Proton Pump Inhibitor (PPI)"}}],
        xray_flags=["cardiomegaly"],
    )
    assert report["is_coherent"] is True
    assert any("cardiomegaly" in s for s in report["matched_indications"])
