from types import SimpleNamespace

from core.math_engine import ClinicalMathEngine
from database.import_prescription_referrals import build_referral_record
from models.schemas import CompleteIntakeResponse, ReferralTelemetry


def _engine_with(stub):
    engine = ClinicalMathEngine.__new__(ClinicalMathEngine)
    engine.supabase = stub
    return engine


class _RpcStub:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error
        self.calls = []

    def rpc(self, name, params):
        self.calls.append((name, params))
        return self

    def execute(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=self.data)


def test_build_medication_blob_joins_raw_texts():
    meds = [
        {"raw_text": "Tab. Sergel 20mg"},
        {"raw_text": "  "},
        {"raw_text": "Cap. Seclo 20mg"},
        {},
    ]
    assert ClinicalMathEngine.build_medication_blob(meds) == (
        "Tab. Sergel 20mg | Cap. Seclo 20mg"
    )


def test_build_medication_blob_empty():
    assert ClinicalMathEngine.build_medication_blob([]) == ""
    assert ClinicalMathEngine.build_medication_blob(None) == ""


def test_build_referral_record_shape():
    record = build_referral_record("42.jpeg", "a | b", 2, [0.1] * 512, "meta_api")
    assert record["source_file"] == "42.jpeg"
    assert record["medication_blob"] == "a | b"
    assert record["med_count"] == 2
    assert record["embedding"] == [0.1] * 512
    assert record["verified"] is False
    assert record["provider_used"] == "meta_api"


def test_match_referrals_maps_rpc_rows():
    stub = _RpcStub(data=[
        {
            "source_file": "7.jpeg",
            "medication_blob": "Tab. X",
            "med_count": 1,
            "verified": True,
            "similarity": 0.2,
        },
        {
            "source_file": "9.jpeg",
            "medication_blob": "Tab. Y",
            "med_count": 3,
            "verified": False,
            "similarity": 0.6,
        },
    ])
    matches = _engine_with(stub).match_prescription_referrals("Tab. X", top_k=2)
    assert stub.calls and stub.calls[0][0] == "match_prescriptions"
    assert len(matches) == 2
    assert matches[0]["source_file"] == "7.jpeg"
    assert matches[0]["referral_cosine_similarity"] == 0.8
    assert matches[0]["verified"] is True
    assert matches[1]["verified"] is False


def test_match_referrals_empty_without_backend_or_rows():
    assert _engine_with(None).match_prescription_referrals("Tab. X") == []
    stub = _RpcStub(data=[])
    assert _engine_with(stub).match_prescription_referrals("Tab. X") == []


def test_match_referrals_rpc_failure_returns_empty():
    stub = _RpcStub(error=RuntimeError("relation does not exist"))
    assert _engine_with(stub).match_prescription_referrals("Tab. X") == []


def test_match_referrals_empty_blob_skips_rpc():
    stub = _RpcStub(data=[{"source_file": "x"}])
    assert _engine_with(stub).match_prescription_referrals("") == []
    assert stub.calls == []


def test_build_referral_telemetry_thresholds():
    def tele(score):
        return ClinicalMathEngine.build_referral_telemetry([
            {"referral_cosine_similarity": score, "source_file": "s",
             "verified": True, "medication_blob": "m"}
        ])
    assert tele(0.9)["referral_confidence"] == "High"
    assert tele(0.4)["referral_confidence"] == "Medium"
    assert tele(0.1)["referral_confidence"] == "Low"
    assert tele(0.9)["top_referral_match_score"] == 0.9


def test_build_referral_telemetry_passthrough_and_empty():
    tele = ClinicalMathEngine.build_referral_telemetry([
        {"referral_cosine_similarity": 0.7, "source_file": "42.jpeg",
         "verified": False, "medication_blob": "Tab. Z"}
    ])
    assert tele["referral_source_file"] == "42.jpeg"
    assert tele["referral_verified"] is False
    assert tele["referral_medications"] == "Tab. Z"

    empty = ClinicalMathEngine.build_referral_telemetry([])
    assert empty["top_referral_match_score"] == 0.0
    assert empty["referral_confidence"] == "Low"

    bad = ClinicalMathEngine.build_referral_telemetry(
        [{"referral_cosine_similarity": "nan"}]
    )
    assert bad["top_referral_match_score"] == 0.0


def test_referral_telemetry_schema_roundtrip():
    tele = ReferralTelemetry(
        top_referral_match_score=0.7,
        referral_source_file="42.jpeg",
        referral_verified=False,
        referral_medications="Tab. Z",
        referral_confidence="High",
    )
    assert tele.referral_confidence == "High"

    resp = CompleteIntakeResponse(
        status="success",
        judge_hud_summary={
            "total_latency_ms": 1.0,
            "hallucinations_intercepted": 0,
            "mean_shannon_entropy": 0.0,
            "active_verification_flags": 0,
        },
        referral_telemetry=tele.model_dump(),
    )
    assert resp.referral_telemetry.referral_source_file == "42.jpeg"

    default = CompleteIntakeResponse(
        status="success",
        judge_hud_summary={
            "total_latency_ms": 1.0,
            "hallucinations_intercepted": 0,
            "mean_shannon_entropy": 0.0,
            "active_verification_flags": 0,
        },
    )
    assert default.referral_telemetry is None
