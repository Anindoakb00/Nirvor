import pytest

from core.math_engine import ClinicalMathEngine
from core.meta_agent import escalate_polypharmacy_wonder


def _gi_med(verified):
    med = {"matched_drug": {"category": "Proton Pump Inhibitor (PPI)"}}
    if not verified:
        med["requires_active_verification"] = True
    return med


def _nsaid_pair():
    return [
        {"generic_name": "Ketorolac Tromethamine 10mg",
         "category": "NSAID (Non-Steroidal Anti-Inflammatory)"},
        {"generic_name": "Aceclofenac 100mg",
         "category": "NSAID (Non-Steroidal Anti-Inflammatory)"},
    ]


GI_SYMPTOMS = [{"icd10_code": "R10.13 / K21.9", "clinical_translation": "gastric"}]


def _coherence_engine():
    return ClinicalMathEngine.__new__(ClinicalMathEngine)


def test_coherence_ignores_unverified_drug():
    report = _coherence_engine().verify_symptom_medication_coherence(
        GI_SYMPTOMS, [_gi_med(False)]
    )
    assert report["is_coherent"] is False
    assert report["coherence_score"] == 0.0
    assert report["matched_indications"] == []
    assert any("require verification" in f for f in report["contradiction_flags"])


def test_coherence_uses_verified_subset():
    report = _coherence_engine().verify_symptom_medication_coherence(
        GI_SYMPTOMS, [_gi_med(True), _gi_med(False)]
    )
    assert report["is_coherent"] is True
    assert any("Gastric" in s for s in report["matched_indications"])
    assert any("1 of 2" in f for f in report["contradiction_flags"])


def test_coherence_unchanged_without_flags():
    report = _coherence_engine().verify_symptom_medication_coherence(
        GI_SYMPTOMS, [{"matched_drug": {"category": "Proton Pump Inhibitor (PPI)"}}]
    )
    assert report["is_coherent"] is True
    assert report["contradiction_flags"] == []


@pytest.mark.asyncio
async def test_polypharmacy_skips_unverified_pair():
    result = await escalate_polypharmacy_wonder(
        {"medications": _nsaid_pair(), "requires_verification": [True, False]}, {}
    )
    assert result["risk_level"] == "LOW"
    assert result["verified_medications_assessed"] == 1
    assert result["unverified_medications_skipped"] == 1
    assert "1 of 2" in result["clinical_mechanism"]


@pytest.mark.asyncio
async def test_polypharmacy_still_catches_verified_interaction():
    result = await escalate_polypharmacy_wonder(
        {"medications": _nsaid_pair(), "requires_verification": [False, False]}, {}
    )
    assert result["risk_level"] == "CRITICAL"
    assert result["verified_medications_assessed"] == 2
    assert result["unverified_medications_skipped"] == 0


@pytest.mark.asyncio
async def test_polypharmacy_defers_when_nothing_verified():
    result = await escalate_polypharmacy_wonder(
        {"medications": _nsaid_pair(), "requires_verification": [True, True]}, {}
    )
    assert result["risk_level"] == "MODERATE"
    assert "deferred" in result["clinical_mechanism"]
    assert result["verified_medications_assessed"] == 0
    assert result["unverified_medications_skipped"] == 2


def test_new_drugs_ground_correctly():
    engine = ClinicalMathEngine()
    cases = [
        ("Tab Penvik (250 mg) 1+1+1+1", "BD_DIMS_011"),
        ("Tab Furoclav (250mg) 1+0+1", "BD_DIMS_012"),
        ("Inj. Lasix (20mg/2ml) 1 amp I/V stat", "BD_DIMS_013"),
    ]
    for raw, expected_id in cases:
        res = engine.ground_drug_entity(raw, "")
        assert res["matched_drug"]["id"] == expected_id, raw
        assert res["requires_active_verification"] is False, raw
