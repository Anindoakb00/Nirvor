import json
from typing import Any, Dict, List

SYSTEM_CLINICAL_JSON = (
    "You are an expert Bangladeshi clinical AI assistant. "
    "Always respond with valid, strict JSON only. Do not include markdown fences."
)


def build_ocr_extraction_prompt(dialect_context: str = "") -> str:
  return f"""
    Extract prescription information from this Bangladeshi clinical prescription image.
    Patient Dialect/Voice Context: {dialect_context}

    Return strict JSON ONLY with keys:
    - "medications": [{{"raw_text": str, "dosage": str, "frequency": str, "duration_days": int}}]
    - "patient_vitals": {{"bp": str, "pulse": str, "temp": str, "creatinine_mg_dl": float}}
    - "raw_symptoms": list[str]
    """


def build_entropy_triage_prompt(
    candidates: List[Dict[str, Any]], dialect: str
) -> str:
  return (
      "High Shannon entropy detected between drug candidates:"
      f" {json.dumps(candidates)}. Generate a single, empathetic clarifying"
      f" question for the patient in {dialect} (with English translation in"
      " parentheses) asking about their primary symptom or tablet appearance"
      " to disambiguate."
  )


def build_wonder_escalation_prompt(
    pk_summary: Dict[str, Any], vitals: Dict[str, Any]
) -> str:
  return f"""
    CRITICAL PHARMACOKINETIC ALERT:
    Pharmacokinetic Profile: {json.dumps(pk_summary)}
    Patient Vitals: {json.dumps(vitals)}

    Provide a strict JSON clinical intervention plan with keys:
    - "risk_level": "CRITICAL" | "HIGH" | "MODERATE" | "LOW"
    - "clinical_mechanism": str
    - "recommended_action": str
    - "bangla_patient_warning": str
    """