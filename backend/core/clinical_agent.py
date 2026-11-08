import json
import httpx
from config.settings import AIConfig

SYSTEM_PROMPT = """You are a clinical reasoning AI. Analyze the patient transcript and telemetry. Write EVERY user-facing text field in Bangla using Bengali script, in both doctor_view and patient_view. Never use romanized or transliterated dialect text.

You MUST return a STRICT JSON object that matches the following schema exactly. Do NOT wrap it in markdown. Do NOT add any extra text. ONLY return the JSON. Keep ICD-10 codes unchanged and keep hallucination_risk_status as Low, Medium, or High in English. verification_telemetry is measured by the system and overwrites your values, so output 0.0, empty string, and Low as placeholders.
{
  "doctor_view": {
    "clinical_summary": "Professional medical summary written in Bangla (Bengali script).",
    "dialect_icd10_alignment": ["ICD-10 code plus Bangla description in Bengali script"],
    "active_triage_question": "One follow-up question written in Bangla (Bengali script).",
    "grounded_medications": ["Suggestion in Bangla with provider safety flag; keep the English generic drug name in parentheses"],
    "verification_telemetry": {
      "primary_vector_match_score": 0.0,
      "grounding_source_text": "",
      "hallucination_risk_status": "Low"
    }
  },
  "patient_view": {
    "layman_explanation": "A simple, jargon-free explanation in Bangla (Bengali script).",
    "next_steps": "Simple actionable advice in Bangla (Bengali script)."
  }
}"""


async def generate_clinical_insights(transcript: str, telemetry: dict) -> dict:
    if not AIConfig.META_API_KEY or AIConfig.META_API_KEY.startswith(
            "your_") or AIConfig.META_API_KEY == "unconfigured":
        raise RuntimeError("Meta API key is unconfigured for clinical reasoning.")

    # Meta's official documentation requires the 'developer' role for system instructions
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Transcript: {transcript}\n\nTelemetry: {json.dumps(telemetry)}"}
    ]

    # Stripped out unsupported OpenAI kwargs (like response_format) to prevent 400 errors
    payload = {
        "model": "muse-spark-1.3",
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {AIConfig.META_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            res = await client.post(
                "https://api.meta.ai/v1/chat/completions",
                json=payload,
                headers=headers
            )

            print(f"DEBUG - Meta HTTP Status: {res.status_code}")
            print(f"DEBUG - Meta Raw Response: {ascii(res.text)}")

            res.raise_for_status()

            content = res.json()["choices"][0]["message"]["content"]

            # Robust JSON Markdown Stripping
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]

            return json.loads(cleaned_content.strip())

    except Exception as e:
        print(f"CRITICAL BACKEND ERROR: {type(e).__name__}: {str(e)}")
        return {
            "doctor_view": {
                "clinical_summary": f"BACKEND ERROR {type(e).__name__}: {str(e)}",
                "dialect_icd10_alignment": [],
                "active_triage_question": "Check Uvicorn terminal for detailed logs.",
                "grounded_medications": [],
                "verification_telemetry": {
                    "primary_vector_match_score": 0.0,
                    "grounding_source_text": "",
                    "hallucination_risk_status": "High"
                }
            },
            "patient_view": {
                "layman_explanation": "Error processing request.",
                "next_steps": ""
            }
        }