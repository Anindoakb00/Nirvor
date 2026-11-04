import json
import httpx
from openai import AsyncOpenAI
from config.settings import AIConfig

SYSTEM_PROMPT = """
You are a clinical reasoning AI. Analyze the patient transcript and telemetry.

You MUST return a STRICT JSON object that matches the following schema exactly:
{
  "doctor_view": {
    "clinical_summary": "Professional medical summary of the Bengali/regional symptoms.",
    "dialect_icd10_alignment": ["Mapped ICD-10 codes"],
    "active_triage_question": "One technical follow-up question.",
    "grounded_medications": ["Generic suggestions with a provider safety flag"],
    "verification_telemetry": {
      "primary_vector_match_score": "Float representing the database Cosine/Inner Product similarity score.",
      "grounding_source_text": "The exact retrieved clinical text from the Supabase pgvector database.",
      "hallucination_risk_status": "Low/Medium/High"
    }
  },
  "patient_view": {
    "layman_explanation": "A simple, jargon-free explanation of symptoms.",
    "next_steps": "Simple actionable advice."
  }
}
"""


async def generate_clinical_insights(transcript: str, telemetry: dict) -> dict:
    if not AIConfig.META_API_KEY or AIConfig.META_API_KEY.startswith(
            "your_") or AIConfig.META_API_KEY == "unconfigured":
        raise RuntimeError("Meta API key is unconfigured for clinical reasoning.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Transcript: {transcript}\n\nTelemetry: {json.dumps(telemetry)}"}
    ]

    try:
        payload = {
            "model": AIConfig.META_SPARK_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        headers = {
            "Authorization": f"Bearer {AIConfig.META_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            res = await client.post(
                f"{AIConfig.META_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            )
            print(f"Raw API response: {res.text}")
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
        print(f"CRITICAL BACKEND ERROR: {str(e)}")  # Forces it to print in terminal
        return {
            "doctor_view": {
                "clinical_summary": f"BACKEND ERROR: {str(e)}",
                "dialect_icd10_alignment": [],
                "active_triage_question": "Check Uvicorn terminal for raw LLM output.",
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