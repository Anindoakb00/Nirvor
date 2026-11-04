import json
from openai import AsyncOpenAI
from config.settings import AIConfig

meta_client = AsyncOpenAI(
    base_url=AIConfig.META_BASE_URL,
    api_key=AIConfig.META_API_KEY or "unconfigured",
)

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

async def analyze_clinical_transcript(transcript: str, telemetry: dict) -> dict:
    if not AIConfig.META_API_KEY or AIConfig.META_API_KEY.startswith("your_") or AIConfig.META_API_KEY == "unconfigured":
        raise RuntimeError("Meta API key is unconfigured for clinical reasoning.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Transcript: {transcript}\n\nTelemetry: {json.dumps(telemetry)}"}
    ]

    try:
        res = await meta_client.chat.completions.create(
            model=AIConfig.META_SPARK_MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = res.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Clinical reasoning failed: {e}")
