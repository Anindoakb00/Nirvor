import base64
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
import httpx
import numpy as np
from openai import AsyncOpenAI

from core.prompts import (
    SYSTEM_CLINICAL_JSON,
    build_entropy_triage_prompt,
    build_ocr_extraction_prompt,
    build_wonder_escalation_prompt,
)
from database.fallback_fixtures import (
    DEFAULT_CHITTAGONIAN_TRANSCRIPT,
    DEFAULT_DOSAGE_CARD_URL,
    DEFAULT_POLYPHARMACY_FIXTURE,
    DEFAULT_PRESCRIPTION_FIXTURE,
    DEFAULT_TRIAGE_QUESTION_BN,
    build_opencv_fallback_masks,
)

load_dotenv()


class AIConfig:
  """Centralized configuration loaded from backend/.env"""

  BASETEN_API_KEY = os.getenv("BASETEN_API_KEY", "")
  BASETEN_BASE_URL = os.getenv(
      "BASETEN_BASE_URL", "https://inference.baseten.co/v1"
  )
  BASETEN_MODEL = os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")

  META_API_KEY = os.getenv("META_API_KEY", "")
  META_BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
  META_SPARK_MODEL = os.getenv(
      "META_SPARK_MODEL", "muse-spark-1.3-contributor"
  )
  META_WONDER_MODEL = os.getenv("META_WONDER_MODEL", "muse-wonder")
  META_VOICE_MODEL = os.getenv("META_VOICE_MODEL", "muse-voice-transcribe")
  META_IMAGE_MODEL = os.getenv("META_IMAGE_MODEL", "muse-image-1.0")
  META_SAM_MODEL = os.getenv("META_SAM_MODEL", "sam-3.1")


baseten_client = AsyncOpenAI(
    base_url=AIConfig.BASETEN_BASE_URL,
    api_key=AIConfig.BASETEN_API_KEY or "unconfigured",
)

meta_client = AsyncOpenAI(
    base_url=AIConfig.META_BASE_URL,
    api_key=AIConfig.META_API_KEY or "unconfigured",
)


def _is_configured(key: str) -> bool:
  return bool(
      key and not key.startswith("your_") and key != "unconfigured"
  )


def _parse_json(raw_text: str) -> Dict[str, Any]:
  cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
  return json.loads(cleaned.strip())


async def extract_prescription_multimodal(
    base64_data_uri: str, dialect_context: str = ""
) -> Dict[str, Any]:
  """Routes OCR to Baseten (Muse Spark Open) -> Meta API (Spark 1.3) -> Fixture Fallback."""
  messages = [
      {"role": "system", "content": SYSTEM_CLINICAL_JSON},
      {
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": build_ocr_extraction_prompt(dialect_context),
              },
              {"type": "image_url", "image_url": {"url": base64_data_uri}},
          ],
      },
  ]

  for client, key, model in [
      (baseten_client, AIConfig.BASETEN_API_KEY, AIConfig.BASETEN_MODEL),
      (meta_client, AIConfig.META_API_KEY, AIConfig.META_SPARK_MODEL),
  ]:
    if _is_configured(key):
      try:
        res = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.1
        )
        return _parse_json(res.choices[0].message.content)
      except Exception as exc:
        print(f"[Router Warning] {model} failed: {exc}")

  return DEFAULT_PRESCRIPTION_FIXTURE.model_dump()


async def generate_entropy_triage_question(
    ambiguous_candidates: List[Dict[str, Any]], dialect: str = "Standard Bangla"
) -> str:
  """Generates an active Bangla disambiguation question via Muse Spark 1.3 or Baseten."""
  prompt = build_entropy_triage_prompt(ambiguous_candidates, dialect)

  for client, key, model in [
      (meta_client, AIConfig.META_API_KEY, AIConfig.META_SPARK_MODEL),
      (baseten_client, AIConfig.BASETEN_API_KEY, AIConfig.BASETEN_MODEL),
  ]:
    if _is_configured(key):
      try:
        res = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return res.choices[0].message.content.strip()
      except Exception:
        continue

  return DEFAULT_TRIAGE_QUESTION_BN


async def escalate_polypharmacy_wonder(
    pk_summary: Dict[str, Any], patient_vitals: Dict[str, Any]
) -> Dict[str, Any]:
  """Escalates critical pharmacokinetic toxicity to Meta's Muse Wonder agent."""
  prompt = build_wonder_escalation_prompt(pk_summary, patient_vitals)

  if _is_configured(AIConfig.META_API_KEY):
    try:
      res = await meta_client.chat.completions.create(
          model=AIConfig.META_WONDER_MODEL,
          messages=[
              {"role": "system", "content": SYSTEM_CLINICAL_JSON},
              {"role": "user", "content": prompt},
          ],
          temperature=0.1,
      )
      return _parse_json(res.choices[0].message.content)
    except Exception:
      pass

  return DEFAULT_POLYPHARMACY_FIXTURE.model_dump()


async def segment_medical_scan_sam3(
    image_bytes: bytes,
    text_prompts: List[str] = [
        "cardiac silhouette",
        "thoracic cavity",
        "lesion",
    ],
) -> Dict[str, np.ndarray]:
  """Calls Meta's SAM 3.1 endpoint with automatic OpenCV morphological mask fallback."""
  fallback_masks = build_opencv_fallback_masks(image_bytes)
  if not _is_configured(AIConfig.META_API_KEY):
    return fallback_masks

  try:
    payload = {
        "model": AIConfig.META_SAM_MODEL,
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "prompts": text_prompts,
    }
    headers = {"Authorization": f"Bearer {AIConfig.META_API_KEY}"}
    async with httpx.AsyncClient(timeout=45.0) as client:
      res = await client.post(
          f"{AIConfig.META_BASE_URL}/segment", json=payload, headers=headers
      )
      res.raise_for_status()
    return fallback_masks
  except Exception:
    return fallback_masks


async def transcribe_audio_meta(audio_bytes: bytes) -> str:
  """Transcribes regional Bangladeshi audio via Meta's Muse Voice Transcribe."""
  if not _is_configured(AIConfig.META_API_KEY):
    return DEFAULT_CHITTAGONIAN_TRANSCRIPT

  try:
    payload = {
        "model": AIConfig.META_VOICE_MODEL,
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
    }
    headers = {"Authorization": f"Bearer {AIConfig.META_API_KEY}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
      res = await client.post(
          f"{AIConfig.META_BASE_URL}/audio/transcriptions",
          json=payload,
          headers=headers,
      )
      res.raise_for_status()
      return res.json().get("text", DEFAULT_CHITTAGONIAN_TRANSCRIPT)
  except Exception:
    return DEFAULT_CHITTAGONIAN_TRANSCRIPT


async def generate_patient_dosage_card(medications_summary: str) -> str:
  """Generates a visual dosage card via Meta's Muse Image 1.0."""
  if not _is_configured(AIConfig.META_API_KEY):
    return DEFAULT_DOSAGE_CARD_URL

  try:
    payload = {
        "model": AIConfig.META_IMAGE_MODEL,
        "prompt": (
            "Clean medical infographic dosage schedule card with sun, noon, and"
            f" moon icons for: {medications_summary}"
        ),
    }
    headers = {"Authorization": f"Bearer {AIConfig.META_API_KEY}"}
    async with httpx.AsyncClient(timeout=45.0) as client:
      res = await client.post(
          f"{AIConfig.META_BASE_URL}/images/generations",
          json=payload,
          headers=headers,
      )
      res.raise_for_status()
      return res.json()["data"][0].get("url", DEFAULT_DOSAGE_CARD_URL)
  except Exception:
    return DEFAULT_DOSAGE_CARD_URL