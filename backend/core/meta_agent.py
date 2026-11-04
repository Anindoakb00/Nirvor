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
from config.settings import AIConfig
from core.polypharmacy import PolypharmacyGraph

load_dotenv()


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
  """Routes OCR to Baseten (Muse Spark Open) -> Meta API (Spark 1.3)."""
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

  # Primary Meta -> Baseten
  for client, key, model, provider_name in [
      (meta_client, AIConfig.META_API_KEY, AIConfig.META_SPARK_MODEL, "meta_api"),
      (baseten_client, AIConfig.BASETEN_API_KEY, AIConfig.BASETEN_MODEL, "baseten_api"),
  ]:
    if _is_configured(key):
      try:
        res = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.1
        )
        parsed = _parse_json(res.choices[0].message.content)
        parsed["provider_used"] = provider_name
        return parsed
      except Exception as exc:
        print(f"[Router Warning] {model} failed: {exc}")

  raise RuntimeError("All providers failed or are unconfigured for multimodal extraction.")


async def generate_entropy_triage_question(
    ambiguous_candidates: List[Dict[str, Any]], dialect: str = "Standard Bangla"
) -> Dict[str, Any]:
  """Generates an active Bangla disambiguation question via Muse Spark 1.3 or Baseten."""
  prompt = build_entropy_triage_prompt(ambiguous_candidates, dialect)

  for client, key, model, provider_name in [
      (meta_client, AIConfig.META_API_KEY, AIConfig.META_SPARK_MODEL, "meta_api"),
      (baseten_client, AIConfig.BASETEN_API_KEY, AIConfig.BASETEN_MODEL, "baseten_api"),
  ]:
    if _is_configured(key):
      try:
        res = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return {"question": res.choices[0].message.content.strip(), "provider_used": provider_name}
      except Exception:
        continue

  raise RuntimeError("All providers failed or are unconfigured for generating triage question.")


async def escalate_polypharmacy_wonder(
    pk_summary: Dict[str, Any], patient_vitals: Dict[str, Any]
) -> Dict[str, Any]:
  """Escalates critical pharmacokinetic toxicity by querying the deterministic PolypharmacyGraph."""
  graph = PolypharmacyGraph()
  medications = pk_summary.get("medications", [])
  result = graph.check_interaction(medications)
  return result


async def segment_medical_scan_sam3(
    image_bytes: bytes,
    text_prompts: List[str] = [
        "cardiac silhouette",
        "thoracic cavity",
        "lesion",
    ],
) -> Dict[str, np.ndarray]:
  """Calls Meta's SAM 3.1 endpoint."""
  if not _is_configured(AIConfig.META_API_KEY):
    raise RuntimeError("Meta API key is unconfigured for SAM 3.1 segmentation.")

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
    masks = res.json()
    masks["provider_used"] = "meta_api"
    return masks
  except Exception as e:
    raise RuntimeError(f"Segmentation failed: {e}")


async def transcribe_audio_meta(audio_bytes: bytes) -> Dict[str, Any]:
  """Transcribes regional Bangladeshi audio via Meta's Muse Voice Transcribe."""
  if not _is_configured(AIConfig.META_API_KEY):
    raise RuntimeError("Meta API key is unconfigured for transcription.")

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
      return {"transcript": res.json().get("text", ""), "provider_used": "meta_api"}
  except Exception as exc:
    # Attempt Baseten fallback
    if _is_configured(AIConfig.BASETEN_API_KEY):
        try:
            b_payload = payload
            b_headers = {"Authorization": f"Bearer {AIConfig.BASETEN_API_KEY}"}
            async with httpx.AsyncClient(timeout=30.0) as b_client:
                b_res = await b_client.post(
                    f"{AIConfig.BASETEN_BASE_URL}/audio/transcriptions",
                    json=b_payload,
                    headers=b_headers,
                )
                b_res.raise_for_status()
                return {"transcript": b_res.json().get("text", ""), "provider_used": "baseten_api"}
        except Exception:
            pass

    raise RuntimeError(f"Transcription failed: {exc}")


async def generate_patient_dosage_card(medications_summary: str) -> Dict[str, Any]:
  """Generates a visual dosage card via Meta's Muse Image 1.0."""
  if not _is_configured(AIConfig.META_API_KEY):
    raise RuntimeError("Meta API key is unconfigured for dosage card generation.")

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
      return {"url": res.json()["data"][0].get("url", ""), "provider_used": "meta_api"}
  except Exception as exc:
    if _is_configured(AIConfig.BASETEN_API_KEY):
      try:
          b_headers = {"Authorization": f"Bearer {AIConfig.BASETEN_API_KEY}"}
          async with httpx.AsyncClient(timeout=45.0) as b_client:
              b_res = await b_client.post(
                  f"{AIConfig.BASETEN_BASE_URL}/images/generations",
                  json=payload,
                  headers=b_headers,
              )
              b_res.raise_for_status()
              return {"url": b_res.json()["data"][0].get("url", ""), "provider_used": "baseten_api"}
      except Exception:
          pass
    raise RuntimeError(f"Dosage card generation failed: {exc}")
