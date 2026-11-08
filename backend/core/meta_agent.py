import base64
import io
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
import httpx
import numpy as np
from pydub import AudioSegment
from openai import AsyncOpenAI

from core.prompts import (
    SYSTEM_CLINICAL_JSON,
    build_entropy_triage_prompt,
    build_ocr_extraction_prompt,
    build_wonder_escalation_prompt,
)
from config.settings import AIConfig
from core.dosage_card import parse_dosage_summary, render_dosage_card_png
from core.polypharmacy import PolypharmacyGraph

load_dotenv()


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
  """Routes OCR to Meta API (Spark 1.3)."""
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

  if _is_configured(AIConfig.META_API_KEY):
    try:
      res = await meta_client.chat.completions.create(
          model=AIConfig.META_SPARK_MODEL, messages=messages, temperature=0.1
      )
      parsed = _parse_json(res.choices[0].message.content)
      parsed["provider_used"] = "meta_api"
      return parsed
    except Exception as exc:
      raise RuntimeError(f"Multimodal extraction failed: {exc}")

  raise RuntimeError("Meta API is unconfigured for multimodal extraction.")


async def generate_entropy_triage_question(
    ambiguous_candidates: List[Dict[str, Any]], dialect: str = "Standard Bangla"
) -> Dict[str, Any]:
  """Generates an active Bangla disambiguation question via Muse Spark 1.3."""
  prompt = build_entropy_triage_prompt(ambiguous_candidates, dialect)

  if _is_configured(AIConfig.META_API_KEY):
    try:
      res = await meta_client.chat.completions.create(
          model=AIConfig.META_SPARK_MODEL,
          messages=[{"role": "user", "content": prompt}],
          temperature=0.2,
      )
      return {"question": res.choices[0].message.content.strip(), "provider_used": "meta_api"}
    except Exception as exc:
      raise RuntimeError(f"Triage question generation failed: {exc}")

  raise RuntimeError("Meta API is unconfigured for generating triage question.")


async def escalate_polypharmacy_wonder(
    pk_summary: Dict[str, Any], patient_vitals: Dict[str, Any]
) -> Dict[str, Any]:
  """Escalates critical pharmacokinetic toxicity by querying the deterministic PolypharmacyGraph.

  Only verified medication matches are assessed. Unverified matches are
  excluded and counted so an interaction check is never "clean" on the
  basis of flagged guesses.
  """
  graph = PolypharmacyGraph()
  medications = pk_summary.get("medications", [])
  flags = pk_summary.get("requires_verification", [False] * len(medications))
  verified = [m for m, f in zip(medications, flags) if not f]
  skipped = len(medications) - len(verified)
  if medications and not verified:
    return {
        "risk_level": "MODERATE",
        "clinical_mechanism": (
            f"Polypharmacy assessment deferred: all {len(medications)} medication "
            "matches require pharmacist verification; no verified drugs to assess."
        ),
        "recommended_action": "Verify the flagged medications before any interaction assessment.",
        "bangla_patient_warning": "যাচাই না হওয়া পর্যন্ত ওষুধের মিথস্ক্রিয়া মূল্যায়ন করা যায়নি। ফার্মাসিস্টের পরামর্শ নিন।",
        "provider_used": "deterministic_graph",
        "verified_medications_assessed": 0,
        "unverified_medications_skipped": skipped,
    }
  result = graph.check_interaction(verified)
  result["verified_medications_assessed"] = len(verified)
  result["unverified_medications_skipped"] = skipped
  if skipped:
    result["clinical_mechanism"] += (
        f" Assessment covers {len(verified)} of {len(medications)} "
        f"medications; {skipped} unverified excluded."
    )
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
  """Transcribes audio via Meta Muse Voice (POST {base}/asr/transcribe, parts request+audio)."""
  if not _is_configured(AIConfig.META_API_KEY):
    raise RuntimeError("Meta API key is unconfigured for transcription.")

  try:
    # Convert .m4a to .wav and downsample
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_bytes = wav_io.getvalue()

    files_payload = {
        "request": (None, json.dumps({"model": AIConfig.META_VOICE_MODEL, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}), "application/json"),
        "audio": ("recording.wav", wav_bytes, "audio/wav")
    }

    headers = {
        "Authorization": f"Bearer {AIConfig.META_API_KEY}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
      res = await client.post(
          f"{AIConfig.META_BASE_URL.rstrip('/')}/asr/transcribe",
          files=files_payload,
          headers=headers,
      )
      res.raise_for_status()
      body = res.json()
      return {
          "transcript": body.get("transcript", ""),
          "sessionId": body.get("sessionId", ""),
          "audioDurationMs": body.get("audioDurationMs", 0),
          "turns": body.get("turns", []),
          "provider_used": "meta_api",
      }
  except Exception as exc:
    raise RuntimeError(f"Transcription failed: {exc} | model={AIConfig.META_VOICE_MODEL} | meta_body={getattr(getattr(exc, 'response', None), 'text', '')[:500]}")


async def generate_patient_dosage_card(medications_summary: str) -> Dict[str, Any]:
  """Renders the patient dosage card locally as a PNG data URI.

  Meta's image model refuses medication prompts by content policy, so no
  network call is made: the card is drawn deterministically with PIL and
  returned inline, usable directly as an <img> source.
  """
  rows = parse_dosage_summary(medications_summary or "")
  png = render_dosage_card_png(rows)
  uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
  return {"url": uri, "provider_used": "local_renderer"}
