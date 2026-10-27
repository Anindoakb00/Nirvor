import os
import requests
from dotenv import load_dotenv

load_dotenv()
META_API_KEY = os.getenv("META_API_KEY")
BASE_URL = "https://api.meta.ai/v1"


def process_regional_voice_note(audio_bytes: bytes) -> dict:
  """Sends regional Bangla / Chittagonian voice notes to Muse Voice Transcribe."""
  endpoint = f"{BASE_URL}/audio/transcriptions"
  headers = {"Authorization": f"Bearer {META_API_KEY}"}

  files = {"file": ("audio_intake.wav", audio_bytes, "audio/wav")}
  data = {"model": "muse-voice-transcribe", "language": "bn", "diarize": "true"}

  # Fallback response simulation for offline/testing if API key is restricted
  if not META_API_KEY or META_API_KEY == "your_key_here":
    return {
        "transcript": "আঁর পেটত ব্যথা গরের আর মাথা ঘুরায়",
        "detected_dialect": "Chittagonian Bangla",
        "speakers": [{"speaker": "Patient", "timestamps": [0.0, 4.2]}],
    }

  response = requests.post(endpoint, headers=headers, files=files, data=data)
  if response.status_code != 200:
    raise Exception(f"Voice Transcribe Error: {response.text}")

  return response.json()