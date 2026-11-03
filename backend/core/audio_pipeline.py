import os
import httpx
import time
from dotenv import load_dotenv

load_dotenv()
META_API_KEY = os.getenv("META_API_KEY")
META_BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")

async def process_regional_voice_note(audio_bytes: bytes, filename: str = "intake.wav") -> dict:
    """Sends regional Bangla / Chittagonian voice notes to Muse Voice Transcribe."""
    endpoint = f"{META_BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {META_API_KEY}"}

    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"model": "muse-voice-transcribe", "language": "bn", "diarize": "true"}

    if not META_API_KEY or META_API_KEY == "your_key_here":
        raise RuntimeError("META_API_KEY is not configured for audio transcription.")

    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000
            result["latency_ms"] = latency_ms
            return result
    except Exception as e:
        raise RuntimeError(f"Audio transcription failed: {e}")
