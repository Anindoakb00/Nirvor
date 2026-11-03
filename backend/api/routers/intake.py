from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import os
import cv2
import numpy as np

from agents.orchestrator import IntakeOrchestrator
from config.settings import AIConfig

router = APIRouter()
orchestrator = IntakeOrchestrator()

@router.get("/api/v2/system-status")
def system_status():
    return {
        "meta_configured": bool(AIConfig.META_API_KEY and AIConfig.META_API_KEY != "your_key_here" and AIConfig.META_API_KEY != "unconfigured"),
        "baseten_configured": bool(AIConfig.BASETEN_API_KEY and AIConfig.BASETEN_API_KEY != "your_key_here" and AIConfig.BASETEN_API_KEY != "unconfigured"),
        "modal_configured": AIConfig.USE_MODAL_CLOUD_CV,
        "version": "0.0.3"
    }

from models.schemas import CompleteIntakeResponse

@router.post("/api/v2/intake-pipeline", response_model=CompleteIntakeResponse)
async def intake_pipeline(
    prescription: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    xray_scan: Optional[UploadFile] = File(None),
    demo_mode: bool = Form(False)
):
    try:
        if not prescription and not demo_mode:
            raise HTTPException(status_code=400, detail="Prescription file is required unless demo_mode is True.")

        prescription_bytes = None
        if prescription:
            prescription_bytes = await prescription.read()
        elif demo_mode:
            # Generate a blank white image for demo mode
            dummy_img = np.full((600, 400, 3), 255, dtype=np.uint8)
            _, encoded = cv2.imencode(".png", dummy_img)
            prescription_bytes = encoded.tobytes()

        audio_bytes = None
        audio_filename = None
        if audio:
            audio_bytes = await audio.read()
            audio_filename = audio.filename
        elif demo_mode:
            # Generate a blank valid wav file using standard header
            import wave
            import io
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(b'\x00\x00' * 44100) # 1 sec of silence
            audio_bytes = wav_io.getvalue()
            audio_filename = "demo.wav"

        xray_bytes = None
        if xray_scan:
            xray_bytes = await xray_scan.read()
        elif demo_mode:
            # Generate a blank image for X-ray
            dummy_xray = np.zeros((256, 256, 3), dtype=np.uint8)
            _, encoded_xray = cv2.imencode(".png", dummy_xray)
            xray_bytes = encoded_xray.tobytes()

        return await orchestrator.process_intake(
            prescription_bytes=prescription_bytes,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            xray_bytes=xray_bytes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
