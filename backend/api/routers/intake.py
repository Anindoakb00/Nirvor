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
        "modal_configured": AIConfig.USE_MODAL_CLOUD_CV or AIConfig.MODAL_TOKEN_EXISTS,
        "version": "0.0.3"
    }

from models.schemas import CompleteIntakeResponse

@router.post("/api/v2/intake-pipeline", response_model=CompleteIntakeResponse)
async def intake_pipeline(
    prescription: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    xray_scan: Optional[UploadFile] = File(None),
):
    try:
        prescription_bytes = None
        if prescription:
            prescription_bytes = await prescription.read()

        audio_bytes = None
        audio_filename = None
        if audio:
            audio_bytes = await audio.read()
            audio_filename = audio.filename

        xray_bytes = None
        if xray_scan:
            xray_bytes = await xray_scan.read()

        return await orchestrator.process_intake(
            prescription_bytes=prescription_bytes,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            xray_bytes=xray_bytes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
