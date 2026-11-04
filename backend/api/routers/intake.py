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
        "modal_configured": AIConfig.USE_MODAL_CLOUD_CV or AIConfig.MODAL_TOKEN_EXISTS,
        "version": "0.0.3"
    }

from models.schemas import CompleteIntakeResponse, ClinicalAnalysisResponse

@router.post("/api/v2/intake/voice", response_model=ClinicalAnalysisResponse)
async def intake_voice(
    audio: Optional[UploadFile] = File(None),
):
    audio_bytes = None
    audio_filename = None
    if audio and audio.filename:
        audio_bytes = await audio.read()
        audio_filename = audio.filename

    if not audio_bytes:
         raise HTTPException(status_code=400, detail="Missing audio file")

    try:
        return await orchestrator.process_voice(
            audio_bytes=audio_bytes,
            audio_filename=audio_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v2/intake/prescription", response_model=CompleteIntakeResponse)
async def intake_prescription(
    prescription: Optional[UploadFile] = File(None),
):
    prescription_bytes = None
    if prescription and prescription.filename:
        prescription_bytes = await prescription.read()

    if not prescription_bytes:
         raise HTTPException(status_code=400, detail="Missing prescription image")

    try:
        return await orchestrator.process_prescription(
            prescription_bytes=prescription_bytes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v2/intake/xray", response_model=CompleteIntakeResponse)
async def intake_xray(
    xray_scan: Optional[UploadFile] = File(None),
):
    xray_bytes = None
    if xray_scan and xray_scan.filename:
        xray_bytes = await xray_scan.read()

    if not xray_bytes:
         raise HTTPException(status_code=400, detail="Missing xray image")

    try:
        return await orchestrator.process_xray(
            xray_bytes=xray_bytes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
