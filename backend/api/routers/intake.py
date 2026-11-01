from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile

from agents.orchestrator import IntakeOrchestrator

router = APIRouter()
orchestrator = IntakeOrchestrator()

@router.post("/api/v2/intake-pipeline")
async def intake_pipeline(
    prescription: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    xray_scan: Optional[UploadFile] = File(None)
):
    try:
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
