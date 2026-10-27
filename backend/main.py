import time
import asyncio
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.audio_processor import process_regional_voice_note
from core.cv_preprocessing import ClinicalCVPreprocessor
from core.math_engine import ClinicalMathEngine
from core.meta_client import extract_prescription_multimodal, segment_medical_scan_sam3

app = FastAPI(
    title="Nirvor - A Neuro-Symbolic Engine",
    version="2.0.0",
    description="Meta Global AI Developer Hackathon Submission - Team Anindo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

math_engine = ClinicalMathEngine()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Neuro-Symbolic Bayesian OCR & Regional Voice Fusion",
        "team": "Team Anindo",
    }

@app.post("/api/v2/intake-pipeline")
async def intake_pipeline(
    prescription: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    xray_scan: Optional[UploadFile] = File(None)
):
    try:
        t0 = time.perf_counter()

        # b) Read prescription bytes and run ClinicalCVPreprocessor
        prescription_bytes = await prescription.read()
        cv_result = ClinicalCVPreprocessor.process_prescription_image(prescription_bytes)
        base64_data_uri = cv_result["base64_data_uri"]
        cv_telemetry = cv_result["telemetry"]

        # c) Concurrently execute process_regional_voice_note and segment_medical_scan_sam3
        audio_task = None
        xray_task = None

        if audio:
            audio_bytes = await audio.read()
            audio_task = process_regional_voice_note(audio_bytes, audio.filename)

        if xray_scan:
            xray_bytes = await xray_scan.read()
            xray_task = segment_medical_scan_sam3(xray_bytes)

        tasks = []
        if audio_task:
            tasks.append(audio_task)
        if xray_task:
            tasks.append(xray_task)

        results = await asyncio.gather(*tasks) if tasks else []

        voice_intake = None
        masks = None

        result_idx = 0
        if audio_task:
            voice_intake = results[result_idx]
            result_idx += 1
        if xray_task:
            masks = results[result_idx]

        # d) Pass voice transcript into math_engine.align_regional_dialect_symptoms
        dialect_icd10_alignment = []
        transcript = ""
        if voice_intake and "transcript" in voice_intake:
            transcript = voice_intake["transcript"]
            dialect_icd10_alignment = math_engine.align_regional_dialect_symptoms(transcript)

        # e) Format dialect context and call multimodal extraction
        dialect_context_str = ", ".join([f"{a['matched_idiom']} ({a['clinical_translation']})" for a in dialect_icd10_alignment])
        extracted = await extract_prescription_multimodal(base64_data_uri, dialect_context_str)

        # f) Ground drug entities
        grounded_medications = []
        ocr_typos_corrected_count = 0
        total_shannon_entropy = 0.0
        active_verification_flags = 0

        if "medications" in extracted:
            for med in extracted["medications"]:
                raw_text = med.get("raw_text", "")
                grounded_result = math_engine.ground_drug_entity(raw_text, transcript)

                # Merge grounded result into med dictionary
                med_enriched = {**med, **grounded_result}
                grounded_medications.append(med_enriched)

                if grounded_result.get("ocr_typo_corrected"):
                    ocr_typos_corrected_count += 1
                total_shannon_entropy += grounded_result.get("shannon_entropy_bits", 0.0)
                if grounded_result.get("requires_active_verification"):
                    active_verification_flags += 1

        mean_shannon_entropy = 0.0
        if grounded_medications:
            mean_shannon_entropy = total_shannon_entropy / len(grounded_medications)

        # g) If xray_scan was provided, run math engine functions
        scan_biometrics = {}
        if masks:
            heart_mask = masks.get("heart_mask")
            thorax_mask = masks.get("thorax_mask")
            lesion_mask = masks.get("lesion_mask")

            if heart_mask is not None and thorax_mask is not None:
                ctr_res = math_engine.calculate_cardiothoracic_ratio(heart_mask, thorax_mask)
                scan_biometrics["cardiothoracic_ratio"] = ctr_res
            if lesion_mask is not None:
                lesion_res = math_engine.analyze_mask_eigen_geometry(lesion_mask)
                scan_biometrics["lesion_geometry"] = lesion_res

        total_latency_ms = (time.perf_counter() - t0) * 1000

        judge_hud_summary = {
            "total_latency_ms": round(total_latency_ms, 2),
            "hallucinations_intercepted": ocr_typos_corrected_count,
            "mean_shannon_entropy": round(mean_shannon_entropy, 4),
            "active_verification_flags": active_verification_flags,
        }

        # h) Return final clean JSON
        return {
            "status": "success",
            "cv_telemetry": cv_telemetry,
            "voice_intake": voice_intake,
            "dialect_icd10_alignment": dialect_icd10_alignment,
            "grounded_medications": grounded_medications,
            "scan_biometrics": scan_biometrics,
            "judge_hud_summary": judge_hud_summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
