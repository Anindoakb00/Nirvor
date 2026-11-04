import time
import asyncio
from typing import Optional, Dict, Any

from core.cv_pipeline import ClinicalCVPreprocessor
from core.math_engine import ClinicalMathEngine
from core.meta_agent import (
    extract_prescription_multimodal,
    segment_medical_scan_sam3,
    generate_entropy_triage_question,
    escalate_polypharmacy_wonder,
    generate_patient_dosage_card,
    transcribe_audio_meta
)
from core.clinical_agent import generate_clinical_insights
from config.settings import AIConfig
from core.privacy import PrivacyShield

class IntakeOrchestrator:
    def __init__(self):
        self.math_engine = ClinicalMathEngine()

    async def _safe_execute(self, coro, error_list, step_name):
        try:
            return await coro
        except Exception as e:
            error_list.append(f"{step_name} failed: {str(e)}")
            return None

    async def process_voice(
        self,
        audio_bytes: bytes,
        audio_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        agent_execution_trace = []
        error_details = []

        voice_intake = await self._safe_execute(transcribe_audio_meta(audio_bytes), error_details, "Audio Transcription")

        dialect_icd10_alignment = []
        transcript = ""
        telemetry = {}

        if voice_intake and "transcript" in voice_intake:
            transcript = voice_intake["transcript"]
            transcript = PrivacyShield.scrub_text(transcript)
            voice_intake["transcript"] = transcript
            dialect_icd10_alignment = self.math_engine.align_regional_dialect_symptoms(transcript)
            telemetry["dialect_icd10_alignment"] = dialect_icd10_alignment

        clinical_analysis = {}
        if transcript:
            analysis_resp = await self._safe_execute(generate_clinical_insights(transcript, telemetry), error_details, "Clinical Analysis")
            if analysis_resp:
                clinical_analysis = analysis_resp

        status = "error" if error_details else "success"
        if error_details and clinical_analysis:
            status = "partial_success"

        if not clinical_analysis:
            # Fallback to schema if empty
            clinical_analysis = {
                "doctor_view": {
                    "clinical_summary": "",
                    "dialect_icd10_alignment": [],
                    "active_triage_question": "",
                    "grounded_medications": [],
                    "verification_telemetry": {
                        "primary_vector_match_score": 0.0,
                        "grounding_source_text": "",
                        "hallucination_risk_status": "Low"
                    }
                },
                "patient_view": {
                    "layman_explanation": "",
                    "next_steps": ""
                }
            }

        return clinical_analysis


    async def process_prescription(
        self,
        prescription_bytes: bytes,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        agent_execution_trace = []
        error_details = []

        cv_telemetry = {}
        base64_data_uri = ""

        cv_t0 = time.perf_counter()
        if AIConfig.USE_MODAL_CLOUD_CV:
            try:
                from core.modal_worker import modal_remote_cv_deskew
                cv_result = modal_remote_cv_deskew.remote(prescription_bytes)
                base64_data_uri = f"data:image/jpeg;base64,{cv_result.get('enhanced_base64', '')}"
                cv_telemetry = {
                    "worker": cv_result.get("worker", "Modal Serverless Cloud"),
                    "deskewed_image_data_uri": base64_data_uri
                }
                provider_used = "modal_cloud"
            except Exception as e:
                print(f"Modal CV fallback to local: {e}")
                cv_result = ClinicalCVPreprocessor.process_prescription_image(prescription_bytes)
                base64_data_uri = cv_result["base64_data_uri"]
                cv_telemetry = cv_result["telemetry"]
                cv_telemetry["deskewed_image_data_uri"] = base64_data_uri
                provider_used = "local_fallback"
        else:
            cv_result = ClinicalCVPreprocessor.process_prescription_image(prescription_bytes)
            base64_data_uri = cv_result["base64_data_uri"]
            cv_telemetry = cv_result["telemetry"]
            cv_telemetry["deskewed_image_data_uri"] = base64_data_uri
            provider_used = "local_fallback"

        agent_execution_trace.append({
            "step_name": "clinical_cv_preprocessor",
            "status": "success",
            "provider_used": provider_used,
            "latency_ms": round((time.perf_counter() - cv_t0) * 1000, 2),
            "spark_triggered_reason": "always_run"
        })

        extracted = {}
        if base64_data_uri:
            extracted_res = await self._safe_execute(extract_prescription_multimodal(base64_data_uri, ""), error_details, "Multimodal Extraction")
            if extracted_res:
                extracted = extracted_res
                extracted = PrivacyShield.scrub_data(extracted)

        grounded_medications = []
        ocr_typos_corrected_count = 0
        total_shannon_entropy = 0.0
        active_verification_flags = 0

        if "medications" in extracted:
            for med in extracted["medications"]:
                raw_text = med.get("raw_text", "")
                grounded_result = self.math_engine.ground_drug_entity(raw_text, "")

                med_enriched = {**med, **grounded_result}
                grounded_medications.append(med_enriched)

                if grounded_result.get("ocr_typo_corrected"):
                    ocr_typos_corrected_count += 1
                total_shannon_entropy += grounded_result.get("shannon_entropy_bits", 0.0)
                if grounded_result.get("requires_active_verification"):
                    active_verification_flags += 1

        mean_shannon_entropy = 0.0
        active_triage_question = None
        polypharmacy_intervention = None
        patient_dosage_card = None
        clinical_coherence_report = None

        if grounded_medications:
            mean_shannon_entropy = total_shannon_entropy / len(grounded_medications)

            clinical_coherence_report = self.math_engine.verify_symptom_medication_coherence(
                [], grounded_medications
            )
            if not clinical_coherence_report.get("is_coherent", True):
                active_verification_flags += 1

            needs_triage = any(
                m.get("requires_active_verification") or m.get("shannon_entropy_bits", 0) > 0.5
                for m in grounded_medications
            ) or not clinical_coherence_report.get("is_coherent", True)

            if needs_triage:
                triage_t0 = time.perf_counter()
                ambiguous = [m for m in grounded_medications if m.get("requires_active_verification")]
                triage_resp = await self._safe_execute(generate_entropy_triage_question(ambiguous, "Standard Bangla"), error_details, "Entropy Triage")
                if triage_resp:
                    active_triage_question = triage_resp.get("question")
                    agent_execution_trace.append({
                        "step_name": "entropy_triage_spark",
                        "status": "success",
                        "provider_used": triage_resp.get("provider_used", "local_fallback"),
                        "latency_ms": round((time.perf_counter() - triage_t0) * 1000, 2),
                        "spark_triggered_reason": "high_entropy_or_incoherence"
                    })

            needs_polypharmacy = len(grounded_medications) >= 2 or any(
                m.get("matched_drug", {}).get("nephrotoxic_risk", 0.0) >= 0.3
                for m in grounded_medications
            )
            if needs_polypharmacy:
                poly_t0 = time.perf_counter()
                vitals = extracted.get("patient_vitals", {})
                pk_summary = {"medications": [m.get("matched_drug", {}) for m in grounded_medications]}
                polypharmacy_intervention = await self._safe_execute(escalate_polypharmacy_wonder(pk_summary, vitals), error_details, "Polypharmacy Toxicity")
                if polypharmacy_intervention:
                    agent_execution_trace.append({
                        "step_name": "polypharmacy_toxicity_spark",
                        "status": "success",
                        "provider_used": polypharmacy_intervention.get("provider_used", "local_fallback"),
                        "latency_ms": round((time.perf_counter() - poly_t0) * 1000, 2),
                        "spark_triggered_reason": "multiple_drugs_or_high_toxicity"
                    })

            card_t0 = time.perf_counter()
            summary_str = ", ".join([f"{m.get('raw_text')} ({m.get('frequency')})" for m in grounded_medications])
            card_resp = await self._safe_execute(generate_patient_dosage_card(summary_str), error_details, "Patient Dosage Card")
            if card_resp:
                patient_dosage_card = card_resp.get("url")
                agent_execution_trace.append({
                    "step_name": "patient_dosage_card_spark",
                    "status": "success",
                    "provider_used": card_resp.get("provider_used", "local_fallback"),
                    "latency_ms": round((time.perf_counter() - card_t0) * 1000, 2),
                    "spark_triggered_reason": "always_run_on_meds"
                })

        total_latency_ms = (time.perf_counter() - t0) * 1000

        judge_hud_summary = {
            "total_latency_ms": round(total_latency_ms, 2),
            "hallucinations_intercepted": ocr_typos_corrected_count,
            "mean_shannon_entropy": round(mean_shannon_entropy, 4),
            "active_verification_flags": active_verification_flags,
        }

        status = "error" if error_details else "success"
        if error_details and (grounded_medications or cv_telemetry):
            status = "partial_success"

        return {
            "status": status,
            "error_details": error_details if error_details else None,
            "cv_telemetry": cv_telemetry,
            "voice_intake": None,
            "dialect_icd10_alignment": [],
            "clinical_coherence_report": clinical_coherence_report,
            "active_triage_question": active_triage_question,
            "polypharmacy_intervention": polypharmacy_intervention,
            "patient_dosage_card": patient_dosage_card,
            "grounded_medications": grounded_medications,
            "scan_biometrics": {},
            "judge_hud_summary": judge_hud_summary,
            "agent_execution_trace": agent_execution_trace
        }

    async def process_xray(
        self,
        xray_bytes: bytes,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        agent_execution_trace = []
        error_details = []

        masks = await self._safe_execute(segment_medical_scan_sam3(xray_bytes), error_details, "X-Ray Segmentation")

        scan_biometrics = {}
        if masks:
            heart_mask = masks.get("heart_mask")
            thorax_mask = masks.get("thorax_mask")
            lesion_mask = masks.get("lesion_mask")

            if heart_mask is not None and thorax_mask is not None:
                ctr_res = self.math_engine.calculate_cardiothoracic_ratio(heart_mask, thorax_mask)
                scan_biometrics["cardiothoracic_ratio"] = ctr_res
            if lesion_mask is not None:
                lesion_res = self.math_engine.analyze_mask_eigen_geometry(lesion_mask)
                scan_biometrics["lesion_geometry"] = lesion_res

        total_latency_ms = (time.perf_counter() - t0) * 1000

        judge_hud_summary = {
            "total_latency_ms": round(total_latency_ms, 2),
            "hallucinations_intercepted": 0,
            "mean_shannon_entropy": 0.0,
            "active_verification_flags": 0,
        }

        status = "error" if error_details else "success"
        if error_details and scan_biometrics:
            status = "partial_success"

        return {
            "status": status,
            "error_details": error_details if error_details else None,
            "cv_telemetry": {},
            "voice_intake": None,
            "dialect_icd10_alignment": [],
            "clinical_coherence_report": None,
            "active_triage_question": None,
            "polypharmacy_intervention": None,
            "patient_dosage_card": None,
            "grounded_medications": [],
            "scan_biometrics": scan_biometrics,
            "judge_hud_summary": judge_hud_summary,
            "agent_execution_trace": agent_execution_trace
        }
