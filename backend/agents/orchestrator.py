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

    async def _transcribe_and_align(self, audio_bytes: bytes):
        """Transcribes audio and aligns dialect symptoms. Never raises.

        Returns (transcript, dialect_matches, error_or_None).
        """
        try:
            voice_intake = await transcribe_audio_meta(audio_bytes)
        except Exception as e:
            return "", [], str(e)
        transcript = (voice_intake.get("transcript", "") if voice_intake else "") or ""
        try:
            matches = self.math_engine.align_regional_dialect_symptoms(transcript)
        except Exception:
            matches = []
        return transcript, matches, None

    async def _clinical_insights_with_telemetry(
        self,
        transcript: str,
        matches,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """Runs clinical reasoning, then overwrites telemetry with measurements.

        Returns (clinical_dict_or_None, error_or_None). Never raises.
        """
        telemetry = {"dialect_matches": matches}
        if extra_context:
            telemetry.update(extra_context)
        try:
            result = await generate_clinical_insights(transcript, telemetry)
        except Exception as e:
            return None, str(e)
        if isinstance(result, dict) and isinstance(result.get("doctor_view"), dict):
            summary = result["doctor_view"].get("clinical_summary", "")
            if not summary.startswith("BACKEND ERROR"):
                result["doctor_view"]["verification_telemetry"] = (
                    self.math_engine.build_verification_telemetry(matches)
                )
        return result, None

    async def run_voice_branch(
        self,
        audio_bytes: bytes,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """Shared voice pipeline for single + combined intake.

        Returns (clinical_dict_or_None, error_or_None). Never raises.
        """
        transcript, matches, err = await self._transcribe_and_align(audio_bytes)
        if err is not None:
            return None, err
        return await self._clinical_insights_with_telemetry(
            transcript, matches, extra_context
        )

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
                        "hallucination_risk_status": "High"
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
        patient_voice_transcript: str = "",
        dialect_symptoms: Optional[list] = None,
        xray_flags: Optional[list] = None,
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
            dialect_context = ", ".join(
                s.get("clinical_translation", "")
                for s in (dialect_symptoms or [])
                if s.get("clinical_translation")
            )
            extracted_res = await self._safe_execute(extract_prescription_multimodal(base64_data_uri, dialect_context), error_details, "Multimodal Extraction")
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
                grounded_result = self.math_engine.ground_drug_entity(raw_text, patient_voice_transcript or "")

                med_enriched = {**med, **grounded_result}
                grounded_medications.append(med_enriched)

                if grounded_result.get("ocr_typo_corrected"):
                    ocr_typos_corrected_count += 1
                total_shannon_entropy += grounded_result.get("shannon_entropy_bits", 0.0)
                if grounded_result.get("requires_active_verification"):
                    active_verification_flags += 1

        referral_telemetry = None
        mean_shannon_entropy = 0.0
        active_triage_question = None
        polypharmacy_intervention = None
        patient_dosage_card = None
        clinical_coherence_report = None

        if grounded_medications:
            mean_shannon_entropy = total_shannon_entropy / len(grounded_medications)

            clinical_coherence_report = self.math_engine.verify_symptom_medication_coherence(
                dialect_symptoms or [], grounded_medications,
                xray_flags=xray_flags or [],
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
                pk_summary = {
                    "medications": [m.get("matched_drug", {}) for m in grounded_medications],
                    "requires_verification": [bool(m.get("requires_active_verification", False)) for m in grounded_medications],
                }
                polypharmacy_intervention = await self._safe_execute(escalate_polypharmacy_wonder(pk_summary, vitals), error_details, "Polypharmacy Toxicity")
                if (polypharmacy_intervention
                        and polypharmacy_intervention.get("verified_medications_assessed", 1) == 0
                        and polypharmacy_intervention.get("unverified_medications_skipped", 0) > 0):
                    active_verification_flags += 1
                if polypharmacy_intervention:
                    agent_execution_trace.append({
                        "step_name": "polypharmacy_toxicity_spark",
                        "status": "success",
                        "provider_used": polypharmacy_intervention.get("provider_used", "local_fallback"),
                        "latency_ms": round((time.perf_counter() - poly_t0) * 1000, 2),
                        "spark_triggered_reason": "multiple_drugs_or_high_toxicity"
                    })

            rx_t0 = time.perf_counter()
            rx_blob = self.math_engine.build_medication_blob(grounded_medications)
            rx_matches = self.math_engine.match_prescription_referrals(rx_blob)
            if rx_matches:
                referral_telemetry = self.math_engine.build_referral_telemetry(rx_matches)
                agent_execution_trace.append({
                    "step_name": "prescription_referral_match",
                    "status": "success",
                    "provider_used": "supabase_pgvector",
                    "latency_ms": round((time.perf_counter() - rx_t0) * 1000, 2),
                    "spark_triggered_reason": "always_run_on_meds"
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
            "referral_telemetry": referral_telemetry,
            "judge_hud_summary": judge_hud_summary,
            "agent_execution_trace": agent_execution_trace
        }

    async def _rx_branch(
        self,
        prescription_bytes: bytes,
        transcript: str,
        dialect_matches,
        xray_flags,
    ):
        """Prescription branch with cross-modal context. Never raises."""
        try:
            res = await self.process_prescription(
                prescription_bytes=prescription_bytes,
                patient_voice_transcript=transcript,
                dialect_symptoms=dialect_matches,
                xray_flags=xray_flags,
            )
            return res, None
        except Exception as e:
            return None, f"Prescription failed: {e}"

    async def _xray_branch(self, xray_bytes: bytes):
        """X-ray branch. Never raises."""
        try:
            res = await self.process_xray(xray_bytes=xray_bytes)
            return res, None
        except Exception as e:
            return None, f"X-Ray failed: {e}"

    @staticmethod
    def _xray_flags_and_summary(xray_res: Optional[Dict[str, Any]]):
        """Derives discrete flags + a one-line summary from X-ray biometrics."""
        flags = []
        bio = (xray_res or {}).get("scan_biometrics", {}) or {}
        ctr = bio.get("cardiothoracic_ratio", {}) or {}
        if ctr.get("cardiomegaly_flagged"):
            flags.append("cardiomegaly")
        lesion = bio.get("lesion_geometry", {}) or {}
        if lesion.get("valid_mask"):
            flags.append("lesion_detected")
        parts = []
        if ctr:
            parts.append(
                f"CTR={ctr.get('ctr')} "
                f"(cardiomegaly={ctr.get('cardiomegaly_flagged')})"
            )
        if lesion.get("valid_mask"):
            parts.append(
                f"lesion area={lesion.get('estimated_area_mm2')}mm2 "
                f"ecc={lesion.get('eccentricity')}"
            )
        return flags, "; ".join(parts)

    def _build_cross_modal_fusion(
        self,
        voice_res: Optional[Dict[str, Any]],
        rx_res: Optional[Dict[str, Any]],
        xray_res: Optional[Dict[str, Any]],
        requested: list,
    ) -> Dict[str, Any]:
        """Fuses branch outputs into one measured confidence block.

        fused_confidence = mean of available modality confidences,
        +0.05 when 2+ modalities succeed without contradiction,
        -0.10 per contradiction flag (floor 0). A lone voice branch reports
        its own telemetry risk verbatim since there is nothing to fuse.
        """
        succeeded = []
        if voice_res is not None:
            succeeded.append("voice")
        if rx_res is not None:
            succeeded.append("prescription")
        if xray_res is not None:
            succeeded.append("xray")

        tele = ((voice_res or {}).get("doctor_view", {}) or {}).get(
            "verification_telemetry", {}) or {}
        voice_conf = None
        if isinstance(tele.get("primary_vector_match_score"), (int, float)):
            voice_conf = float(tele["primary_vector_match_score"])

        meds = (rx_res or {}).get("grounded_medications", []) or []
        posteriors = [m.get("bayesian_posterior_confidence") for m in meds
                      if isinstance(m.get("bayesian_posterior_confidence"),
                                    (int, float))]
        rx_conf = None
        if posteriors:
            rx_conf = float(sum(posteriors) / len(posteriors))

        xray_present = bool((xray_res or {}).get("scan_biometrics"))

        coherence = (rx_res or {}).get("clinical_coherence_report", {}) or {}
        flags = coherence.get("contradiction_flags", []) or []
        indications = coherence.get("matched_indications", []) or []
        is_coherent = bool(coherence.get("is_coherent", True)) and not flags

        available = [c for c in (voice_conf, rx_conf) if c is not None]
        fused = sum(available) / len(available) if available else 0.0
        if len(succeeded) >= 2 and is_coherent:
            fused += 0.05
        fused -= 0.10 * min(len(flags), 3)
        fused = round(max(0.0, min(1.0, fused)), 4)

        if fused >= 0.6:
            risk = "Low"
        elif fused >= 0.25:
            risk = "Medium"
        else:
            risk = "High"
        if succeeded == ["voice"] and tele.get("hallucination_risk_status"):
            risk = tele["hallucination_risk_status"]

        notes = []
        if len(succeeded) <= 1:
            notes.append("Single modality succeeded: fused score mirrors it.")
        for ind in indications:
            notes.append(f"Agreement: {ind}")
        for flag in flags:
            notes.append(f"Contradiction: {flag}")
        if xray_present:
            _, summary = self._xray_flags_and_summary(xray_res)
            if summary:
                notes.append(f"X-ray biometrics: {summary}")
        if not succeeded:
            notes.append("No modality succeeded.")

        return {
            "modalities_present": requested,
            "modalities_succeeded": succeeded,
            "voice_confidence": voice_conf,
            "prescription_confidence": rx_conf,
            "xray_biometrics_present": xray_present,
            "fused_confidence": fused,
            "overall_risk": risk,
            "is_coherent": is_coherent,
            "agreement_notes": notes,
        }

    async def process_combined(
        self,
        prescription_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        xray_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Runs every provided modality and fuses the results cross-modally.

        Phase 1 runs the independent first passes (voice transcribe+align,
        X-ray segmentation) concurrently. Phase 2 runs drug grounding and
        clinical reasoning with the full cross-modal context. Phase 3 fuses
        the branch outputs into one measured confidence block.
        """
        error_details = []

        # Phase 1: independent first passes.
        transcript, dialect_matches, align_err = "", [], None
        xray_res = None
        phase1 = {}
        if audio_bytes:
            phase1["align"] = self._transcribe_and_align(audio_bytes)
        if xray_bytes:
            phase1["xray"] = self._xray_branch(xray_bytes)
        if phase1:
            outs = await asyncio.gather(*phase1.values())
            for name, out in zip(phase1.keys(), outs):
                if name == "align":
                    transcript, dialect_matches, align_err = out
                    if align_err:
                        error_details.append(align_err)
                else:
                    xray_res, xray_err = out
                    if xray_err:
                        error_details.append(xray_err)

        xray_flags, xray_summary = self._xray_flags_and_summary(xray_res)

        # Phase 2: context-aware branches.
        voice_res, rx_res = None, None
        phase2 = {}
        if audio_bytes and align_err is None:
            extra = {"xray_biometrics": xray_summary} if xray_summary else None
            phase2["voice"] = self._clinical_insights_with_telemetry(
                transcript, dialect_matches, extra
            )
        if prescription_bytes:
            phase2["rx"] = self._rx_branch(
                prescription_bytes, transcript, dialect_matches, xray_flags
            )
        if phase2:
            outs = await asyncio.gather(*phase2.values())
            for name, (res, err) in zip(phase2.keys(), outs):
                if name == "voice":
                    voice_res = res
                else:
                    rx_res = res
                if err:
                    error_details.append(err)

        for label, res in (("prescription", rx_res), ("xray", xray_res)):
            for nested in (res or {}).get("error_details") or []:
                error_details.append(f"{label}: {nested}")

        requested = []
        if audio_bytes:
            requested.append("voice")
        if prescription_bytes:
            requested.append("prescription")
        if xray_bytes:
            requested.append("xray")

        succeeded = [r for r in (voice_res, rx_res, xray_res) if r is not None]
        if not requested:
            status = "error"
            error_details.append("No input provided")
        elif not succeeded:
            status = "error"
        elif error_details or len(succeeded) < len(requested):
            status = "partial_success"
        else:
            status = "success"

        fusion = self._build_cross_modal_fusion(
            voice_res, rx_res, xray_res, requested
        )

        return {
            "status": status,
            "error_details": error_details if error_details else None,
            "voice": voice_res,
            "prescription": rx_res,
            "xray": xray_res,
            "cross_modal_fusion": fusion,
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
