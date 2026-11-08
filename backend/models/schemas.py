from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class MedicationItem(BaseModel):
  raw_text: str = Field(
      ..., description="Raw OCR medication line, e.g., 'Tab. Sergel 20mg'"
  )
  dosage: str = Field(..., description="Strength per unit, e.g., '20mg'")
  frequency: str = Field(
      ..., description="Bangladeshi dosing format, e.g., '1+0+1'"
  )
  duration_days: int = Field(..., description="Total treatment duration in days")


class PatientVitals(BaseModel):
  bp: str = Field(default="120/80", description="Blood pressure string")
  pulse: str = Field(default="72", description="Heart rate in bpm")
  temp: str = Field(default="98.6", description="Body temperature in F")
  creatinine_mg_dl: float = Field(
      default=1.0, description="Serum creatinine for renal clearance"
  )


class PrescriptionExtraction(BaseModel):
  medications: List[MedicationItem]
  patient_vitals: PatientVitals
  raw_symptoms: List[str]


class PolypharmacyIntervention(BaseModel):
  risk_level: Literal["CRITICAL", "HIGH", "MODERATE", "LOW"]
  clinical_mechanism: str
  recommended_action: str
  bangla_patient_warning: str
  verified_medications_assessed: int = 0
  unverified_medications_skipped: int = 0


class ClinicalCoherenceReport(BaseModel):
    is_coherent: bool
    coherence_score: float
    matched_indications: List[str]
    contradiction_flags: List[str]


class AgentExecutionTraceItem(BaseModel):
    step_name: str
    status: str
    provider_used: str
    latency_ms: float
    spark_triggered_reason: str


class JudgeHudSummary(BaseModel):
    total_latency_ms: float
    hallucinations_intercepted: int
    mean_shannon_entropy: float
    active_verification_flags: int


class ReferralTelemetry(BaseModel):
    top_referral_match_score: float
    referral_source_file: str
    referral_verified: bool
    referral_medications: str
    referral_confidence: Literal["Low", "Medium", "High"]


class CompleteIntakeResponse(BaseModel):
    status: str
    error_details: Optional[List[str]] = None
    cv_telemetry: dict = Field(default_factory=dict)
    voice_intake: dict | None = None
    dialect_icd10_alignment: list = Field(default_factory=list)
    clinical_coherence_report: ClinicalCoherenceReport | None = None
    active_triage_question: str | None = None
    polypharmacy_intervention: PolypharmacyIntervention | None = None
    patient_dosage_card: str | None = None
    grounded_medications: list = Field(default_factory=list)
    scan_biometrics: dict = Field(default_factory=dict)
    referral_telemetry: ReferralTelemetry | None = None
    judge_hud_summary: JudgeHudSummary
    agent_execution_trace: List[AgentExecutionTraceItem] = Field(default_factory=list)

class VerificationTelemetry(BaseModel):
    primary_vector_match_score: float
    grounding_source_text: str
    hallucination_risk_status: Literal["Low", "Medium", "High"]

class DoctorView(BaseModel):
    clinical_summary: str
    dialect_icd10_alignment: List[str]
    active_triage_question: str
    grounded_medications: List[str]
    verification_telemetry: VerificationTelemetry

class PatientView(BaseModel):
    layman_explanation: str
    next_steps: str

class ClinicalAnalysisResponse(BaseModel):
    doctor_view: DoctorView
    patient_view: PatientView


class CrossModalFusion(BaseModel):
    modalities_present: List[str]
    modalities_succeeded: List[str]
    voice_confidence: Optional[float] = None
    prescription_confidence: Optional[float] = None
    xray_biometrics_present: bool = False
    fused_confidence: float
    overall_risk: Literal["Low", "Medium", "High"]
    is_coherent: bool
    agreement_notes: List[str] = Field(default_factory=list)


class CombinedIntakeResponse(BaseModel):
    status: str
    error_details: Optional[List[str]] = None
    voice: Optional[ClinicalAnalysisResponse] = None
    prescription: Optional[CompleteIntakeResponse] = None
    xray: Optional[CompleteIntakeResponse] = None
    cross_modal_fusion: Optional[CrossModalFusion] = None
