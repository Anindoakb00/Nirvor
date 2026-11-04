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
    judge_hud_summary: JudgeHudSummary
    agent_execution_trace: List[AgentExecutionTraceItem] = Field(default_factory=list)
