from typing import List, Literal
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