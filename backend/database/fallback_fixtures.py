
from typing import Any, Dict
import cv2
import numpy as np
from core.schemas import (
    MedicationItem,
    PatientVitals,
    PolypharmacyIntervention,
    PrescriptionExtraction,
)

DEFAULT_PRESCRIPTION_FIXTURE = PrescriptionExtraction(
    medications=[
        MedicationItem(
            raw_text="Tab. Sarjel 20mg",
            dosage="20mg",
            frequency="1+0+1",
            duration_days=14,
        ),
        MedicationItem(
            raw_text="Tab. Rolac 10mg",
            dosage="10mg",
            frequency="1+0+1",
            duration_days=5,
        ),
        MedicationItem(
            raw_text="Tab. Flexi 100mg",
            dosage="100mg",
            frequency="1+0+1",
            duration_days=7,
        ),
    ],
    patient_vitals=PatientVitals(
        bp="120/80", pulse="72", temp="98.6", creatinine_mg_dl=1.0
    ),
    raw_symptoms=["chest pain", "dizziness"],
)

DEFAULT_POLYPHARMACY_FIXTURE = PolypharmacyIntervention(
    risk_level="CRITICAL",
    clinical_mechanism=(
        "Concurrent Ketorolac (Rolac) and Aceclofenac (Flexi) administration "
        "causes cumulative COX inhibition and acute renal afferent arteriolar vasoconstriction."
    ),
    recommended_action=(
        "Immediately discontinue one NSAID and monitor serum creatinine."
    ),
    bangla_patient_warning=(
        "সতর্কতা: রোলাক (Rolac) এবং ফ্লেক্সি (Flexi) একসাথে সেবন করলে কিডনির "
        "মারাত্মক ক্ষতি হতে পারে। অবিলম্বে ডাক্তারের পরামর্শ নিন।"
    ),
)

DEFAULT_TRIAGE_QUESTION_BN = (
    "আপনার ওষুধের নামটি নিশ্চিত করতে দয়া করে বলবেন কি এটি গ্যাস্ট্রিকের নাকি ব্যথার ওষুধ? "
    "(Could you confirm if this medication is for gastric relief or pain?)"
)

DEFAULT_CHITTAGONIAN_TRANSCRIPT = "আঁর বুকত ব্যথা আর মাথা চক্কর দেয়"
DEFAULT_DOSAGE_CARD_URL = (
    "https://placehold.co/600x400/0a0a0a/00ffcc?text=Sonakto+Visual+Dosage+Card"
)


def build_opencv_fallback_masks(image_bytes: bytes) -> Dict[str, np.ndarray]:
  """Generates deterministic 2D binary morphological masks from raw image bytes."""
  nparr = np.frombuffer(image_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
  if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)

  _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  return {
      "heart_mask": np.where(thresh > 127, 1, 0).astype(np.uint8),
      "thorax_mask": np.ones_like(thresh, dtype=np.uint8),
      "lesion_mask": np.zeros_like(thresh, dtype=np.uint8),
  }