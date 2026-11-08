"""Imports human-verified dialect rows from _transcript_review.csv to Supabase.

Usage (from the backend/ directory):
  python database/import_verified_lexicon.py [path/to/_transcript_review.csv]

Only rows with verified=TRUE and complete labels are imported. Run the SQL in
database/supabase_migrations/001_symptoms.sql in the Supabase dashboard first.
"""
import csv
import os
import sys
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_FIELDS = ("dialect_phrase", "clinical_term", "icd10_code")


def validate_row(row: Dict[str, str]) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates one review-sheet row. Pure: no DB, no network, fully testable."""
    cleaned = {k: (v or "").strip() for k, v in row.items()}
    if cleaned.get("verified", "").upper() != "TRUE":
        return False, "not marked verified=TRUE", {}
    for field in REQUIRED_FIELDS:
        if not cleaned.get(field):
            return False, f"missing required field: {field}", {}
    return True, "ok", {
        "dialect_phrase": cleaned["dialect_phrase"],
        "clinical_term": cleaned["clinical_term"],
        "icd10_code": cleaned["icd10_code"],
        "dialect_region": cleaned.get("dialect_region", ""),
        "source_file": cleaned.get("audio_file", ""),
        "verified_by": "manual-review CSV",
    }


def main() -> int:
    from supabase import create_client

    from config.settings import AIConfig
    from core.math_engine import ClinicalMathEngine

    if not AIConfig.SUPABASE_URL or not AIConfig.SUPABASE_KEY:
        print("Supabase is not configured in backend/.env - nothing imported.")
        return 1

    default_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Voices",
        "_transcript_review.csv",
    )
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    if not os.path.isfile(csv_path):
        print(f"Review sheet not found: {csv_path}")
        return 1

    records = []
    skipped = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            ok, reason, record = validate_row(row)
            if not ok:
                skipped += 1
                print(f"row {i} skipped ({reason}): {row.get('audio_file', '')}")
                continue
            vec = ClinicalMathEngine.text_to_embedding(
                ClinicalMathEngine.to_phonetic_skeleton(record["dialect_phrase"])
            )
            record["embedding"] = vec.tolist()
            record["verified"] = True
            records.append(record)

    if not records:
        print(f"No verified rows to import ({skipped} skipped).")
        return 1

    client = create_client(AIConfig.SUPABASE_URL, AIConfig.SUPABASE_KEY)
    client.table("symptoms").upsert(records, on_conflict="dialect_phrase").execute()
    print(f"Imported {len(records)} verified rows ({skipped} skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
