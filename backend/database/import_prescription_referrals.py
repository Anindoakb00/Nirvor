"""Bulk-imports prescription images as referral vectors into Supabase.

Usage (from the backend/ directory):
  python database/import_prescription_referrals.py [--images-dir DIR] [--limit N] [--resume]

Each image goes through the same CV deskew + Meta multimodal extraction as the
live prescription branch. Only real provider extractions are stored: fixture
fallbacks and failures are skipped, never imported. Rows arrive with
verified=false; a human reviews them before they count as grounding truth.

Run database/supabase_migrations/002_prescription_referrals.sql in the
Supabase dashboard first. A full 200-image run takes a while (one Meta API
call per image) - use --limit for samples and --resume to continue.
"""
import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
UPSERT_BATCH_SIZE = 25
REAL_PROVIDERS = ("meta_api", "baseten")


def build_referral_record(
    source_file: str,
    blob: str,
    med_count: int,
    embedding: List[float],
    provider_used: str,
) -> Dict[str, Any]:
    """Builds one prescription_referrals row. Pure: no DB, no network."""
    return {
        "source_file": source_file,
        "medication_blob": blob,
        "med_count": med_count,
        "embedding": embedding,
        "verified": False,
        "provider_used": provider_used,
    }


async def extract_one(image_path: str) -> Dict[str, Any]:
    """Runs CV deskew + multimodal extraction for one image.

    Raises on any failure so the caller can skip the file. Returns the
    privacy-scrubbed extraction dict on success.
    """
    from core.cv_pipeline import ClinicalCVPreprocessor
    from core.meta_agent import extract_prescription_multimodal
    from core.privacy import PrivacyShield

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    cv_result = ClinicalCVPreprocessor.process_prescription_image(image_bytes)
    data_uri = cv_result["base64_data_uri"]
    if not data_uri:
        raise RuntimeError("CV pipeline produced no image")
    extracted = await extract_prescription_multimodal(data_uri)
    if extracted.get("provider_used") not in REAL_PROVIDERS:
        raise RuntimeError(
            f"refusing non-provider extraction ({extracted.get('provider_used')})"
        )
    return PrivacyShield.scrub_data(extracted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import prescription referrals.")
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from supabase import create_client

    from config.settings import AIConfig
    from core.math_engine import ClinicalMathEngine

    if not AIConfig.SUPABASE_URL or not AIConfig.SUPABASE_KEY:
        print("Supabase is not configured in backend/.env - nothing imported.")
        return 1

    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "Prescribtions",
        "images",
    )
    images_dir = os.path.abspath(args.images_dir or default_dir)
    if not os.path.isdir(images_dir):
        print(f"Images directory not found: {images_dir}")
        return 1

    files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("No prescription images found - nothing imported.")
        return 1

    client = create_client(AIConfig.SUPABASE_URL, AIConfig.SUPABASE_KEY)

    already: set = set()
    if args.resume:
        try:
            res = client.table("prescription_referrals").select("source_file").execute()
            already = {r.get("source_file", "") for r in (res.data or [])}
            print(f"Resume: {len(already)} files already imported, skipping them.")
        except Exception as e:
            print(f"Resume lookup failed ({e}); importing everything.")

    engine = ClinicalMathEngine()
    imported = 0
    skipped = 0
    batch: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal imported
        if not batch:
            return
        client.table("prescription_referrals").upsert(
            batch, on_conflict="source_file"
        ).execute()
        imported += len(batch)
        batch.clear()

    total = len(files)
    for i, name in enumerate(files, start=1):
        if name in already:
            continue
        print(f"[{i}/{total}] {name} ...", flush=True)
        try:
            scrubbed = asyncio.run(extract_one(os.path.join(images_dir, name)))
            meds = scrubbed.get("medications", []) or []
            blob = ClinicalMathEngine.build_medication_blob(meds)
            if not blob:
                print("  skipped (no medications extracted)")
                skipped += 1
                continue
            vec = ClinicalMathEngine.text_to_embedding(blob)
            batch.append(build_referral_record(
                name, blob, len(meds), vec.tolist(),
                scrubbed.get("provider_used", ""),
            ))
            print(f"  ok ({len(meds)} meds)")
            if len(batch) >= UPSERT_BATCH_SIZE:
                flush()
        except Exception as e:
            print(f"  skipped ({str(e)[:160]})")
            skipped += 1

    flush()
    print(f"Done: {imported} imported, {skipped} skipped.")
    return 0 if imported else 1


if __name__ == "__main__":
    sys.exit(main())
