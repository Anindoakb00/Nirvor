# Nirvor — A Neuro-Symbolic Engine (v0.0.3)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.3` is the latest development branch: everything on `main`, plus
> combined voice+prescription intake, a local patient dosage-card renderer,
> prescription-referral matching with telemetry, Supabase migrations, and
> verification gating tests.

## Added in this version (vs `main`)

- **Combined intake** — `POST /api/v2/intake/combined` → new
  `process_combined()` + `_build_cross_modal_fusion()`: voice branch and
  prescription branch run through shared helpers (`_transcribe_and_align`,
  `_clinical_insights_with_telemetry`, `_rx_branch`, `_xray_branch`,
  `_xray_flags_and_summary`) and fuse into one `CombinedIntakeResponse`.
  Orchestrator grows to 13 methods.
- **Dosage card, drawn locally** (`core/dosage_card.py`): Meta's image model
  refuses medication prompts by content policy, so the card is rendered
  offline with Pillow — `parse_dosage_summary()` → `CardRow` list →
  `render_dosage_card_png()` (900px card, auto-fit fonts, zebra rows;
  columns Medicine / Morning / Noon / Night / Schedule with `1+0+1` dose
  patterns). Exact drug names, no network, no cost. English headers: PIL
  can't shape Bangla script. This fixes the Section 4 bug where
  `patient_dosage_card` came back `null` with a 400 in `error_details`.
- **Math +4**: `build_verification_telemetry()`, `build_medication_blob()`,
  `match_prescription_referrals()`, `build_referral_telemetry()` —
  referral matching and structured verification telemetry builders.
- **Supabase grounding**: `database/supabase_migrations/001_symptoms.sql` +
  `002_prescription_referrals.sql`, importers
  (`import_prescription_referrals.py`, `import_verified_lexicon.py`), and
  `SUPABASE_URL` / `SUPABASE_KEY` in `.env.example`.
- **Gating tests**: `test_verification_gating.py` enforces coherence
  (`is_coherent`, `coherence_score`, `contradiction_flags` like "require
  verification" / "1 of 2") and safety gating where
  `unverified_medications_skipped` are excluded from assessment and counted
  separately from `verified_medications_assessed`.
- **Nov 5–7 fixes**: clinical agent ensures all data reaches both
  doctor and patient views; meta-agent intake fixes; `pillow` in the
  backend venv; 11 backend tests total.
- **Hygiene**: `.gitignore` blocks secrets + patient data; IDE files and
  logs untracked.

## Endpoints

| Method | Path                              | Purpose                                |
|--------|-----------------------------------|----------------------------------------|
| GET    | `/health`                         | Liveness check                         |
| GET    | `/api/v2/system-status`           | Engine / configuration status          |
| POST   | `/api/v2/intake/voice`            | Voice note → clinical analysis         |
| POST   | `/api/v2/intake/prescription`     | Prescription image → full intake       |
| POST   | `/api/v2/intake/xray`             | X-ray scan → full intake               |
| POST   | `/api/v2/intake/combined`         | Voice + prescription → combined intake |
| GET    | `/`                               | Static dashboard (test UI)             |

Interactive API docs at `/docs` (served alongside the dashboard).

## Math engine — 15 methods

v0.0.1's 10 (embeddings, cosine, cursive Levenshtein, Bayesian posterior +
entropy, dialect alignment, CTR, lesion PCA) + `main`'s coherence check +
the 4 telemetry/referral builders above. Audit: `MATH_IMPLEMENTATION.md`.

## Accuracy & trust — honest notes

- Strongest branch for trust: coherence scoring is **test-enforced**
  (gating tests assert exact flags/scores), unverified meds are explicitly
  skipped from safety assessment rather than silently passed, and every
  branch reports `error_details` on partial failure.
- Still **no measured accuracy benchmark** (no labeled eval set). The
  dosage-card fix was verified by rendering (`dosage_card_sample.png`-style
  output) and the Section 4 re-test comparison, not by a metric.
- Same standing caveats as `main`: fixture-backed offline paths, 2-rule
  polypharmacy graph, isolated Modal worker.

## Project layout (additions vs `main`)

```text
backend/
  core/dosage_card.py              # local PIL dosage-card renderer
  database/
    supabase_migrations/
      001_symptoms.sql  002_prescription_referrals.sql
    import_prescription_referrals.py  import_verified_lexicon.py
  test_dosage_card.py test_combined_intake.py test_verification_gating.py
  test_vector_ingest.py test_clinical_prompt.py test_prescription_referrals.py
  test_cv_pipeline.py   (+ test_api/baseten/cloud_cv/modal)
```

## Setup

Prereqs: Python 3.10+.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env: set BASETEN_API_KEY and META_API_KEY
# optional: SUPABASE_URL + SUPABASE_KEY for pgvector grounding
python -m uvicorn main:app --reload --port 8000
```

Deps: FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy, OpenCV (headless),
Pillow, httpx/requests, python-multipart, python-dotenv, Modal, pytest,
pytest-asyncio, openai, supabase, networkx, pydub.

Optional Modal cloud GPU: run `modal setup`, then `USE_MODAL_CLOUD=true`.

## Tests

```powershell
cd backend
pytest
```

## Notes

- Never commit `backend/.env`. Voice/prescription files are patient data;
  this branch's `.gitignore` already excludes them.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
