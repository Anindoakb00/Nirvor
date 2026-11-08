# Nirvor — A Neuro-Symbolic Engine (v0.0.3)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.3` is the latest development branch: everything on `main`, plus
> combined voice+prescription intake, a patient dosage-card generator,
> Supabase migrations, and verification gating tests.

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

Interactive API docs: `/docs` (served alongside the dashboard).

## New in this version

- `core/dosage_card.py` — renders the `patient_dosage_card` (data + PNG)
  from verified intake; fixes the Section 4 `partial_success` case where the
  card came back `null`.
- `database/supabase_migrations/` — `001_symptoms.sql` and
  `002_prescription_referrals.sql` for pgvector-backed grounding.
- `database/import_prescription_referrals.py` and
  `database/import_verified_lexicon.py` — data importers.
- 11 backend tests, including `test_dosage_card.py`,
  `test_combined_intake.py`, `test_verification_gating.py`,
  `test_vector_ingest.py`, `test_clinical_prompt.py`,
  `test_prescription_referrals.py`, and `test_cv_pipeline.py`.
- Hygiene: `.gitignore` blocks secrets and patient data (`.env`, voice
  recordings, prescription photos, result dumps); IDE files and logs are
  untracked.

Also see `ARCHITECTURE.md`, `MATH_IMPLEMENTATION.md`, and
`CURRENT_STATE_v0.0.3.md` for design background.

## Project layout (additions vs `main`)

```text
backend/
  core/dosage_card.py              # patient dosage-card generator
  database/
    supabase_migrations/
      001_symptoms.sql
      002_prescription_referrals.sql
    import_prescription_referrals.py
    import_verified_lexicon.py
  test_dosage_card.py test_combined_intake.py test_verification_gating.py
  test_vector_ingest.py test_clinical_prompt.py test_prescription_referrals.py
  test_cv_pipeline.py
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

Dependencies: FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy, OpenCV
(headless), Pillow, httpx/requests, python-multipart, python-dotenv, Modal,
pytest, pytest-asyncio, openai, supabase, networkx, pydub.

Optional Modal cloud GPU: run `modal setup`, then set `USE_MODAL_CLOUD=true`
in `backend/.env`.

## Tests

```powershell
cd backend
pytest
```

## Notes

- Never commit `backend/.env` (real keys). Voice recordings and prescription
  photos are patient data — keep them local; the `.gitignore` on this branch
  already excludes them.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
