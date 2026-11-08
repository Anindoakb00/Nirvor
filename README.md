# Nirvor — A Neuro-Symbolic Engine (v0.0.1)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> This branch (`v-0.0.1`) is the first working prototype: one FastAPI app that
> fuses prescription OCR, regional voice notes, and X-ray scans into a single
> clinical intake call.

## What this version does

One all-in-one endpoint:

| Method | Path                      | Purpose                                                        |
|--------|---------------------------|----------------------------------------------------------------|
| GET    | `/`                       | Health check                                                   |
| POST   | `/api/v2/intake-pipeline` | Prescription image (required) + voice audio / X-ray (optional) |

Pipeline inside the endpoint:

1. `ClinicalCVPreprocessor` cleans the prescription photo (OpenCV) and builds
   a data URI plus telemetry.
2. A Baseten-hosted multimodal model (`meta-models/muse-spark-open-1026`)
   extracts the prescription text.
3. `process_regional_voice_note` transcribes the voice note (Meta voice model).
4. SAM 3.1 segments the X-ray / medical scan.
5. `ClinicalMathEngine` applies Bayesian grounding math over the fused signals.

## Project layout (this branch)

```text
backend/
  main.py                # FastAPI app + the /api/v2/intake-pipeline endpoint
  requirements.txt
  .env.example           # copy to .env and fill in keys (never commit .env)
  core/
    audio_processor.py   # regional voice-note transcription
    cv_preprocessing.py  # prescription image cleanup + telemetry
    math_engine.py       # Bayesian clinical math
    meta_client.py       # Baseten multimodal OCR + SAM 3.1 client
    modal_worker.py      # Modal serverless GPU worker
    prompts.py           # model prompts
    schemas.py           # request/response models
  database/
    dims_drug_db.json    # drug reference data
    fallback_fixtures.py # offline fallback fixtures
  test_baseten.py test_cloud_cv.py test_modal.py
test_api.py              # repo-level API smoke test
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
python -m uvicorn main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/docs for the interactive API docs.

Dependencies: FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy, OpenCV
(headless), Pillow, httpx/requests, python-multipart, python-dotenv, Modal.

## Notes

- Single-file style app: wiring and the endpoint all live in `main.py`.
- No frontend on this branch — use `/docs` to exercise the API.
- Never commit `backend/.env` (real keys). Voice recordings and prescription
  photos are patient data — keep them local.
- Superseded by `v-0.0.2` (agentic refactor); the stable line is `main`.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
