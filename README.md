# Nirvor — A Neuro-Symbolic Engine (v0.0.2)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> This branch (`v-0.0.2`) refactors the v0.0.1 prototype into a modular
> agentic architecture: an orchestrator agent coordinating audio, vision, and
> reasoning pipelines behind one intake router.

## What this version does

Same API surface as v0.0.1, now routed through modules:

| Method | Path                      | Purpose                                                        |
|--------|---------------------------|----------------------------------------------------------------|
| GET    | `/`                       | Health check                                                   |
| POST   | `/api/v2/intake-pipeline` | Prescription image (required) + voice audio / X-ray (optional) |

`api/routers/intake.py` serves the endpoint; `agents/orchestrator.py` drives
the audio pipeline, CV pipeline, and Meta reasoning agent underneath.

## Project layout (this branch)

```text
backend/
  main.py                  # app wiring (CORS + intake router)
  requirements.txt
  .env.example             # copy to .env and fill in keys (never commit .env)
  config/settings.py       # central env-driven settings
  models/schemas.py        # request/response models
  agents/orchestrator.py   # intake orchestrator agent
  api/routers/intake.py    # POST /api/v2/intake-pipeline
  core/
    audio_pipeline.py      # voice-note pipeline
    cv_pipeline.py         # prescription image pipeline
    math_engine.py         # Bayesian clinical math
    meta_agent.py          # Meta reasoning agent
    modal_worker.py        # Modal serverless GPU worker
    prompts.py             # model prompts
  database/
    dims_drug_db.json      # drug reference data
    fallback_fixtures.py   # offline fallback fixtures
  test_baseten.py test_cloud_cv.py test_modal.py
ARCHITECTURE.md            # design notes for this refactor
test_api.py                # repo-level API smoke test
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

- See `ARCHITECTURE.md` for the agentic design behind this version.
- No frontend on this branch — use `/docs` to exercise the API.
- Never commit `backend/.env` (real keys). Voice recordings and prescription
  photos are patient data — keep them local.
- `main` builds on this with per-modality endpoints and a dashboard.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
