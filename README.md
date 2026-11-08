# Nirvor — A Neuro-Symbolic Engine (v0.0.2)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.2` is the **agentic refactor** of v0.0.1: identical math, reorganized
> into an orchestrator agent driving audio, vision, and reasoning modules
> behind one intake router. See `ARCHITECTURE.md` for the design.

## Added in this version (vs v0.0.1)

- `agents/orchestrator.py` — `IntakeOrchestrator` (`__init__` +
  `process_intake`, ~101 lines). The 9-step flow: timer → sync CV
  preprocessing → parallel voice transcription + SAM 3.1 segmentation →
  dialect→ICD-10 alignment → multimodal LLM extraction → per-medication
  Bayesian grounding with entropy aggregation → CTR/lesion biometrics →
  one consolidated JSON payload.
- `api/routers/intake.py` — `POST /api/v2/intake-pipeline` moved out of
  `main.py` (same contract: prescription required, audio/xray optional).
- `config/settings.py` — central env-driven settings (`AIConfig`).
- `models/schemas.py` — Pydantic models in their own package.
- Pure renames (0 content lines changed): `audio_processor.py` →
  `audio_pipeline.py`, `cv_preprocessing.py` → `cv_pipeline.py`,
  `core/schemas.py` → `models/schemas.py`, `meta_client.py` →
  `meta_agent.py` (21-line touch-up; same 6 async functions + fixtures).

## Endpoints

| Method | Path                      | Purpose                                                        |
|--------|---------------------------|----------------------------------------------------------------|
| GET    | `/`                       | Health check                                                   |
| POST   | `/api/v2/intake-pipeline` | Prescription image (required) + voice audio / X-ray (optional) |

## Math engine — unchanged from v0.0.1

`core/math_engine.py` is byte-identical in behavior (same 10 methods):
512-dim hashed phonetic embeddings, cosine similarity, cursive-weighted
Levenshtein DP, Bayesian posterior grounding with Shannon entropy,
dialect→ICD-10 alignment, CTR, and lesion PCA. See the v0.0.1 README or
`MATH_IMPLEMENTATION.md` (on `main`/`v-0.0.3`) for the full audit.

## Model clients — `core/meta_agent.py` (~178 lines)

Same 6 async functions as v0.0.1's `meta_client.py`: multimodal extraction,
entropy triage questions, polypharmacy escalation, SAM 3.1 segmentation,
audio transcription, dosage-card URL — all fixture-backed offline.

## Accuracy & trust — honest notes

- Same signals as v0.0.1: per-drug posterior + entropy bits, triage on high
  entropy. The refactor centralizes the flow (one orchestrator computes the
  aggregates) but adds no new scoring.
- Still **no measured accuracy benchmark** — validation is the same three
  test scripts plus `test_api.py`.
- `modal_worker.py` remains isolated (never called by the orchestrator).

## Project layout (this branch)

```text
backend/
  main.py                  # thin wiring: CORS + intake router + GET /
  requirements.txt         # same base deps as v0.0.1
  .env.example             # copy to .env and fill in keys (never commit .env)
  config/settings.py       # AIConfig
  models/schemas.py
  agents/orchestrator.py   # IntakeOrchestrator.process_intake
  api/routers/intake.py    # POST /api/v2/intake-pipeline
  core/
    audio_pipeline.py cv_pipeline.py math_engine.py
    meta_agent.py modal_worker.py prompts.py
  database/
    dims_drug_db.json fallback_fixtures.py
  test_baseten.py test_cloud_cv.py test_modal.py
ARCHITECTURE.md
test_api.py
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

Docs at http://127.0.0.1:8000/docs.

## Tests

```powershell
cd backend
python test_baseten.py
python test_cloud_cv.py
python test_modal.py
```

## Notes

- No frontend — use `/docs`, cURL, or `test_api.py`.
- Never commit `backend/.env`. Voice/prescription files are patient data.
- Next: `main` splits the monolith into per-modality endpoints and adds
  clinical reasoning, safety graph, and dashboard.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
