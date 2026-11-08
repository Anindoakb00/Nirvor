# Nirvor — A Neuro-Symbolic Engine (v0.0.2)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.2` is the **agentic refactor** of v0.0.1: identical math, reorganized
> into an orchestrator agent driving audio, vision, and reasoning modules
> behind one intake router. See `ARCHITECTURE.md` for the design.

## Added in this version (vs v0.0.1)

- `agents/orchestrator.py` — `IntakeOrchestrator` (`__init__` +
  `process_intake`, ~101 lines). 9-step flow: timer → sync CV → parallel
  voice + SAM segmentation → dialect→ICD-10 alignment → multimodal LLM
  extraction → per-medication Bayesian grounding + entropy aggregation →
  CTR/lesion biometrics → consolidated JSON.
- `api/routers/intake.py` — `POST /api/v2/intake-pipeline` out of `main.py`.
- `config/settings.py` — central `AIConfig`; `models/schemas.py` — models
  package.
- Pure renames (0 content lines): `audio_processor.py` →
  `audio_pipeline.py`, `cv_preprocessing.py` → `cv_pipeline.py`,
  `core/schemas.py` → `models/schemas.py`, `meta_client.py` →
  `meta_agent.py` (21-line touch-up; same 6 async functions + fixtures).

## Endpoints

| Method | Path                      | Request (multipart)                                        |
|--------|---------------------------|------------------------------------------------------------|
| GET    | `/`                       | Health JSON                                                |
| POST   | `/api/v2/intake-pipeline` | `prescription` (required), `audio` + `xray_scan` (optional) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/intake-pipeline \
  -F "prescription=@rx.jpg" -F "audio=@note.mp3" -F "xray_scan=@chest.png"
```

Flow:

```text
client → intake router → IntakeOrchestrator → ┬ CV pipeline (sync)
                                              ├ voice ─┐ (parallel)
                                              └ SAM ───┘
       → math grounding → biometrics → one JSON response
```

## Math engine — unchanged from v0.0.1

Same 10 methods: 512-dim hashed phonetic embeddings, cosine similarity,
cursive-weighted Levenshtein DP, Bayesian posterior + Shannon entropy,
dialect→ICD-10 alignment, CTR, lesion PCA. Full audit in
`MATH_IMPLEMENTATION.md` (on `main`/`v-0.0.3`).

## What we gained in this version

- **Separation of concerns**: routers handle HTTP, the orchestrator owns the
  flow, pipelines own their modality — new agents (e.g. MRI) can plug in
  without touching the API layer.
- **One place for the pipeline**: timing, aggregation, and the response
  envelope live in `process_intake`, not scattered in `main.py`.
- **Config discipline**: all keys/URLs behind `AIConfig` + `.env.example`.
- Proof the architecture scales: same behavior, half the coupling.

## Accuracy & trust — honest notes

- Same signals as v0.0.1 (posterior + entropy, triage on high entropy),
  now aggregated centrally. No new scoring, no benchmark yet.
- `modal_worker.py` still isolated (never called by the orchestrator).

## Project layout (this branch)

```text
backend/
  main.py                  # thin wiring: CORS + router + GET /
  requirements.txt         # same base deps as v0.0.1
  .env.example             # copy to .env (never commit .env)
  config/settings.py  models/schemas.py
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

## What's next (delivered on `main`)

- Split the monolith into per-modality endpoints (voice / prescription /
  xray) with per-branch error capture.
- Add clinical reasoning (dual doctor/patient view), a polypharmacy safety
  graph, PII scrubbing, and a test dashboard.

## Notes

- No frontend — use `/docs`, cURL, or `test_api.py`.
- Never commit `backend/.env`. Voice/prescription files are patient data.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
