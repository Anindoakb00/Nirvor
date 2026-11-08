# Nirvor — A Neuro-Symbolic Engine (main)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `main` is the stable line: per-modality intake endpoints, a clinical
> reasoning agent, a polypharmacy safety graph, PII scrubbing, and a static
> test dashboard.

## Endpoints

| Method | Path                              | Purpose                                |
|--------|-----------------------------------|----------------------------------------|
| GET    | `/health`                         | Liveness check                         |
| GET    | `/api/v2/system-status`           | Engine / configuration status          |
| POST   | `/api/v2/intake/voice`            | Voice note → clinical analysis         |
| POST   | `/api/v2/intake/prescription`     | Prescription image → full intake       |
| POST   | `/api/v2/intake/xray`             | X-ray scan → full intake               |
| GET    | `/`                               | Static dashboard (test UI)             |

Interactive API docs: `/docs` ( served alongside the dashboard).

## Highlights

- Clinical agent fusing OCR + voice + scan signals with a Bayesian math
  engine (grounding, entropy, CTR, PCA) — see `MATH_IMPLEMENTATION.md`.
- Deterministic polypharmacy interaction graph (`core/polypharmacy.py`).
- PII scrubbing on voice/inputs (`core/privacy.py`).
- Dashboard (`backend/static/index.html`) with a section per intake type and
  raw JSON output.
- Supabase + OpenAI clients available; Modal serverless GPU optional.
- Design docs: `ARCHITECTURE.md`, `MATH_IMPLEMENTATION.md`,
  `CURRENT_STATE_v0.0.3.md`.

## Project layout

```text
backend/
  main.py                  # app wiring + static dashboard mount
  requirements.txt
  .env.example             # copy to .env and fill in keys (never commit .env)
  config/settings.py
  models/schemas.py
  agents/orchestrator.py
  api/routers/intake.py    # voice / prescription / xray / system-status
  core/
    clinical_agent.py      # fused clinical reasoning
    cv_pipeline.py         # prescription image pipeline
    math_engine.py         # Bayesian clinical math
    meta_agent.py          # Meta reasoning agent
    modal_worker.py        # Modal serverless GPU worker
    polypharmacy.py        # drug-interaction graph
    privacy.py             # PII scrubbing
    prompts.py             # model prompts
  database/
    dims_drug_db.json      # drug reference data
    fallback_fixtures.py   # offline fallback fixtures
  static/index.html        # test dashboard
  test_*.py                # backend tests (pytest)
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

Dependencies: the v0.0.x base set plus pytest, pytest-asyncio, openai,
supabase, networkx, and pydub.

Optional Modal cloud GPU: run `modal setup`, then set `USE_MODAL_CLOUD=true`
in `backend/.env`.

## Tests

```powershell
cd backend
pytest
```

## Notes

- Never commit `backend/.env` (real keys). Voice recordings and prescription
  photos are patient data — keep them local.
- Newest development happens on `v-0.0.3` (adds combined intake, a patient
  dosage-card generator, and Supabase migrations).

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
