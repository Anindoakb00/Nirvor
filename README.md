# Nirvor — A Neuro-Symbolic Engine (main)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `main` is the stable line: the v0.0.2 monolith split into per-modality
> intake endpoints, with a clinical reasoning agent, a deterministic
> polypharmacy safety graph, PII scrubbing, and a static test dashboard.

## Added in this version (vs v0.0.2)

- **Per-modality endpoints** (below) — voice, prescription, and xray each
  get their own route, response model, and orchestrator branch instead of
  one combined call.
- **Orchestrator rewrite** (~259 lines): `_safe_execute` wraps every step so
  one failing branch yields `status: partial_success` + `error_details`
  instead of a 500; `process_voice` / `process_prescription` /
  `process_xray` run the modality pipelines.
- **Clinical agent** (`core/clinical_agent.py`, ~79 lines):
  `generate_clinical_insights()` calls `muse-spark-1.3` with a strict-JSON
  system prompt returning a dual view — `doctor_view` (clinical summary,
  ICD-10 alignment, triage question, grounded medications,
  `verification_telemetry` with `hallucination_risk_status`) and
  `patient_view` (layman explanation + next steps). Raises loudly if the
  Meta key is unconfigured; returns a `BACKEND ERROR` payload (never
  silence) on failure.
- **Polypharmacy graph** (`core/polypharmacy.py`, ~62 lines):
  deterministic `networkx` graph, `check_interaction()` over medication
  pairs, returning `risk_level`, mechanism, recommended action, a Bangla
  patient warning, and `provider_used: deterministic_graph`. Seeded with
  exactly **2 rules** (ketorolac+aceclofenac, NSAID+NSAID); anything else
  returns LOW. Honest scope: a real but tiny rule base — see trust notes.
- **Privacy shield** (`core/privacy.py`): `scrub_text` / `scrub_data` PII
  scrubbing for text and nested payloads.
- **Math +1**: `verify_symptom_medication_coherence()` in the math engine
  (symptom↔medication coherence scoring).
- **Meta agent hardening**: same 6 functions, but `transcribe_audio_meta`
  and `generate_patient_dosage_card` now return structured dicts.
- **Dashboard** (`backend/static/index.html`, served at `/`): one section
  per intake type with raw JSON output + deskewed image preview.
- **Deps**: + pytest, pytest-asyncio, openai, supabase, networkx, pydub.

## Endpoints

| Method | Path                              | Purpose                                |
|--------|-----------------------------------|----------------------------------------|
| GET    | `/health`                         | Liveness check                         |
| GET    | `/api/v2/system-status`           | Engine / configuration status          |
| POST   | `/api/v2/intake/voice`            | Voice note → `ClinicalAnalysisResponse`|
| POST   | `/api/v2/intake/prescription`     | Prescription image → full intake       |
| POST   | `/api/v2/intake/xray`             | X-ray scan → full intake               |
| GET    | `/`                               | Static dashboard (test UI)             |

Interactive API docs at `/docs` (served alongside the dashboard).

## Math engine — v0.0.1 core + coherence

Everything from v0.0.1 (512-dim phonetic embeddings, cosine similarity,
cursive-weighted Levenshtein, Bayesian posterior + Shannon entropy,
dialect→ICD-10 alignment, CTR, lesion PCA — audited real in
`MATH_IMPLEMENTATION.md`), plus `verify_symptom_medication_coherence()`.

## Accuracy & trust — honest notes

- Confidence signals per call: Bayesian posteriors, `mean_shannon_entropy`,
  coherence flags, `hallucination_risk_status`, and per-branch
  `error_details`. High entropy triggers an LLM triage question.
- There is **no measured accuracy benchmark** in the repo (no labeled eval
  set / scoring script). Treat posteriors as ranking signals, not validated
  accuracy %.
- Offline behavior is fixture-backed (prescription, transcript, masks,
  triage, polypharmacy) — the MATH audit flags the LLM polypharmacy
  escalation path as fixture-delegated when unconfigured. The deterministic
  graph above is the only always-real interaction check, and it holds 2
  rules. Grow it before any safety claim.
- `modal_worker.py` is still isolated from the orchestrator flow.

## Project layout

```text
backend/
  main.py                  # wiring + static dashboard mount + GET /health
  requirements.txt
  .env.example             # copy to .env and fill in keys (never commit .env)
  config/settings.py  models/schemas.py  agents/orchestrator.py
  api/routers/intake.py    # voice / prescription / xray / system-status
  core/
    clinical_agent.py cv_pipeline.py math_engine.py meta_agent.py
    modal_worker.py polypharmacy.py privacy.py prompts.py
    (audio transcription lives in meta_agent on this branch)
  database/
    dims_drug_db.json fallback_fixtures.py
  static/index.html
  test_api.py test_baseten.py test_cloud_cv.py test_modal.py
ARCHITECTURE.md  MATH_IMPLEMENTATION.md  CURRENT_STATE_v0.0.3.md
test_api.py
```

Note: `CURRENT_STATE_v0.0.3.md` was written against the older single-pipeline
flow — the endpoint tables above (and the code) are authoritative for `main`.

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

Optional Modal cloud GPU: run `modal setup`, then `USE_MODAL_CLOUD=true`.

## Tests

```powershell
cd backend
pytest
```

## Notes

- Never commit `backend/.env`. Voice/prescription files are patient data.
- Newest work is on `v-0.0.3`: combined intake, local dosage-card renderer,
  referral matching + telemetry, Supabase migrations, gating tests.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
