# Nirvor — A Neuro-Symbolic Engine (main)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `main` is the stable line: the v0.0.2 monolith split into per-modality
> intake endpoints, with a clinical reasoning agent, a deterministic
> polypharmacy safety graph, PII scrubbing, and a static test dashboard.

## Added in this version (vs v0.0.2)

- **Per-modality endpoints** — voice, prescription, xray each get their own
  route, response model, and orchestrator branch.
- **Orchestrator rewrite** (~259 lines): `_safe_execute` wraps every step so
  one failing branch yields `status: partial_success` + `error_details`
  instead of a 500; `process_voice` / `process_prescription` /
  `process_xray` run the pipelines.
- **Clinical agent** (`core/clinical_agent.py`): `generate_clinical_insights()`
  via `muse-spark-1.3`, strict-JSON dual view — `doctor_view` (summary,
  ICD-10 alignment, triage question, grounded meds, `verification_telemetry`
  with `hallucination_risk_status`) + `patient_view` (layman explanation,
  next steps). Loud errors, never silence.
- **Polypharmacy graph** (`core/polypharmacy.py`): deterministic `networkx`
  graph, `check_interaction()` → risk level, mechanism, action, Bangla
  warning, `provider_used: deterministic_graph`. Seeded with **2 rules**
  (ketorolac+aceclofenac, NSAID+NSAID); else LOW.
- **Privacy shield** (`core/privacy.py`): `scrub_text` / `scrub_data`.
- **Math +1**: `verify_symptom_medication_coherence()`.
- **Meta agent hardening**: `transcribe_audio_meta` + dosage-card fn return
  structured dicts now.
- **Dashboard** (`backend/static/index.html` at `/`): section per intake
  type, raw JSON + deskewed preview.
- **Deps**: + pytest, pytest-asyncio, openai, supabase, networkx, pydub.

## Endpoints

| Method | Path                          | Request (multipart file) |
|--------|-------------------------------|--------------------------|
| GET    | `/health`                     | Liveness check           |
| GET    | `/api/v2/system-status`       | Engine / config status   |
| POST   | `/api/v2/intake/voice`        | `audio` → clinical analysis |
| POST   | `/api/v2/intake/prescription` | `prescription` → full intake |
| POST   | `/api/v2/intake/xray`         | `xray_scan` → full intake |
| GET    | `/`                           | Dashboard (test UI)      |

Response envelope (typical): `status` (`success`/`partial_success`),
branch payload (grounded meds, views, telemetry), `error_details` per
failed step. Interactive docs at `/docs`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/intake/voice -F "audio=@note.mp3"
```

Flow (prescription branch):

```text
client → intake router → process_prescription → CV pipeline
       → LLM extraction → per-drug grounding → coherence check
       → clinical agent → polypharmacy graph → privacy scrub → response
```

## Math engine — v0.0.1 core + coherence

512-dim phonetic embeddings, cosine similarity, cursive-weighted
Levenshtein, Bayesian posterior + Shannon entropy, dialect→ICD-10
alignment, CTR, lesion PCA (audited real in `MATH_IMPLEMENTATION.md`),
plus `verify_symptom_medication_coherence()`.

## What we gained in this version

- **Modality independence**: test/fix/ship voice without touching xray.
- **Graceful degradation**: partial results with named errors instead of
  all-or-nothing 500s.
- **Two-audience reasoning**: doctor-grade + patient-friendly outputs from
  one call, with a hallucination-risk flag.
- **First real safety rail**: deterministic interaction check with Bangla
  warnings (small but honest).
- **A clickable product**: the dashboard turns curl-only testing into a UI.

## Accuracy & trust — honest notes

- Signals per call: posteriors, `mean_shannon_entropy`, coherence flags,
  `hallucination_risk_status`, per-branch `error_details`; triage on high
  entropy.
- **No measured accuracy benchmark** (no labeled eval set). Posteriors are
  ranking signals, not validated %.
- Offline paths are fixture-backed; the LLM polypharmacy escalation is
  fixture-delegated when unconfigured — the 2-rule graph is the only
  always-real check. Grow it before any safety claim.
- `modal_worker.py` still isolated from the flow.

## Project layout

```text
backend/
  main.py  requirements.txt  .env.example
  config/settings.py  models/schemas.py  agents/orchestrator.py
  api/routers/intake.py
  core/
    clinical_agent.py cv_pipeline.py math_engine.py meta_agent.py
    modal_worker.py polypharmacy.py privacy.py prompts.py
  database/  static/index.html
  test_api.py test_baseten.py test_cloud_cv.py test_modal.py
ARCHITECTURE.md  MATH_IMPLEMENTATION.md  CURRENT_STATE_v0.0.3.md
test_api.py
```

Note: `CURRENT_STATE_v0.0.3.md` still describes the old single-pipeline
flow — the endpoint tables above (and code) are authoritative for `main`.

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

Optional Modal GPU: `modal setup` + `USE_MODAL_CLOUD=true`.

## Tests

```powershell
cd backend
pytest
```

## What's next (delivered in v0.0.3)

- Combined voice+prescription intake with cross-modal fusion.
- Real dosage-card output (local renderer, not a placeholder URL).
- Referral matching + verification telemetry, Supabase migrations.
- Gating tests that enforce coherence and skip unverified meds.

## Notes

- Never commit `backend/.env`. Voice/prescription files are patient data.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
