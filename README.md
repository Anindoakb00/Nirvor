# Nirvor — A Neuro-Symbolic Engine (v0.0.3)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.3` is the latest development branch: everything on `main`, plus
> combined voice+prescription intake, a local patient dosage-card renderer,
> prescription-referral matching with telemetry, Supabase migrations, and
> verification gating tests.

## Added in this version (vs `main`)

- **Combined intake** — `POST /api/v2/intake/combined` → `process_combined()`
  + `_build_cross_modal_fusion()`: voice and prescription branches share
  helpers (`_transcribe_and_align`, `_clinical_insights_with_telemetry`,
  `_rx_branch`, `_xray_branch`, `_xray_flags_and_summary`) and fuse into
  one `CombinedIntakeResponse`. Orchestrator grows to 13 methods.
- **Dosage card, drawn locally** (`core/dosage_card.py`): Meta's image model
  refuses medication prompts by policy, so the card renders offline with
  Pillow — `parse_dosage_summary()` → `CardRow` → `render_dosage_card_png()`
  (900px, auto-fit fonts, zebra rows; Medicine / Morning / Noon / Night /
  Schedule with `1+0+1` patterns). Exact names, no network, no cost.
  English headers (PIL can't shape Bangla). Fixes the Section 4 bug where
  `patient_dosage_card` was `null` + 400 in `error_details`.
- **Math +4**: `build_verification_telemetry()`, `build_medication_blob()`,
  `match_prescription_referrals()`, `build_referral_telemetry()`.
- **Supabase grounding**: migrations `001_symptoms.sql` +
  `002_prescription_referrals.sql`, importers, `SUPABASE_URL`/`SUPABASE_KEY`.
- **Gating tests**: coherence enforced (`is_coherent`, `coherence_score`,
  flags like "require verification" / "1 of 2"); safety gating counts
  `verified_medications_assessed` vs `unverified_medications_skipped`.
- **Nov 5–7 fixes**: clinical agent feeds both doctor + patient views,
  meta-agent intake fixes, `pillow` in backend venv, 11 backend tests.
- **Hygiene**: `.gitignore` blocks secrets + patient data; IDE/logs
  untracked.

## Endpoints

| Method | Path                          | Request (multipart files) |
|--------|-------------------------------|---------------------------|
| GET    | `/health`                     | Liveness check            |
| GET    | `/api/v2/system-status`       | Engine / config status    |
| POST   | `/api/v2/intake/voice`        | `audio` → clinical analysis |
| POST   | `/api/v2/intake/prescription` | `prescription` → full intake incl. dosage card |
| POST   | `/api/v2/intake/xray`         | `xray_scan` → full intake |
| POST   | `/api/v2/intake/combined`     | `prescription` + `audio` (+`xray_scan`) → fused intake |
| GET    | `/`                           | Dashboard (test UI)       |

Response envelope: `status`, branch payloads (`doctor_view` /
`patient_view`, grounded meds, telemetry, `patient_dosage_card`),
`error_details` per failed step. Docs at `/docs`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/intake/combined \
  -F "prescription=@rx.jpg" -F "audio=@note.mp3"
```

Flow (combined):

```text
client → process_combined → ┬ voice branch (transcribe → align → insights)
                            └ prescription branch (CV → extract → ground)
       → cross-modal fusion → referrals + verification telemetry
       → dosage-card render → safety graph → privacy scrub → response
```

## Math engine — 15 methods

v0.0.1's 10 + `main`'s coherence check + the 4 telemetry/referral builders.
Audit: `MATH_IMPLEMENTATION.md`. Response models (15 in
`models/schemas.py`): intake, combined/fusion, coherence, verification +
referral telemetry, doctor/patient views, judge HUD, execution trace.

## What we gained in this version

- **True multimodality**: voice + prescription + xray fuse in one call with
  a shared telemetry story.
- **A real patient artifact**: the dosage card is now data + rendered PNG,
  not a placeholder URL — the product's visible output.
- **Test-enforced trust**: gating tests assert exact coherence flags and
  force unverified meds out of safety assessment.
- **Production footing**: migrations, importers, secret hygiene, 11 tests.

## Accuracy & trust — honest notes

- Strongest branch: coherence is test-enforced, unverified meds are
  explicitly skipped (counted, not hidden), partial failures are named.
- **Still no measured accuracy benchmark** — the card fix was verified by
  rendering + Section 4 re-test comparison, not a metric. The eval set is
  the next trust milestone (see below).
- Standing caveats: fixture-backed offline paths, 2-rule graph, isolated
  Modal worker.

## Project layout (additions vs `main`)

```text
backend/
  core/dosage_card.py
  database/
    supabase_migrations/001_symptoms.sql  002_prescription_referrals.sql
    import_prescription_referrals.py  import_verified_lexicon.py
  test_dosage_card.py test_combined_intake.py test_verification_gating.py
  test_vector_ingest.py test_clinical_prompt.py test_prescription_referrals.py
  test_cv_pipeline.py  (+ test_api/baseten/cloud_cv/modal)
```

## Setup

Prereqs: Python 3.10+.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env: BASETEN_API_KEY + META_API_KEY
# optional: SUPABASE_URL + SUPABASE_KEY for pgvector grounding
python -m uvicorn main:app --reload --port 8000
```

Deps: FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy, OpenCV (headless),
Pillow, httpx/requests, python-multipart, python-dotenv, Modal, pytest,
pytest-asyncio, openai, supabase, networkx, pydub. Optional Modal GPU:
`modal setup` + `USE_MODAL_CLOUD=true`.

## Tests

```powershell
cd backend
pytest
```

## What's next (planned — not built yet)

Headline for the next version: **frontend**. The backend can reason; now
patients and doctors need to see it:

- **Dosage-card display** in the dashboard (rendered PNG + data), replacing
  raw JSON for patients.
- **Triage-question UI**: show `active_triage_question` and accept the
  follow-up answer back into intake.
- **History + polish**: past intakes list, verification telemetry /
  hallucination-risk display, styling pass.
- Behind the UI: grow the polypharmacy graph beyond 2 rules; build the
  first labeled eval set + scoring script (first real accuracy %); wire the
  Modal worker into the orchestrator (or remove it); Dockerfile + CI;
  reconcile version strings and the `baseten_configured` status field.

## Notes

- Never commit `backend/.env`. Voice/prescription files are patient data;
  this branch's `.gitignore` already excludes them.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
