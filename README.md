# Nirvor — A Neuro-Symbolic Engine (v0.0.1)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.1` is the first working prototype: a single-file FastAPI app that
> fuses prescription OCR, regional voice notes, and X-ray scans into one
> clinical intake call. The full math core already lives here — later
> versions reorganize it and add agents, safety, and endpoints around it.

## Endpoints

| Method | Path                      | Request (multipart)                                        | Response |
|--------|---------------------------|------------------------------------------------------------|----------|
| GET    | `/`                       | —                                                          | Health JSON (`status`, `engine`, `team`) |
| POST   | `/api/v2/intake-pipeline` | `prescription` (required), `audio` + `xray_scan` (optional) | Consolidated intake JSON (grounded meds, alignment, biometrics, telemetry) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/intake-pipeline \
  -F "prescription=@rx.jpg" -F "audio=@note.mp3"
```

Pipeline flow:

```text
client → main.py → CV cleanup → ┬─ voice transcription ─┐
                                └─ SAM 3.1 segmentation ┘
       → dialect alignment → LLM extraction → per-drug Bayesian grounding
       → CTR / lesion biometrics → one JSON response
```

## Math engine — `core/math_engine.py` (`ClinicalMathEngine`, ~393 lines)

All real NumPy math on CPU (audited in `MATH_IMPLEMENTATION.md` on later
branches):

- `to_phonetic_skeleton()` — phonetic signature with Bangla→English
  normalization.
- `text_to_embedding()` — char 2/3-gram hashing via `hashlib.md5` into a
  **512-dim L2-normalized vector**; drug DB vectors precomputed at startup.
- `compute_cosine_similarity()` — dot-product scan, no vector DB needed.
- `weighted_cursive_levenshtein()` — 2D DP edit distance with float costs
  (0.8 insert/delete; ~0.15 for cursive pairs like `l`/`i`, `m`/`n`).
- `ground_drug_entity()` — **Bayesian cross-modal fusion**: energy score
  (OCR + voice + Levenshtein) → stable softmax → posterior distribution +
  **Shannon entropy bits** (`-Σ P·log2(P)`).
- `align_regional_dialect_symptoms()` — transcript → ICD-10 symptoms via
  phonetic cosine match.
- `calculate_cardiothoracic_ratio()` — CTR from mask bounding boxes.
- `analyze_mask_eigen_geometry()` — lesion PCA: centroid, covariance,
  eigendecomposition, eccentricity.

## CV + voice + model clients

- `core/cv_preprocessing.py` (~248 lines): shadow removal, SVD-homography
  deskew (Hartley-normalized), Hough skew fallback, CLAHE → Base64 data URI
  + geometry telemetry.
- `core/audio_processor.py`: Meta transcription; Chittagonian mock fallback.
- `core/meta_client.py` (193 lines), 6 async functions: multimodal
  extraction (Baseten → Meta), entropy triage questions, polypharmacy
  escalation, SAM 3.1 segmentation (OpenCV-mask fallback), audio
  transcription, dosage-card URL. All `DEFAULT_*` fixture-backed offline.

## What we gained in this version

- First **end-to-end proof**: photo + voice + scan in, grounded clinical
  JSON out.
- The **entire neuro-symbolic math core** (embeddings → Bayes → entropy →
  biometrics) that every later version reuses untouched.
- Offline resilience pattern: fixtures for every external call.
- Baseten + Meta + SAM + Modal integrations proven with test scripts.

## Accuracy & trust — honest notes

- Per-drug **posterior + entropy bits**; high entropy drives triage.
- **No measured accuracy benchmark** (no eval set/scoring script) —
  validation is the test scripts + fixture runs.
- Offline runs return clearly fixture-shaped data, never fake confidence.

## Project layout (this branch)

```text
backend/
  main.py                # app + endpoint (single-file style)
  requirements.txt       # base deps (below)
  .env.example           # copy to .env (never commit .env)
  core/
    audio_processor.py cv_preprocessing.py math_engine.py
    meta_client.py modal_worker.py prompts.py schemas.py
  database/
    dims_drug_db.json    # Bangladeshi drug reference
    fallback_fixtures.py # offline fixtures + OpenCV fallback masks
  test_baseten.py test_cloud_cv.py test_modal.py
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

Docs at http://127.0.0.1:8000/docs. Deps: FastAPI, Uvicorn, Pydantic v2,
NumPy, SciPy, OpenCV (headless), Pillow, httpx/requests, python-multipart,
python-dotenv, Modal.

## Tests

```powershell
cd backend
python test_baseten.py
python test_cloud_cv.py
python test_modal.py
```

## What's next (delivered in v0.0.2)

- Extract the endpoint into an intake router + orchestrator agent.
- Split config, schemas, and pipelines into packages (`config/`, `models/`).
- Same math, same API surface — pure architecture upgrade.

## Notes

- No frontend — use `/docs`, cURL, or `test_api.py`.
- Never commit `backend/.env`. Voice/prescription files are patient data.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
