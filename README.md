# Nirvor — A Neuro-Symbolic Engine (v0.0.1)

> Meta Global AI Developer Hackathon submission — Team Anindo.
> `v-0.0.1` is the first working prototype: a single-file FastAPI app that
> fuses prescription OCR, regional voice notes, and X-ray scans into one
> clinical intake call. The full math core already lives here — later
> versions reorganize it and add agents, safety, and endpoints around it.

## Endpoints

| Method | Path                      | Purpose                                                        |
|--------|---------------------------|----------------------------------------------------------------|
| GET    | `/`                       | Health check                                                   |
| POST   | `/api/v2/intake-pipeline` | Prescription image (required) + voice audio / X-ray (optional) |

Everything is wired directly in `backend/main.py` (no router split yet).

## Math engine — `core/math_engine.py` (`ClinicalMathEngine`, ~393 lines)

All of this is real NumPy math on CPU (audited in `MATH_IMPLEMENTATION.md`
on later branches):

- `to_phonetic_skeleton()` — strips vowels / non-letters into a phonetic
  signature, with Bangla→English normalization.
- `text_to_embedding()` — hashes character 2-grams/3-grams (original +
  phonetic) via `hashlib.md5` into a **512-dim L2-normalized vector**
  (index = hash bytes mod 512, sign = byte parity). Drug DB vectors are
  precomputed once at startup.
- `compute_cosine_similarity()` — `dot(a,b)/(norm_a*norm_b)` linear scan for
  closest drug match. No vector DB needed.
- `weighted_cursive_levenshtein()` — real 2D dynamic-programming edit
  distance with float costs (0.8 insert/delete; down to 0.15 for cursive
  confusion pairs like `l`/`i`, `m`/`n`).
- `ground_drug_entity()` — **Bayesian cross-modal fusion**: energy score
  from OCR similarity + voice similarity + Levenshtein → numerically stable
  softmax (`log(prior) + temperature*score`, max-shift, `exp`) → true
  posterior distribution + **Shannon entropy bits** (`-Σ P·log2(P)`).
- `align_regional_dialect_symptoms()` — phonetic cosine match of transcripts
  to ICD-10 symptoms in the dialect lexicon.
- `calculate_cardiothoracic_ratio()` — CTR from heart/thorax mask bounding
  boxes (`np.where(mask > 0)`).
- `analyze_mask_eigen_geometry()` — 2D PCA on lesion masks: centroid,
  `np.cov()` 2×2 covariance, `np.linalg.eigh()` eigenvalues + eccentricity.

## CV pipeline — `core/cv_preprocessing.py` (~248 lines)

`ClinicalCVPreprocessor`: illumination/shadow removal (morphological
background estimate + ink reflectance + CLAHE), document corner finding
(Canny + contours), deskew via SVD homography (Hartley-normalized points),
Probabilistic-Hough skew fallback, then a Base64 PNG data URI plus geometry
telemetry for the LLM step.

## Voice + model clients — `core/audio_processor.py`, `core/meta_client.py`

- `process_regional_voice_note()` — Meta transcription API; falls back to a
  hardcoded Chittagonian mock transcript offline.
- `meta_client.py` (193 lines) — 6 async functions: prescription multimodal
  extraction (Baseten → Meta fallback), entropy triage-question generation,
  polypharmacy escalation, SAM 3.1 scan segmentation (OpenCV-threshold mask
  fallback), Meta audio transcription, dosage-card image URL. Every one has
  a `DEFAULT_*` fixture fallback so the pipeline runs with no keys.

## Accuracy & trust — honest notes

- Per-drug **Bayesian posterior + entropy bits** are returned with every
  grounding; high entropy drives a triage question.
- There is **no measured accuracy benchmark** on this branch (no eval set,
  no scoring script) — validation is the three test scripts plus fixture
  runs. Any accuracy number would have to come from a labeled prescription
  set, which doesn't exist in the repo.
- Offline runs return fixture data (clearly fixture-shaped), never fake
  confidence.

## Project layout (this branch)

```text
backend/
  main.py                # FastAPI app + the /api/v2/intake-pipeline endpoint
  requirements.txt       # base deps (below)
  .env.example           # copy to .env and fill in keys (never commit .env)
  core/
    audio_processor.py   # regional voice-note transcription
    cv_preprocessing.py  # prescription cleanup + telemetry
    math_engine.py       # the math above
    meta_client.py       # Baseten/Meta/SAM wrappers + fixtures
    modal_worker.py      # Modal cloud worker (isolated; not called by main flow)
    prompts.py           # model prompts
    schemas.py           # request/response models
  database/
    dims_drug_db.json    # Bangladeshi drug reference data
    fallback_fixtures.py # offline fixtures + OpenCV fallback masks
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

Docs at http://127.0.0.1:8000/docs.

Dependencies: FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy, OpenCV
(headless), Pillow, httpx/requests, python-multipart, python-dotenv, Modal.

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
- Next: `v-0.0.2` keeps this math identical and wraps it in agents.

Other versions: `main`, `v-0.0.1`, `v-0.0.2`, `v-0.0.3`.

## License

MIT — © 2026 Anindo Kumar Biswas. See `LICENSE`.
