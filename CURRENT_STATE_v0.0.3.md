# Nirvor (v0.0.3) - Deep Codebase Audit & Execution Analysis

## 1. What is Inside the Code (Module-by-Module Breakdown)

### Repository File List
- `ARCHITECTURE.md` - System architecture documentation.
- `test_api.py` - Script to locally test the FastAPI endpoint.
- `backend/main.py` - FastAPI entry point.
- `backend/requirements.txt` - Python package dependencies.
- `backend/.env.example` - Example environment variables configuration.
- `backend/test_baseten.py` - Test script for Baseten model endpoint.
- `backend/test_cloud_cv.py` - Test script for Modal CV endpoint.
- `backend/test_modal.py` - Basic test script for Modal cloud.
- `backend/api/routers/intake.py` - FastAPI router for main intake pipeline.
- `backend/agents/orchestrator.py` - Orchestration logic for AI pipeline.
- `backend/config/settings.py` - Configuration and environment variable loader.
- `backend/core/audio_pipeline.py` - Module for audio processing.
- `backend/core/cv_pipeline.py` - Classical computer vision processing for images.
- `backend/core/math_engine.py` - Bayesian math, string similarity, and clinical logic.
- `backend/core/meta_agent.py` - Functions coordinating external Meta/Baseten APIs.
- `backend/core/modal_worker.py` - Modal serverless worker configuration.
- `backend/core/prompts.py` - String templates for LLM prompts.
- `backend/database/dims_drug_db.json` - Static dataset of Bangladeshi drugs.
- `backend/database/fallback_fixtures.py` - Mock data fallback objects.
- `backend/models/schemas.py` - Pydantic validation schemas.
- `backend/static/index.html` - **MISSING** - Frontend UI is not implemented.

### Module Breakdown

#### `backend/main.py`
- **FastAPI App**: Initializes the FastAPI web application, sets up CORS middleware to allow all origins.
- **`health_check()`**: GET `/` route that simply returns a static dictionary indicating the service is online.
- **Router inclusion**: Includes the `intake_router` from the `api/routers/intake` module.

#### `backend/api/routers/intake.py`
- **`intake_pipeline()`**: POST `/api/v2/intake-pipeline` route. Accepts multipart form data containing a `prescription` image file (required), `audio` file (optional), and `xray_scan` file (optional). It reads the bytes of the provided files and invokes `orchestrator.process_intake()` passing the read bytes, then returns the result. Catches any exception and returns a 500 error.

#### `backend/agents/orchestrator.py`
- **`IntakeOrchestrator` (Class)**
  - **`__init__()`**: Initializes an instance of `ClinicalMathEngine`.
  - **`process_intake()`**: The core pipeline flow.
    1. Starts a timer.
    2. Runs synchronous Computer Vision preprocessing via `ClinicalCVPreprocessor.process_prescription_image()`.
    3. Asynchronously fires off `process_regional_voice_note()` and `segment_medical_scan_sam3()` (if audio and xray are present).
    4. Waits for asynchronous tasks to complete.
    5. Feeds the resulting voice transcript to the Math Engine's `align_regional_dialect_symptoms()` to map dialect.
    6. Constructs a dialect context string and calls the multimodal extraction LLM agent (`extract_prescription_multimodal`) with the image data URI and context.
    7. Loops through extracted medications and runs Bayesian cross-modal fusion via `math_engine.ground_drug_entity()` to correct OCR errors and determine confidence. Aggregates entropy and flags.
    8. If an X-Ray was provided, calculates Cardiothoracic Ratio and lesion eigen geometry using the masks returned by the SAM 3 model.
    9. Returns the final consolidated JSON payload.

#### `backend/core/cv_pipeline.py`
- **`ClinicalCVPreprocessor` (Class)**
  - **`_normalize_points_hartley()`**: Shifts and scales 2D coordinate points for numerical stability before calculating Homography via SVD.
  - **`compute_svd_homography()`**: Computes a transformation matrix (Homography) mapping source points to destination points using Singular Value Decomposition, and returns the matrix with telemetry data.
  - **`remove_illumination_shadows()`**: Applies morphological closing to estimate background illumination, divides the original by the background to extract ink reflectance, and applies CLAHE for contrast enhancement.
  - **`_order_quad_points()`**: Orders a 4-point array into top-left, top-right, bottom-right, bottom-left.
  - **`_hough_deskew_fallback()`**: Tries to detect horizontal text lines using Probabilistic Hough Transform and calculates a median skew angle to rotate and align the image if the document edges aren't found.
  - **`process_prescription_image()`**: The main CV pipeline. Decodes the image, removes shadows, attempts to find document corners using Canny edge detection and Contours. If 4 corners are found occupying enough space, it deskews via Homography. Otherwise, it falls back to Hough deskew. It blends the result, encodes it into a Base64 PNG data URI, and returns the image array, data URI, and geometry telemetry.
- **`deskew_prescription_image()`**: Helper function wrapping `process_prescription_image()`.

#### `backend/core/audio_pipeline.py`
- **`process_regional_voice_note()`**: Takes audio bytes and makes an asynchronous HTTP POST request to a Meta API (`META_BASE_URL/audio/transcriptions`). If the `META_API_KEY` is not set, or if the request fails/times out, it returns a hardcoded mock JSON `fallback_response` (Chittagonian dialect).

#### `backend/core/math_engine.py`
- **`ClinicalMathEngine` (Class)**
  - **`__init__()`**: Loads the drug database (`dims_drug_db.json`), pre-computes phonetic skeletons and vector embeddings for all drugs in the database and a predefined dialect lexicon.
  - **`_clean_clinical_title()`**: Lowercases, strips, and removes generic prefixes like "tab.", "cap." from medication names.
  - **`to_phonetic_skeleton()`**: Removes vowels and non-alphabetic characters from a string to create a phonetic signature.
  - **`text_to_embedding()`**: Creates a hashed frequency vector representing character bigrams and trigrams (weighted heavier for phonetic skeletons) to serve as a fast vector embedding.
  - **`compute_cosine_similarity()`**: Calculates cosine similarity between two numeric vectors.
  - **`weighted_cursive_levenshtein()`**: Calculates edit distance between two strings using dynamic programming, but with custom weights for common cursive handwriting confusion pairs (e.g. 'm'/'n', 'l'/'t').
  - **`ground_drug_entity()`**: Takes an OCR medication string and an optional voice transcript. Runs Bayesian cross-modal fusion. Calculates similarity between the OCR input and every drug in the database using cosine similarity, voice phonetic alignment, and weighted Levenshtein distance. Computes posterior probabilities via Softmax, Shannon entropy, and returns the highest confidence drug match along with metrics.
  - **`align_regional_dialect_symptoms()`**: Uses phonetic cosine similarity to match the raw transcript to the closest ICD-10 clinical symptoms defined in the dialect lexicon.
  - **`calculate_cardiothoracic_ratio()`**: Analyzes 2D binary segmentation masks (heart and thorax) to compute the transverse diameter ratio.
  - **`analyze_mask_eigen_geometry()`**: Performs 2D Principal Component Analysis (PCA) on a lesion mask to extract area, orientation, eigenvalues, and eccentricity.

#### `backend/core/meta_agent.py`
- Uses OpenAI client interfaces configured for Meta and Baseten endpoints.
- **`extract_prescription_multimodal()`**: Tries Baseten API then Meta API to extract text from the prescription image. If both fail or are unconfigured, returns `DEFAULT_PRESCRIPTION_FIXTURE` mock data.
- **`generate_entropy_triage_question()`**: Asks the LLM to generate an ambiguous triage question. Tries Meta then Baseten APIs. Falls back to `DEFAULT_TRIAGE_QUESTION_BN`.
- **`escalate_polypharmacy_wonder()`**: Asks the Muse Wonder model for a polypharmacy escalation plan. Falls back to `DEFAULT_POLYPHARMACY_FIXTURE`.
- **`segment_medical_scan_sam3()`**: Calls Meta's SAM 3.1 endpoint to get segmentation masks from X-Rays. Falls back to deterministic OpenCV threshold masks via `build_opencv_fallback_masks()`.
- **`transcribe_audio_meta()`**: Transcribes audio. Redundant with `audio_pipeline.py`. Falls back to `DEFAULT_CHITTAGONIAN_TRANSCRIPT`.
- **`generate_patient_dosage_card()`**: Calls Muse Image generation. Falls back to a Placeholder URL.

#### `backend/core/modal_worker.py`
- Defines a Modal app (`cloud-worker`) to run a CV CLAHE deskew script in the cloud. However, this is isolated and **never imported or invoked** in the main orchestrator flow.

#### `backend/database/fallback_fixtures.py`
- Contains Pydantic model instances representing mock fallback data.
- **`build_opencv_fallback_masks()`**: Creates synthetic heart/thorax/lesion masks using basic OpenCV thresholding of the uploaded X-Ray to allow math pipeline execution without the real SAM 3 model.

#### `backend/models/schemas.py`
- Defines Pydantic data schemas: `MedicationItem`, `PatientVitals`, `PrescriptionExtraction`, `PolypharmacyIntervention`. These are used primarily in prompts and fallback fixtures, but are not strongly enforced on the actual router output.


## 2. What It Tries to Do (End-to-End Execution Trace)

1. **Upload**: A user hits `POST /api/v2/intake-pipeline` with a prescription image (required), and optionally audio/xray scans. *(Note: Because `index.html` is completely missing, there is no frontend to do this; it must be done via cURL, Postman, or a custom script like `test_api.py`.)*
2. **Routing**: `main.py` -> `intake.py` Router intercepts the request, reads file bytes, and passes them to `IntakeOrchestrator.process_intake`.
3. **Synchronous CV**: Orchestrator synchronously calls `ClinicalCVPreprocessor.process_prescription_image`. The image undergoes Retinex Shadow Removal, Homography Deskew (or Hough Fallback), and CLAHE. It returns a Base64 data URI of the cleaned image.
4. **Asynchronous Parallel Processing**: Orchestrator spins up two parallel tasks:
   - Voice audio is sent to Meta API (via `audio_pipeline.process_regional_voice_note`).
   - X-Ray is sent to Meta SAM 3 API (via `meta_agent.segment_medical_scan_sam3`).
5. **Wait & Align**: Once parallel tasks finish, the voice transcript is passed to `ClinicalMathEngine.align_regional_dialect_symptoms()` to map the dialect to standard ICD-10 symptoms.
6. **Multimodal LLM Extraction**: Orchestrator calls `meta_agent.extract_prescription_multimodal` with the Base64 image and dialect context. This agent calls Baseten/Meta API to return a structured JSON array of extracted raw medication text.
7. **Math Engine Grounding (The Neuro-Symbolic Loop)**: For each extracted medication, the Orchestrator feeds the raw OCR string and the voice transcript into `math_engine.ground_drug_entity()`. The engine compares this against `dims_drug_db.json` using Cosine Similarity, Phonetic matching, and Weighted Cursive Levenshtein distance. It calculates Bayesian posterior probabilities, applies corrections, and measures Shannon entropy.
8. **Scan Biometrics**: If masks are returned from the X-Ray step, `math_engine` calculates the Cardiothoracic Ratio and Lesion PCA Eigen geometry.
9. **HUD Summary & Output**: Finally, execution time and entropy aggregates are tallied, and the Orchestrator returns a monolithic JSON object encapsulating the CV telemetry, alignment, grounded medications, biometrics, and a judge summary block.


## 3. Expected Output (What Will Be the Output?)

If executed locally without Baseten/Meta API keys properly configured in `.env`, the output relies entirely on fallback mock fixtures (`DEFAULT_PRESCRIPTION_FIXTURE`, `DEFAULT_CHITTAGONIAN_TRANSCRIPT`, `build_opencv_fallback_masks`).

**Expected JSON Response (Mock/Fallback execution):**
```json
{
  "status": "success",
  "cv_telemetry": {
    "original_resolution": [100, 100],
    "output_resolution": [100, 100],
    "illumination_normalization": "Retinex_Morphological_Division + CLAHE_2.2",
    "geometry": {
      "method": "Hough_SE2_Affine_Rotation_Fallback",
      "detected_skew_angle_deg": 0.0,
      "sampled_text_lines": 0,
      "rotation_matrix_2x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    }
  },
  "voice_intake": null,
  "dialect_icd10_alignment": [],
  "grounded_medications": [
    {
      "raw_text": "Tab. Sarjel 20mg",
      "dosage": "20mg",
      "frequency": "1+0+1",
      "duration_days": 14,
      "raw_ocr_input": "Tab. Sarjel 20mg",
      "matched_drug": {
        "id": "BD_DIMS_002",
        "brand_name": "Sergel 20",
        "generic_name": "Esomeprazole Magnesium Trihydrate 20mg",
        "manufacturer": "Healthcare Pharmaceuticals Ltd.",
        "category": "Proton Pump Inhibitor (PPI)",
        "bangla_phonetic_aliases": ["সার্জেল ২০", "sergel", "sarjel", "সারজেল"],
        "prior_probability": 0.16,
        "half_life_hours": 1.5,
        "nephrotoxic_risk": 0.1,
        "hepatotoxic_threshold_mg": 240
      },
      "bayesian_posterior_confidence": 0.956,
      "shannon_entropy_bits": 0.2765,
      "ocr_typo_corrected": true,
      "requires_active_verification": false,
      "top_3_manifold_candidates": [
        {"brand_name": "Sergel 20", "generic_name": "Esomeprazole Magnesium Trihydrate 20mg", "posterior_prob": 0.956, "cos_ocr": 0.2331, "cos_voice": 0.0, "weighted_levenshtein": 0.2636},
        {"brand_name": "Seclo 20", "generic_name": "Omeprazole 20mg", "posterior_prob": 0.0421, "cos_ocr": 0.2412, "cos_voice": 0.0, "weighted_levenshtein": 0.4773},
        {"brand_name": "Maxpro 40", "generic_name": "Esomeprazole 40mg", "posterior_prob": 0.0004, "cos_ocr": 0.1291, "cos_voice": 0.0, "weighted_levenshtein": 0.6909}
      ]
    },
    ...
  ],
  "scan_biometrics": {},
  "judge_hud_summary": {
    "total_latency_ms": 20.57,
    "hallucinations_intercepted": 3,
    "mean_shannon_entropy": 0.1008,
    "active_verification_flags": 0
  }
}
```

**Dynamics:**
- The CV telemetry (`cv_telemetry`), Mathematical Bayesian matching output (`grounded_medications.*` calculations), and Mask algorithms (`scan_biometrics`) are executing **real** logic on the CPU natively.
- Due to a missing `.env` API Key, the raw OCR extraction and SAM3 models are silently catching exceptions and returning **hardcoded mock data**.


## 4. Completeness Check (What Works vs. What is Incomplete/Broken)

### 🟢 Working (100% Functional)
- **FastAPI Core Framework**: Routing, app lifecycle, file ingestion.
- **Agent Orchestrator Loop**: Correctly fires parallel requests, handles sync vs async.
- **Clinical Math Engine**: The neuro-symbolic algorithms (Levenshtein, Cosine, Posterior Probabilities) are extremely robust and fully functional.
- **CV Preprocessor Pipeline**: Homography deskewing and Retinex shadow removal work perfectly.
- **Fallback Mock Mechanisms**: API key absence gracefully falls back to mock responses, preventing the server from crashing.

### 🔴 Incomplete / Broken / Disconnected
1. **Missing Frontend UI (`backend/static/index.html`)**: There is absolutely no frontend code inside the repository. Users cannot interact with the system via a browser.
2. **Unused Serverless Components (`backend/core/modal_worker.py`)**: The Modal CV worker script exists but is never imported or utilized by the `IntakeOrchestrator` or CV Pipeline.
3. **Redundant Audio Pipeline Logic**: Both `core/audio_pipeline.py` and `core/meta_agent.py` contain distinct functions for transcribing audio (`process_regional_voice_note` vs `transcribe_audio_meta`). Only the former is actually called in the Orchestrator.
4. **Missing Prompt Implementations**: Functions like `generate_entropy_triage_question`, `escalate_polypharmacy_wonder`, and `generate_patient_dosage_card` in `meta_agent.py` exist but are never called anywhere in the Orchestrator pipeline.
5. **No Actual Database Connection**: The "database" is just a static JSON file (`dims_drug_db.json`). There is no Postgres/SQLite usage as one might expect in a production environment.


## 5. Immediate Action Plan for Next Version (v0.0.4)

To complete the pipeline for the hackathon submission, prioritize the following tasks:

1. **Develop and Mount Frontend UI (`index.html`)**
   - Create a Vue.js or Vanilla JS single-page application in `backend/static/index.html`.
   - Setup a `FastAPI.mount()` call in `main.py` to serve this static file.
   - Build a file upload form that maps directly to the `/api/v2/intake-pipeline` multipart format.
2. **Integrate Unused Meta Agent Features**
   - Update `IntakeOrchestrator` to invoke `escalate_polypharmacy_wonder` if the combined drug lists flag critical interactions.
   - Update `IntakeOrchestrator` to invoke `generate_patient_dosage_card` using the grounded drug list to output a URL for the frontend.
3. **Consolidate Cloud & Pipeline Logic**
   - Either remove `modal_worker.py` or integrate it as a fallback inside `ClinicalCVPreprocessor`.
   - Remove redundant `transcribe_audio_meta` in `meta_agent.py` in favor of `audio_pipeline.py`.
4. **Environment Setup Guidance**
   - Provide a `.env` loader script or explicit error raising if API keys are missing when the user explicitly requests production mode (so judges don't accidentally evaluate the mock data without realizing it).
