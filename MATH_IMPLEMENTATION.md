# Audit of Mathematical & Vector Implementations (v0.0.3)

This document provides a comprehensive audit of the mathematical concepts used in the backend, verifying whether they are executing real clinical/vector math or relying on hardcoded fixtures/mock values.

## 1. List of Mathematical Concepts Claimed & Evaluated

### 1.1. Phonetic Cosine Similarity (`phonetic_cosine_similarity`)
*   **File Path:** `backend/core/math_engine.py`
*   **Function:** `text_to_embedding()`, `compute_cosine_similarity()`, and `align_regional_dialect_symptoms()`
*   **Truth Check: ✅ REAL.**
    The system does *not* use mocked metrics. It leverages `hashlib` to hash character n-grams and phonetic skeletons into a 512-dimensional vector. The cosine similarity is dynamically computed using `numpy.dot()` and `numpy.linalg.norm()` directly in Python.

### 1.2. Cursive-Weighted Dynamic Programming Levenshtein Distance (`weighted_levenshtein`)
*   **File Path:** `backend/core/math_engine.py`
*   **Function:** `weighted_cursive_levenshtein()`
*   **Truth Check: ✅ REAL.**
    This executes a real 2D dynamic programming grid using a `numpy.zeros` array to compute edit distance. Instead of standard `+1` cost, it applies float penalties (e.g., 0.8 for inserts/deletes, and custom weights down to 0.15 for cursive visual confusion pairs like 'l' and 'i').

### 1.3. Bayesian Posterior Probability (`bayesian_posterior_confidence`, `composite_energy`)
*   **File Path:** `backend/core/math_engine.py`
*   **Function:** `ground_drug_entity()`
*   **Truth Check: ✅ REAL.**
    This uses a mathematical energy-based softmax function over the database candidates. It calculates an energy score combining OCR similarity, voice similarity, and weighted Levenshtein. It then computes the unnormalized log posteriors `np.log(prior) + (temperature * score)`, applies numerical stability shifts (`log_arr -= np.max(log_arr)`), and uses `np.exp` to perform Softmax normalization to yield a true probability distribution.

### 1.4. Shannon Entropy (`mean_shannon_entropy`, `shannon_entropy_bits`)
*   **File Path:** `backend/core/math_engine.py` (computation) & `backend/agents/orchestrator.py` (aggregation)
*   **Function:** `ground_drug_entity()` and `IntakeOrchestrator.process_intake()`
*   **Truth Check: ✅ REAL.**
    Entropy is dynamically computed directly on the Bayesian posterior distribution using the information theory formula: `H(D) = -sum(P * np.log2(P))`. The Orchestrator correctly accumulates these individual bit measurements and calculates a true arithmetic mean (`mean_shannon_entropy`).

### 1.5. Cardiothoracic Ratio (CTR) (`ctr`)
*   **File Path:** `backend/core/math_engine.py`
*   **Function:** `calculate_cardiothoracic_ratio()`
*   **Truth Check: ✅ REAL.**
    The logic extracts bounding box dimensions using `np.where(mask > 0)` to find the minimum and maximum X coordinates of the heart and thorax binary masks (provided via SAM 3.1 or a fallback morphological CV mask) and performs the ratio division.

### 1.6. 2D Principal Component Analysis (Eigen Geometry) (`lesion_geometry`)
*   **File Path:** `backend/core/math_engine.py`
*   **Function:** `analyze_mask_eigen_geometry()`
*   **Truth Check: ✅ REAL.**
    Extracts geometric pixel coordinates from a binary mask, centers them around a calculated centroid, computes a 2x2 covariance matrix via `np.cov()`, and solves the symmetric eigendecomposition via `np.linalg.eigh()`. It dynamically yields real eigenvalues and eccentricity.

### 1.7. Bayesian Polypharmacy Logic (`polypharmacy_intervention`)
*   **File Path:** `backend/core/meta_agent.py`
*   **Function:** `escalate_polypharmacy_wonder()`
*   **Truth Check: ❌ FAKED / HARDCODED (LLM-Delegated / Mocked).**
    There is no actual mathematical Bayesian or graph-based logic running to compute drug interactions or toxicity on the backend CPU. The system delegates this entirely by prompting an LLM (Meta/Baseten agent). If the API fails or is unconfigured, it immediately returns `DEFAULT_POLYPHARMACY_FIXTURE` (a hardcoded warning about Ketorolac and Aceclofenac) to simply pass tests.


---

## 2. In-Memory Vector Cosine Similarity (Without a DB)

Since a dedicated vector database like Supabase/Pinecone is not yet integrated, the application uses an elegant in-memory technique to generate and query vector embeddings natively using Python and Numpy.

### Generation Mechanism (`text_to_embedding`)
1. **Normalization & Phonetics:** The input string is converted into a phonetic skeleton mapping (translating Bangla unicode and normalizing English).
2. **N-Gram Hashing:** The system creates character 2-grams and 3-grams of both the original string and the phonetic string.
3. **Deterministic Projection:** For every token, `hashlib.md5` is used to get a deterministic hash digest.
   - The first two bytes of the hash define the index `modulo 512` (the vector dimension).
   - The parity of the third byte dictates the sign (+1.0 or -1.0).
4. **L2 Normalization:** The resulting 512-dimensional array is L2 normalized using `numpy.linalg.norm()` so that all vectors lie on a unit hypersphere.

### Comparison Mechanism (`compute_cosine_similarity`)
Because all generated vectors are L2-normalized unit vectors initialized during application startup (in `__init__` for `dims_drug_db.json` items), comparing them is mathematically lightweight.
The system simply runs a linear scan across the arrays and calls `np.dot(vec_a, vec_b) / (norm_a * norm_b)` to find the closest match mathematically. This avoids the overhead of a dedicated vector database while preserving dense mathematical similarity searching on the CPU.
