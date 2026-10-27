import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ClinicalMathEngine:
  """Neuro-Symbolic Mathematical & Biometric Engine for Sonakto v2.0.

  Implements:
    1. R^512 Character N-Gram & South Asian Phonetic Feature Hashing
    2. Cursive-Weighted Dynamic Programming Levenshtein Distance
    3. Bayesian Cross-Modal Posterior Drug Grounding (OCR + Voice Fusion)
    4. Dialect-to-ICD10 Vector Retrieval
    5. SAM 3.1 Binary Mask Linear Algebra: Cardiothoracic Ratio (CTR) & 2D
    Eigen-PCA
  """

  VECTOR_DIM = 512

  # Low-cost substitution pairs typical of rapid cursive medical handwriting & OCR slips
  CURSIVE_CONFUSION_PAIRS = {
      frozenset({"e", "c"}): 0.25,
      frozenset({"l", "i"}): 0.20,
      frozenset({"l", "1"}): 0.15,
      frozenset({"o", "0"}): 0.15,
      frozenset({"s", "5"}): 0.20,
      frozenset({"z", "2"}): 0.25,
      frozenset({"r", "n"}): 0.30,
      frozenset({"m", "n"}): 0.35,
      frozenset({"u", "v"}): 0.25,
      frozenset({"g", "q"}): 0.30,
      frozenset({"c", "k"}): 0.30,
      frozenset({"s", "c"}): 0.35,
      frozenset({"j", "g"}): 0.30,
  }

  # Bangla Unicode to phonetic Latin transliteration map for cross-modal voice/OCR alignment
  BANGLA_PHONETIC_MAP = str.maketrans({
      "অ": "a",
      "আ": "a",
      "ই": "i",
      "ঈ": "i",
      "উ": "u",
      "ঊ": "u",
      "এ": "e",
      "ঐ": "oi",
      "ও": "o",
      "ঔ": "ou",
      "া": "a",
      "ি": "i",
      "ী": "i",
      "ু": "u",
      "ূ": "u",
      "ে": "e",
      "ৈ": "oi",
      "ো": "o",
      "ৌ": "ou",
      "ক": "k",
      "খ": "kh",
      "গ": "g",
      "ঘ": "gh",
      "চ": "ch",
      "ছ": "s",
      "জ": "j",
      "ঝ": "jh",
      "ট": "t",
      "ঠ": "th",
      "ড": "d",
      "ঢ": "dh",
      "ত": "t",
      "থ": "th",
      "দ": "d",
      "ধ": "dh",
      "ন": "n",
      "ণ": "n",
      "প": "p",
      "ফ": "f",
      "ব": "b",
      "ভ": "v",
      "ম": "m",
      "য": "j",
      "র": "r",
      "ল": "l",
      "শ": "s",
      "ষ": "s",
      "স": "s",
      "হ": "h",
      "ড়": "r",
      "ঢ়": "rh",
      "য়": "y",
      "ং": "ng",
      "ঁ": "",
      "্": "",
      "০": "0",
      "১": "1",
      "২": "2",
      "৩": "3",
      "৪": "4",
      "৫": "5",
      "৬": "6",
      "৭": "7",
      "৮": "8",
      "৯": "9",
  })

  def __init__(self):
    db_path = os.path.join(
        os.path.dirname(__file__), "../database/dims_drug_db.json"
    )
    with open(db_path, "r", encoding="utf-8") as f:
      raw_db = json.load(f)

    self.drug_db = raw_db.get("drugs", [])
    self.dialect_lexicon = raw_db.get("dialect_symptom_lexicon", [])

    # Pre-compute R^512 embeddings for all verified DIMS drugs and dialect idioms
    for drug in self.drug_db:
      brand_clean = self._clean_clinical_title(drug["brand_name"])
      drug["_clean_brand"] = brand_clean
      drug["_ocr_vec"] = self.text_to_embedding(
          f"{drug['brand_name']} {drug['generic_name']}"
      )
      alias_blob = " ".join(
          [drug["brand_name"]] + drug.get("bangla_phonetic_aliases", [])
      )
      drug["_phonetic_vec"] = self.text_to_embedding(
          self.to_phonetic_skeleton(alias_blob)
      )

    for item in self.dialect_lexicon:
      combined_dialect = f"{item['dialect_phrase']} {item['phonetic_roman']}"
      item["_vec"] = self.text_to_embedding(
          self.to_phonetic_skeleton(combined_dialect)
      )

  @staticmethod
  def _clean_clinical_title(text: str) -> str:
    """Strips common dosage prefixes (Tab., Cap., Inj., Syp.) for direct brand matching."""
    cleaned = re.sub(
        r"\b(tab|cap|inj|syp|susp|drops|mg|mcg|ml)\b\.?",
        " ",
        text.lower(),
    )
    return re.sub(r"\s+", " ", cleaned).strip()

  @classmethod
  def to_phonetic_skeleton(cls, text: str) -> str:
    """Projects Bangla Unicode + English strings into a canonical South Asian phonetic skeleton."""
    s = text.lower().translate(cls.BANGLA_PHONETIC_MAP)
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"z", "j", s)
    s = re.sub(r"v", "b", s)
    s = re.sub(r"ck|q", "k", s)
    s = re.sub(r"sh|ss", "s", s)
    # Collapse consecutive duplicate characters (e.g., 'sergell' -> 'sergel')
    s = re.sub(r"(.)\1+", r"\1", s)
    return s.strip()

  @classmethod
  def text_to_embedding(cls, text: str) -> np.ndarray:
    """Projects a string into a deterministic, L2-normalized R^512 vector space

    using signed character 2-gram, 3-gram, and phonetic skeleton hashing.
    """
    vec = np.zeros(cls.VECTOR_DIM, dtype=np.float64)
    clean = f"^{text.lower().strip()}$"
    phonetic = f"^{cls.to_phonetic_skeleton(text)}$"

    features: List[Tuple[str, float]] = []
    # Character bigrams & trigrams
    for n in (2, 3):
      for i in range(max(0, len(clean) - n + 1)):
        features.append((f"c{n}_{clean[i:i+n]}", 1.0))
      for i in range(max(0, len(phonetic) - n + 1)):
        features.append((f"p{n}_{phonetic[i:i+n]}", 1.35))

    for token, weight in features:
      digest = hashlib.md5(token.encode("utf-8")).digest()
      idx = int.from_bytes(digest[:2], "little") % cls.VECTOR_DIM
      sign = 1.0 if (digest[2] & 1) == 0 else -1.0
      vec[idx] += sign * weight

    norm = np.linalg.norm(vec)
    if norm > 1e-12:
      vec /= norm
    return vec

  @staticmethod
  def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom < 1e-12:
      return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

  @classmethod
  def weighted_cursive_levenshtein(cls, s1: str, s2: str) -> float:
    """Dynamic Programming Edit Distance weighted for doctor handwriting character confusions."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    m, n = len(s1), len(s2)
    if m == 0:
      return float(n)
    if n == 0:
      return float(m)

    dp = np.zeros((m + 1, n + 1), dtype=np.float64)
    for i in range(m + 1):
      dp[i, 0] = i * 0.8
    for j in range(n + 1):
      dp[0, j] = j * 0.8

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if s1[i - 1] == s2[j - 1]:
          sub_cost = 0.0
        else:
          pair = frozenset({s1[i - 1], s2[j - 1]})
          sub_cost = cls.CURSIVE_CONFUSION_PAIRS.get(pair, 1.0)

        dp[i, j] = min(
            dp[i - 1, j] + 0.8,  # Deletion
            dp[i, j - 1] + 0.8,  # Insertion
            dp[i - 1, j - 1] + sub_cost,  # Handwriting-weighted substitution
        )

    max_len = max(m, n, 1)
    return float(dp[m, n] / max_len)

  def ground_drug_entity(
      self,
      ocr_drug_string: str,
      patient_voice_transcript: Optional[str] = None,
      alpha: float = 2.4,
      beta: float = 1.8,
      gamma: float = 3.2,
      temperature: float = 4.5,
  ) -> Dict[str, Any]:
    """Executes Bayesian Cross-Modal Fusion over the Bangladesh DIMS drug manifold.

    Combines OCR N-gram cosine similarity, Phonetic Voice alignment, and
    Cursive-Weighted Levenshtein Distance into a normalized Posterior
    Probability distribution P(D | s_ocr, s_voice).
    """
    clean_ocr = self._clean_clinical_title(ocr_drug_string)
    v_ocr = self.text_to_embedding(ocr_drug_string)
    v_voice = (
        self.text_to_embedding(
            self.to_phonetic_skeleton(patient_voice_transcript)
        )
        if patient_voice_transcript
        else None
    )

    log_unnormalized_posteriors = []
    candidate_breakdown = []

    for drug in self.drug_db:
      cos_ocr = self.compute_cosine_similarity(v_ocr, drug["_ocr_vec"])
      lev_norm = self.weighted_cursive_levenshtein(
          clean_ocr, drug["_clean_brand"]
      )

      # If patient mentioned medication in voice note, compute cross-modal phonetic boost
      cos_voice = 0.0
      if v_voice is not None:
        cos_voice = max(
            0.0,
            self.compute_cosine_similarity(v_voice, drug["_phonetic_vec"]),
        )

      # Energy score E(d_i)
      score = (
          (alpha * cos_ocr)
          + (beta * cos_voice if v_voice is not None else 0.0)
          - (gamma * lev_norm)
      )
      prior = max(float(drug.get("prior_probability", 0.05)), 1e-4)

      log_post = np.log(prior) + (temperature * score)
      log_unnormalized_posteriors.append(log_post)

      candidate_breakdown.append({
          "drug": drug,
          "cos_ocr": round(cos_ocr, 4),
          "cos_voice": round(cos_voice, 4),
          "weighted_levenshtein": round(lev_norm, 4),
          "composite_energy": round(score, 4),
      })

    # Softmax normalization for exact Bayesian Posterior P(d_i | E)
    log_arr = np.array(log_unnormalized_posteriors, dtype=np.float64)
    log_arr -= np.max(log_arr)  # Numerical stability shift
    exp_arr = np.exp(log_arr)
    posteriors = exp_arr / np.sum(exp_arr)

    # Compute Shannon Entropy H(D) = -sum P(d_i) log2 P(d_i)
    shannon_entropy = float(
        -np.sum(posteriors * np.log2(np.clip(posteriors, 1e-12, 1.0)))
    )

    # Rank candidates by posterior probability
    ranked_indices = np.argsort(posteriors)[::-1]
    top_idx = int(ranked_indices[0])
    second_idx = int(ranked_indices[1]) if len(ranked_indices) > 1 else top_idx

    best_candidate = candidate_breakdown[top_idx]
    best_drug = best_candidate["drug"]
    top_posterior = float(posteriors[top_idx])

    # Clean internal numpy vectors from output dictionary
    public_drug_view = {
        k: v for k, v in best_drug.items() if not k.startswith("_")
    }

    return {
        "raw_ocr_input": ocr_drug_string,
        "matched_drug": public_drug_view,
        "bayesian_posterior_confidence": round(top_posterior, 4),
        "shannon_entropy_bits": round(shannon_entropy, 4),
        "ocr_typo_corrected": (
            clean_ocr.lower() != best_drug["_clean_brand"].lower()
        ),
        "requires_active_verification": bool(
            shannon_entropy > 1.35 or top_posterior < 0.55
        ),
        "top_3_manifold_candidates": [
            {
                "brand_name": candidate_breakdown[int(idx)]["drug"][
                    "brand_name"
                ],
                "generic_name": candidate_breakdown[int(idx)]["drug"][
                    "generic_name"
                ],
                "posterior_prob": round(float(posteriors[int(idx)]), 4),
                "cos_ocr": candidate_breakdown[int(idx)]["cos_ocr"],
                "cos_voice": candidate_breakdown[int(idx)]["cos_voice"],
                "weighted_levenshtein": candidate_breakdown[int(idx)][
                    "weighted_levenshtein"
                ],
            }
            for idx in ranked_indices[:3]
        ],
    }

  def align_regional_dialect_symptoms(
      self, raw_transcript: str, top_k: int = 2
  ) -> List[Dict[str, Any]]:
    """Retrieves the mathematically closest ICD-10 clinical symptoms from a noisy

    Chittagonian/Sylheti transcript using Phonetic Skeleton Cosine Similarity.
    """
    if not raw_transcript:
      return []
    q_vec = self.text_to_embedding(self.to_phonetic_skeleton(raw_transcript))
    scored = []
    for item in self.dialect_lexicon:
      sim = self.compute_cosine_similarity(q_vec, item["_vec"])
      scored.append({
          "dialect_region": item["dialect_region"],
          "matched_idiom": item["dialect_phrase"],
          "clinical_translation": item["clinical_term"],
          "icd10_code": item["icd10_code"],
          "phonetic_cosine_similarity": round(sim, 4),
      })
    scored.sort(key=lambda x: x["phonetic_cosine_similarity"], reverse=True)
    return scored[:top_k]

  @staticmethod
  def calculate_cardiothoracic_ratio(
      heart_mask: np.ndarray, thorax_mask: np.ndarray
  ) -> Dict[str, Any]:
    """Calculates Cardiothoracic Ratio (CTR) from SAM 3.1 binary segmentation masks:

    CTR = (max(x_heart) - min(x_heart)) / (max(x_thorax) - min(x_thorax))
    """
    h_x = np.where(heart_mask > 0)[1]
    t_x = np.where(thorax_mask > 0)[1]

    if len(h_x) == 0 or len(t_x) == 0:
      return {"ctr": 0.0, "cardiomegaly_flagged": False}

    heart_span = int(np.max(h_x) - np.min(h_x))
    thorax_span = max(int(np.max(t_x) - np.min(t_x)), 1)
    ctr = float(heart_span / thorax_span)

    return {
        "cardiac_transverse_diameter_px": heart_span,
        "thoracic_transverse_diameter_px": thorax_span,
        "ctr": round(ctr, 4),
        "clinical_threshold": 0.50,
        "cardiomegaly_flagged": bool(ctr > 0.50),
    }

  @staticmethod
  def analyze_mask_eigen_geometry(
      binary_mask: np.ndarray, pixel_spacing_mm: float = 0.25
  ) -> Dict[str, Any]:
    """Executes 2D Principal Component Analysis (PCA) on a SAM 3.1 binary mask:

    1. Extracts foreground pixel coordinates X in R^(N x 2)
    2. Computes Covariance Matrix Sigma = (X - mu)^T (X - mu) / (N - 1)
    3. Solves Sigma * v = lambda * v via symmetric eigendecomposition
    """
    y_indices, x_indices = np.where(binary_mask > 0)
    num_pixels = len(x_indices)
    if num_pixels < 5:
      return {"valid_mask": False, "pixel_count": num_pixels}

    coords = np.column_stack([x_indices, y_indices]).astype(np.float64)
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid

    # 2x2 Covariance Matrix Sigma
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending: lambda_1 >= lambda_2
    order = np.argsort(eigenvalues)[::-1]
    lambda_1 = max(float(eigenvalues[order[0]]), 1e-8)
    lambda_2 = max(float(eigenvalues[order[1]]), 0.0)
    principal_axis = eigenvectors[:, order[0]]

    # Eccentricity e = sqrt(1 - lambda_2 / lambda_1)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (lambda_2 / lambda_1))))
    orientation_deg = float(
        np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
    )
    area_mm2 = float(num_pixels * (pixel_spacing_mm**2))

    return {
        "valid_mask": True,
        "centroid_xy": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
        "covariance_matrix_2x2": np.round(cov_matrix, 3).tolist(),
        "eigenvalues_lambda1_lambda2": [round(lambda_1, 3), round(lambda_2, 3)],
        "principal_eigenvector": [
            round(float(principal_axis[0]), 4),
            round(float(principal_axis[1]), 4),
        ],
        "eccentricity": round(eccentricity, 4),
        "orientation_angle_deg": round(orientation_deg, 2),
        "estimated_area_mm2": round(area_mm2, 2),
    }