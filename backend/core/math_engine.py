import json
import os
import numpy as np


class ClinicalMathEngine:

  def __init__(self):
    db_path = os.path.join(
        os.path.dirname(__file__), "../database/dims_drug_db.json"
    )
    with open(db_path, "r", encoding="utf-8") as f:
      self.drug_db = json.load(f)["drugs"]

  def compute_cosine_similarity(
      self, vector_a: list[float], vector_b: list[float]
  ) -> float:
    """Computes cosine similarity between two embedding vectors."""
    a = np.array(vector_a)
    b = np.array(vector_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
      return 0.0
    return float(dot_product / (norm_a * norm_b))

  def ground_drug_entity(self, extracted_embedding: list[float]) -> dict:
    """Projects noisy OCR strings onto verified DIMS database vectors to eliminate hallucination."""
    best_match = None
    highest_score = -1.0

    for drug in self.drug_db:
      score = self.compute_cosine_similarity(
          extracted_embedding, drug["embedding_vector"]
      )
      if score > highest_score:
        highest_score = score
        best_match = drug

    return {
        "matched_drug": best_match,
        "confidence_score": highest_score,
        "hallucination_intercepted": highest_score < 0.75,
    }

  def calculate_cardiothoracic_ratio(
      self, heart_mask: np.ndarray, thorax_mask: np.ndarray
  ) -> float:
    """Calculates Cardiothoracic Ratio (CTR) from SAM 3.1 binary masks using horizontal extremes."""
    h_indices = np.where(heart_mask > 0)[1]
    t_indices = np.where(thorax_mask > 0)[1]

    if len(h_indices) == 0 or len(t_indices) == 0:
      return 0.0

    heart_width = np.max(h_indices) - np.min(h_indices)
    thorax_width = np.max(t_indices) - np.min(t_indices)

    ctr = float(heart_width / thorax_width)
    return round(ctr, 4)