import base64
from typing import Any, Dict, Tuple
import cv2
import numpy as np


class ClinicalCVPreprocessor:
  """Classical Computer Vision & Linear Algebra Pre-Processor for Sonakto v2.0.

  Implements:
    1. Retinex Morphological Illumination Division (Shadow Removal)
    2. Explicit Hartley-Normalized Direct Linear Transform (DLT) via SVD for
    Homography
    3. Probabilistic Hough Line Skew Correction Fallback
    4. CLAHE Contrast Enhancement & Base64 Multimodal Encoding
  """

  @staticmethod
  def _normalize_points_hartley(
      pts: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Hartley Isotropic Normalization to 2D coordinates prior to SVD.

    Shifts centroid to (0,0) and scales mean distance to sqrt(2) to prevent
    numerical ill-conditioning in the 8x9 DLT matrix A.
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-8)

    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pts_hom = np.column_stack([pts, np.ones(len(pts))])
    norm_pts_hom = (T @ pts_hom.T).T
    return norm_pts_hom[:, :2], T

  @classmethod
  def compute_svd_homography(
      cls, src_pts: np.ndarray, dst_pts: np.ndarray
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solves A * h = 0 explicitly using Singular Value Decomposition (A = U * Sigma * V^T)

    with Hartley normalization.
    """
    norm_src, T_src = cls._normalize_points_hartley(
        src_pts.astype(np.float64)
    )
    norm_dst, T_dst = cls._normalize_points_hartley(
        dst_pts.astype(np.float64)
    )

    A = np.zeros((8, 9), dtype=np.float64)
    for i in range(4):
      x, y = norm_src[i]
      xp, yp = norm_dst[i]
      A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp]
      A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp]

    # Perform full SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    # The solution h is the last row of V^T (null-space vector corresponding to smallest singular value)
    h_norm = Vt[-1, :]
    H_tilde = h_norm.reshape((3, 3))

    # De-normalize: H = T_dst^-1 * H_tilde * T_src
    H = np.linalg.inv(T_dst) @ H_tilde @ T_src
    if abs(H[2, 2]) > 1e-12:
      H = H / H[2, 2]

    # Compute SVD conditioning telemetry for Judge HUD
    sigma_ratio = float(S[-2] / max(S[-1], 1e-12))
    frobenius_norm = float(np.linalg.norm(H, ord="fro"))

    svd_telemetry = {
        "method": "Explicit_Hartley_DLT_SVD",
        "singular_values": [round(float(val), 5) for val in S],
        "nullspace_conditioning_ratio": round(sigma_ratio, 3),
        "homography_frobenius_norm": round(frobenius_norm, 4),
        "homography_matrix": np.round(H, 4).tolist(),
    }
    return H, svd_telemetry

  @staticmethod
  def remove_illumination_shadows(gray_img: np.ndarray) -> np.ndarray:
    """Estimates the low-frequency illumination field L(x,y) via morphological closing

    and extracts high-frequency ink reflectance R(x,y) = (I / L) * 255.
    """
    kernel_size = max(25, int(min(gray_img.shape[:2]) * 0.05) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background_illumination = cv2.morphologyEx(
        gray_img, cv2.MORPH_CLOSE, kernel
    )
    background_illumination = cv2.GaussianBlur(
        background_illumination, (0, 0), sigmaX=15
    )

    # Prevent division by zero
    bg_float = np.maximum(
        background_illumination.astype(np.float32), 1.0
    )
    reflectance = (gray_img.astype(np.float32) / bg_float) * 255.0
    reflectance = np.clip(reflectance, 0, 255).astype(np.uint8)

    # Apply CLAHE to sharpen faint ballpoint/fountain pen strokes
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(reflectance)

  @staticmethod
  def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 contour points strictly as: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

  @staticmethod
  def _hough_deskew_fallback(
      img: np.ndarray, enhanced_gray: np.ndarray
  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fallback rotational deskewing when all 4 outer paper corners are not visible.

    Uses Probabilistic Hough Line Transform on handwriting text lines to compute
    skew angle theta.
    """
    edges = cv2.Canny(enhanced_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=int(img.shape[1] * 0.15),
        maxLineGap=20,
    )

    angles = []
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Filter for near-horizontal text lines (-30 deg to +30 deg)
        if -30.0 <= angle_deg <= 30.0:
          angles.append(angle_deg)

    median_angle = float(np.median(angles)) if angles else 0.0
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    rot_matrix = cv2.getRotationMatrix2D(
        center, median_angle, 1.0
    )
    rotated = cv2.warpAffine(
        img,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated, {
        "method": "Hough_SE2_Affine_Rotation_Fallback",
        "detected_skew_angle_deg": round(median_angle, 3),
        "sampled_text_lines": len(angles),
        "rotation_matrix_2x3": np.round(rot_matrix, 4).tolist(),
    }

  @classmethod
  def process_prescription_image(
      cls, image_bytes: bytes
  ) -> Dict[str, Any]:
    """Full end-to-end preprocessing pipeline.

    Returns processed numpy array, Base64 PNG data URI for Meta Muse Spark 1.3,
    and full mathematical telemetry for the Judge HUD.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
      raise ValueError("Invalid or corrupted image byte stream.")

    orig_h, orig_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Retinex Illumination Division & CLAHE
    reflectance_gray = cls.remove_illumination_shadows(gray)

    # Step 2: Detect if 4 outer paper corners exist (occupying >= 25% of frame area)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    warped_color = None
    geo_telemetry = {}

    for c in contours:
      area = cv2.contourArea(c)
      if area < 0.25 * (orig_h * orig_w):
        break
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4:
        src_rect = cls._order_quad_points(approx.reshape(4, 2))
        (tl, tr, br, bl) = src_rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB), 100)

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB), 100)

        dst_rect = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype=np.float32,
        )

        # Solve Homography explicitly via Hartley-normalized SVD
        H, geo_telemetry = cls.compute_svd_homography(
            src_rect, dst_rect
        )
        warped_color = cv2.warpPerspective(
            img, H, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC
        )
        break

    # Step 3: If 4 paper corners weren't isolated, run Hough Line Skew Correction
    if warped_color is None:
      warped_color, geo_telemetry = cls._hough_deskew_fallback(
          img, reflectance_gray
      )

    # Step 4: Blend illumination-normalized contrast into final image for crisp OCR
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    final_enhanced_gray = cls.remove_illumination_shadows(warped_gray)
    final_bgr = cv2.cvtColor(final_enhanced_gray, cv2.COLOR_GRAY2BGR)
    blended_output = cv2.addWeighted(
        warped_color, 0.65, final_bgr, 0.35, 0
    )

    # Encode to Base64 Data URI for Meta Multimodal API
    success, encoded_png = cv2.imencode(".png", blended_output)
    if not success:
      raise RuntimeError("Failed to encode preprocessed image to PNG.")
    b64_str = base64.b64encode(encoded_png.tobytes()).decode("utf-8")

    return {
        "processed_image": blended_output,
        "base64_data_uri": f"data:image/png;base64,{b64_str}",
        "telemetry": {
            "original_resolution": [orig_w, orig_h],
            "output_resolution": [
                blended_output.shape[1],
                blended_output.shape[0],
            ],
            "illumination_normalization": (
                "Retinex_Morphological_Division + CLAHE_2.2"
            ),
            "geometry": geo_telemetry,
        },
    }


# Backwards-compatible helper function
def deskew_prescription_image(image_bytes: bytes) -> Dict[str, Any]:
  return ClinicalCVPreprocessor.process_prescription_image(image_bytes)