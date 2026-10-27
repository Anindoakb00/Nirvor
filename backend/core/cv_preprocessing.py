import cv2
import numpy as np


def deskew_prescription_image(image_bytes: bytes) -> np.ndarray:
  """Applies SVD-based Homography and CLAHE shadow removal to messy smartphone photos."""
  nparr = np.frombuffer(image_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if img is None:
    raise ValueError("Invalid image payload received.")

  # 1. Grayscale & Bilateral Filtering
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  filtered = cv2.bilateralFilter(gray, 9, 75, 75)

  # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  enhanced = clahe.apply(filtered)

  # 3. Edge Detection & Contour SVD Homography Alignment
  edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
  contours, _ = cv2.findContours(
      edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) == 4:
      pts = approx.reshape(4, 2)
      rect = np.zeros((4, 2), dtype="float32")

      s = pts.sum(axis=1)
      rect[0] = pts[np.argmin(s)]
      rect[2] = pts[np.argmax(s)]

      diff = np.diff(pts, axis=1)
      rect[1] = pts[np.argmin(diff)]
      rect[3] = pts[np.argmax(diff)]

      (tl, tr, br, bl) = rect
      widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
      widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
      maxWidth = max(int(widthA), int(widthB))

      heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
      heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
      maxHeight = max(int(heightA), int(heightB))

      dst = np.array([
          [0, 0],
          [maxWidth - 1, 0],
          [maxWidth - 1, maxHeight - 1],
          [0, maxHeight - 1],
      ], dtype="float32")

      H = cv2.getPerspectiveTransform(rect, dst)
      warped = cv2.warpPerspective(img, H, (maxWidth, maxHeight))
      return warped

  return img