import base64
import os
from typing import Any, Dict
import modal

MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "cloud-worker")

modal_app = modal.App(MODAL_APP_NAME)
modal_cv_image = modal.Image.debian_slim().pip_install(
    "opencv-python-headless", "numpy"
)


@modal_app.function(image=modal_cv_image, timeout=120)
def modal_remote_cv_deskew(image_bytes: bytes) -> Dict[str, Any]:
  """Executes CLAHE illumination normalization on Modal's serverless cloud."""
  import cv2
  import numpy as np

  nparr = np.frombuffer(image_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  if img is None:
    return {"status": "error", "message": "Invalid image buffer"}

  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l_channel, a_channel, b_channel = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
  enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
  _, buffer = cv2.imencode(".jpg", enhanced)

  return {
      "status": "success",
      "worker": "Modal Serverless Cloud",
      "height": int(img.shape[0]),
      "width": int(img.shape[1]),
      "enhanced_base64": base64.b64encode(buffer).decode("utf-8"),
  }