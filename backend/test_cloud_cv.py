import modal

# Define the Modal app
app = modal.App("sonakto-cloud-test")

# Define a cloud container image that has OpenCV and NumPy installed
image = modal.Image.debian_slim().pip_install("opencv-python-headless", "numpy")


@app.function(image=image)
def process_prescription_mock(image_bytes: bytes) -> dict:
  """This runs entirely on Modal's cloud servers, not your local laptop!"""
  import cv2
  import numpy as np

  # Decode image bytes using OpenCV on the cloud
  nparr = np.frombuffer(image_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  # Get image dimensions
  h, w, c = img.shape

  return {
      "status": "success",
      "cloud_worker": "Modal Serverless T4/CPU",
      "image_height": h,
      "image_width": w,
      "channels": c,
  }


@app.local_entrypoint()
def main():
  import cv2
  import numpy as np

  # Create a dummy image locally
  dummy_img = np.full((400, 600, 3), 255, dtype=np.uint8)
  _, encoded_img = cv2.imencode(".png", dummy_img)

  print("Sending image bytes to Modal cloud worker...")
  result = process_prescription_mock.remote(encoded_img.tobytes())
  print("Received response from cloud:", result)