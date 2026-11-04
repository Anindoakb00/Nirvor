import requests
from PIL import Image
import io

img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

url = "http://127.0.0.1:8000/api/v2/intake-pipeline"
files = {
    "prescription": ("presc.jpg", img_byte_arr, "image/jpeg")
}
try:
    response = requests.post(url, files=files)
    print("Status:", response.status_code)
    print("JSON:", response.json())
except Exception as e:
    print("Error:", e)
