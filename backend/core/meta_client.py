import os
import requests
from dotenv import load_dotenv

load_dotenv()
META_API_KEY = os.getenv("META_API_KEY")
BASE_URL = "https://api.meta.ai/v1"


async def query_muse_spark(prompt: str) -> dict:
  endpoint = f"{BASE_URL}/chat/completions"
  headers = {
      "Authorization": f"Bearer {META_API_KEY}",
      "Content-Type": "application/json",
  }

  payload = {
      "model": "muse-spark-1.3",
      "messages": [{"role": "user", "content": prompt}],
      "temperature": 0.1,
  }

  response = requests.post(endpoint, json=payload, headers=headers)
  if response.status_code != 200:
    raise Exception(f"Meta API Error: {response.text}")

  return response.json()