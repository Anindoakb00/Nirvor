import os
from dotenv import load_dotenv

load_dotenv()

class AIConfig:
    """Centralized configuration loaded from backend/.env"""

    BASETEN_API_KEY = os.getenv("BASETEN_API_KEY", "")
    BASETEN_BASE_URL = os.getenv(
        "BASETEN_BASE_URL", "https://inference.baseten.co/v1"
    )
    BASETEN_MODEL = os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")

    META_API_KEY = os.getenv("META_API_KEY", "")
    META_BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
    META_SPARK_MODEL = os.getenv(
        "META_SPARK_MODEL", "muse-spark-1.3-contributor"
    )
    META_WONDER_MODEL = os.getenv("META_WONDER_MODEL", "muse-wonder")
    META_VOICE_MODEL = os.getenv("META_VOICE_MODEL", "muse-voice-transcribe")
    META_IMAGE_MODEL = os.getenv("META_IMAGE_MODEL", "muse-image-1.0")
    META_SAM_MODEL = os.getenv("META_SAM_MODEL", "sam-3.1")

    USE_MODAL_CLOUD_CV = os.getenv("USE_MODAL_CLOUD_CV", "false").lower() == "true"
