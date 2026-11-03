import re
from typing import Any, Dict, List, Union

class PrivacyShield:
    """CareShield Architecture: Scrubs PII from data."""

    # Regex patterns for common PII
    PHONE_REGEX = re.compile(r'\b(?:\+88)?01[3-9]\d{8}\b') # BD phone number format
    ID_REGEX = re.compile(r'\b\d{10,17}\b') # NID, Passport, etc.
    # Simple name scrubbing: looking for Mr., Ms., Mrs. followed by capitalized words
    NAME_REGEX = re.compile(r'\b(Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)\b')

    @classmethod
    def scrub_text(cls, text: str) -> str:
        if not isinstance(text, str):
            return text

        text = cls.PHONE_REGEX.sub('[PHONE_REDACTED]', text)
        text = cls.ID_REGEX.sub('[ID_REDACTED]', text)
        text = cls.NAME_REGEX.sub(r'\1 [NAME_REDACTED]', text)
        return text

    @classmethod
    def scrub_data(cls, data: Union[Dict, List, str, Any]) -> Union[Dict, List, str, Any]:
        if isinstance(data, str):
            return cls.scrub_text(data)
        elif isinstance(data, dict):
            return {k: cls.scrub_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.scrub_data(item) for item in data]
        else:
            return data
