import importlib

import config.settings as settings
from database.import_verified_lexicon import validate_row


def test_supabase_env_fallback_names(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_URL", "https://xyz.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key")
    try:
        importlib.reload(settings)
        assert settings.AIConfig.SUPABASE_URL == "https://xyz.supabase.co"
        assert settings.AIConfig.SUPABASE_KEY == "svc-key"
    finally:
        importlib.reload(settings)


def test_supabase_explicit_names_take_precedence(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://explicit.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "explicit-key")
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_URL", "https://other.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "other-key")
    try:
        importlib.reload(settings)
        assert settings.AIConfig.SUPABASE_URL == "https://explicit.supabase.co"
        assert settings.AIConfig.SUPABASE_KEY == "explicit-key"
    finally:
        importlib.reload(settings)


def _row(**overrides):
    base = {
        "audio_file": "clip.mp3",
        "transcript": "পেটে ব্যথা",
        "dialect_region": "Chittagonian",
        "dialect_phrase": "পেটে ব্যথা",
        "clinical_term": "Abdominal pain",
        "icd10_code": "R10.9",
        "verified": "TRUE",
        "notes": "",
    }
    base.update(overrides)
    return base


def test_validate_row_accepts_complete_verified_row():
    ok, reason, record = validate_row(_row())
    assert ok and reason == "ok"
    assert record["dialect_phrase"] == "পেটে ব্যথা"
    assert record["source_file"] == "clip.mp3"


def test_validate_row_accepts_loose_true():
    ok, _, _ = validate_row(_row(verified=" true "))
    assert ok


def test_validate_row_rejects_unverified_and_incomplete():
    ok, reason, _ = validate_row(_row(verified="FALSE"))
    assert not ok and "verified" in reason
    for field in ("dialect_phrase", "clinical_term", "icd10_code"):
        ok, reason, _ = validate_row(_row(**{field: "  "}))
        assert not ok and field in reason
