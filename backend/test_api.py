import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_system_status():
    response = client.get("/api/v2/system-status")
    assert response.status_code == 200
    data = response.json()
    assert "meta_configured" in data
    assert "baseten_configured" in data
    assert "modal_configured" in data
    assert data["version"] == "0.0.3"

@pytest.mark.asyncio
async def test_intake_voice_empty():
    response = client.post("/api/v2/intake/voice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

@pytest.mark.asyncio
async def test_intake_prescription_empty():
    response = client.post("/api/v2/intake/prescription")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

@pytest.mark.asyncio
async def test_intake_xray_empty():
    response = client.post("/api/v2/intake/xray")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
