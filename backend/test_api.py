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
async def test_intake_pipeline_demo_mode():
    response = client.post("/api/v2/intake-pipeline", data={"demo_mode": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "cv_telemetry" in data
    assert "clinical_coherence_report" in data
    assert "agent_execution_trace" in data

    trace = data["agent_execution_trace"]
    assert len(trace) > 0
    assert trace[0]["step_name"] == "clinical_cv_preprocessor"
