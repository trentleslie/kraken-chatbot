from unittest.mock import patch
from fastapi.testclient import TestClient
from kestrel_backend.main import app


def test_validate_key_ok():
    with patch("kestrel_backend.main._probe_anthropic_key", return_value=(True, None)):
        r = TestClient(app).post("/api/validate-key", json={"key": "sk-good"})
        assert r.status_code == 200 and r.json()["valid"] is True


def test_validate_key_bad():
    with patch("kestrel_backend.main._probe_anthropic_key",
               return_value=(False, "invalid_api_key")):
        r = TestClient(app).post("/api/validate-key", json={"key": "sk-bad"})
        assert r.json() == {"valid": False, "reason": "invalid_api_key"}
