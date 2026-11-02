import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "R2 (Experiment C Winner)"


def test_predict_valid_url():
    response = client.post("/api/predict", json={"url": "https://www.google.com"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["valid"] is True
    assert payload["risk_level"] == "SAFE"
    assert payload["prediction_label"] == "benign"
    assert payload["model"]["variant"] == "winner_r2"


def test_predict_invalid_url_returns_400():
    response = client.post("/api/predict", json={"url": "   "})
    assert response.status_code == 422

    response = client.post("/api/predict", json={"url": "javascript:alert(1)"})
    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "non_web_scheme"


# ---------------------------------------------------------------------------
# Layered /api/scan: all four security verdicts reachable with the same
# request path the frontend uses (synthetic inspection backends).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from backend import risk_engine as eng  # noqa: E402


def _tls_ok():
    return {"handshake": "ok", "tls_version": "TLSv1.3", "cipher": "X", "subject_cn": "example.org",
            "issuer": "CA", "not_after": "Jan 01 00:00:00 2028 GMT", "hostname": "example.org"}


_FULL_PAGE = {
    "requested_url": "https://example.org/", "reachable": True, "blocked": None, "error": None,
    "availability": "REACHABLE", "title_available": True, "host_flags": [],
    "hops": [{"url": "https://example.org/", "status": 200, "ip": "1.2.3.4", "location": None}],
    "final_url": "https://example.org/", "status_code": 200, "content_type": "text/html",
    "is_html": True, "served_executable": False, "content_truncated": False, "tls": _tls_ok(),
    "dns": {"host": "example.org", "validated_ips": ["1.2.3.4"], "error": None},
    "redirect_scheme_downgrade": False,
    "page": {"title": "Welcome", "text_length": 900, "text_sample": "content", "forms": 0,
             "form_actions": [], "password_fields": 0, "sensitive_inputs": 0, "links": 3,
             "external_scripts": 0, "iframes": [], "meta_refresh": None, "http_images": 0},
    "elapsed_ms": 100.0,
}


def _run_scan(monkeypatch, inspection, model_score=0.5, raw=0):
    import backend.app as appmod

    monkeypatch.setattr(appmod, "inspect_website", lambda url: inspection)
    monkeypatch.setattr(
        appmod,
        "predict_url",
        lambda url: {"valid": True, "raw_prediction": raw, "model_score": model_score,
                     "policy": {"approved": False}, "prediction_label": "benign", "risk_level": "SAFE"},
    )
    import json as _json

    client.headers["content-type"] = "application/json"
    response = client.request(
        "POST", "/api/scan",
        content=_json.dumps({"url": "https://example.org/"}),
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_scan_api_produces_safe_verdict(monkeypatch):
    data = _run_scan(monkeypatch, dict(_FULL_PAGE), raw=0)
    assert data["verdict"] == eng.VERDICT_SAFE
    assert data["availability"] == eng.AVAILABILITY_REACHABLE
    assert data["signals"]["r2_raw_decision"] == 0


def test_scan_api_produces_malicious_verdict_on_hw(monkeypatch):
    phishing = dict(_FULL_PAGE)
    phishing["tls"] = None  # password form over unencrypted transport
    phishing["page"] = dict(_FULL_PAGE["page"], password_fields=1, title="Sign in")
    data = _run_scan(monkeypatch, phishing, raw=1)
    assert data["verdict"] == eng.VERDICT_MALICIOUS
    assert any(f["code"] == "password_over_http" for f in data["findings"])


def test_scan_api_verdicts_are_consistent_with_engine(monkeypatch):
    """Direct property: the API response equals the engine's decision for
    the same inputs (no extra filtering between layers)."""
    data = _run_scan(monkeypatch, dict(_FULL_PAGE), raw=1)
    assessment = eng.decide(
        {"raw_prediction": 1, "model_score": 0.5, "policy": {"approved": False}},
        dict(_FULL_PAGE),
    )
    assert data["verdict"] == assessment["verdict"]
    assert data["availability"] == assessment["availability"]
    assert data["findings"] == assessment["findings"]


def test_scan_api_invalid_url_422():
    import json as _json

    client.headers["content-type"] = "application/json"
    response = client.request("POST", "/api/scan", content=_json.dumps({"url": "javascript:alert(1)"}))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "non_web_scheme"
