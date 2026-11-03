"""Regression tests for the trusted-URL policy layer and integration path.

These tests verify the false-positive fix for legitimate YouTube URLs while
guaranteeing lookalikes, IP hosts, and deceptive structures are never
approved, and that phishing detection is unchanged.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.url_policy import evaluate_trusted_url_policy

client = TestClient(app)

# The exact URL class that triggered the false positive report.
BUG_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLU6SqdYcYsfJz9FAzbgocIjlkw4NXAar-"


def _api_predict(url: str) -> dict:
    response = client.post("/api/predict", json={"url": url})
    assert response.status_code == 200, response.text
    return response.json()


def test_reported_youtube_playlist_false_positive_now_safe():
    result = _api_predict(BUG_PLAYLIST_URL)
    assert result["risk_level"] == "SAFE"
    assert result["prediction"] == 0
    assert result["prediction_label"] == "benign"
    assert result["policy"]["approved"] is True
    assert result["policy"]["adjusted"] is True
    assert result["policy"]["decision"] == "policy_approved_safe"
    # Raw model output must stay visible and unchanged.
    assert result["model_score"] > 0.5
    assert result["raw_prediction"] == 1
    assert result["raw_risk_level"] in ("SUSPICIOUS", "HIGH RISK")
    assert result["risk_score"] is None
    # Explanation must mention the trusted structure.
    assert any("Trusted-platform URL structure verified" in reason for reason in result["reasons"])


# ---------------------------------------------------------------------------
# Legitimate YouTube URL variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/",
        "https://youtube.com",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLU6SqdYcYsfJz9FAzbgocIjlkw4NXAar-",
        "https://www.youtube.com/playlist?list=OLAK5uy_kABCdefGHIjklMNOpqrSTUvwxYZabcdefg",
        "https://music.youtube.com/playlist?list=OLAK5uy_kABCdefGHIjklMNOpqrSTUvwxYZabcdefg",
        "https://www.youtube.com/shorts/aBc123Déf_9".replace("é", "e"),
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=90",
    ],
)
def test_legitimate_youtube_urls_approved_safe(url: str):
    result = _api_predict(url)
    assert result["risk_level"] == "SAFE", result
    assert result["prediction"] == 0
    assert result["policy"]["approved"] is True


def test_legitimate_watch_url_evaluated_separately():
    """Watch URL: approved independently from playlist URLs."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = _api_predict(url)
    assert result["policy"]["url_kind"] == "watch"
    assert result["risk_level"] == "SAFE"


def test_raw_score_preserved_and_distinct_from_final():
    raw = _api_predict(BUG_PLAYLIST_URL)
    assert round(raw["model_score"], 4) == round(raw["raw_risk_score"] / 100.0, 4)
    assert raw["raw_prediction"] == 1 and raw["prediction"] == 0
    assert raw["raw_risk_level"] != raw["risk_level"]


# ---------------------------------------------------------------------------
# Lookalikes, credential tricks, non-HTTPS: never approved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com.evil-example.com/playlist?list=PLAB",
        "https://youtube-login.example.com/playlist?list=PLAB",
        "https://www.youtube.com.evil-example.com/playlist?list=PLAB",
        "https://members.youtube.com/playlist?list=PLAB",  # unlisted subdomain
        "http://www.youtube.com/playlist?list=PLU6SqdYcYsfJz9FAzbgocIjlkw4NXAar-",  # no HTTPS
        "https://user:pass@www.youtube.com/playlist?list=PLAB",  # credentials
        "https://www.youtube.com:8080/playlist?list=PLAB",  # odd port
        "http://192.0.2.123/playlist?list=PLAB",  # IP host
        "https://www.youtube.com/redirect?q=https://evil.example.com",  # redirect
        "https://www.youtube.com/playlist?list=https://evil.example.com",  # param
        "https://www.youtube.com/playlist?list=.INVALID%2F..",  # encoded path chars
        "https://www.youtube.com/watch?t=19999",  # watch with no v param
        "https://www.youtube.com/playlist?list=PLAB&token=deadbeef",  # extra param
        "https://www.youtube.com/results?search_query=free+crypto",  # results page
    ],
)
def test_suspicious_or_nonstandard_structures_never_approved(url: str):
    result = _api_predict(url)
    assert result["policy"]["approved"] is not True, result
    assert result["risk_level"] != "SAFE" or result["policy"]["adjusted"] is False


def test_lookalike_host_policy_not_even_applied():
    policy = evaluate_trusted_url_policy(
        "https://youtube.com.evil-example.com/playlist?list=PLAB"
    )
    assert policy.platform is None


def test_http_youtube_rejected_with_reason():
    policy = evaluate_trusted_url_policy("http://www.youtube.com/playlist?list=PLAB")
    assert policy.platform == "youtube"
    assert policy.approved is False
    assert policy.rejection == "https_required"


def test_redirect_endpoint_not_approved():
    policy = evaluate_trusted_url_policy(
        "https://www.youtube.com/redirect?q=https://evil.example.com"
    )
    assert policy.approved is False
    assert policy.rejection == "unrecognized_path"


def test_deceptive_playlist_value_not_approved():
    policy = evaluate_trusted_url_policy(
        "https://www.youtube.com/playlist?list=https%3A%2F%2Fevil.example.com"
    )
    assert policy.approved is False


# ---------------------------------------------------------------------------
# Original phishing / malware detection unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://secure-login.example/verify/account",
        "http://192.0.2.123/login/verify/password",
        "http://account-security-update.example/confirm/password",
        "http://bank-login-verification.example/account/secure",
    ],
)
def test_phishing_detection_unchanged(url: str):
    result = _api_predict(url)
    assert result["risk_level"] == "HIGH RISK"
    assert result["prediction"] == 1
    assert result["raw_prediction"] == 1
    assert result["policy"]["approved"] is False
    assert result["policy"]["decision"] == "model_verdict"
