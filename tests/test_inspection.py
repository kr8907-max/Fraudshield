"""
Tests for the layered website inspection engine.

- Deterministic safety tests (SSRF blocking, input refusal) run offline.
- Engine unit tests use synthetic inspection results.
- Real-site integration tests run only when the network is available.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import risk_engine as eng
from backend.inspector import _ip_is_unsafe, extract_brand_claim, inspect_website
from backend.url_policy import evaluate_trusted_url_policy
import ipaddress


def _network_available() -> bool:
    import socket

    try:
        socket.gethostbyname("github.com")
        return True
    except OSError:
        return False


def _model(raw_prediction: int = 1, score: float = 0.9) -> dict:
    return {
        "raw_prediction": raw_prediction,
        "model_score": score,
        "policy": {"approved": False},
    }


def _base_inspection(**overrides) -> dict:
    base = {
        "requested_url": "https://example.org/",
        "reachable": True,
        "blocked": None,
        "error": None,
        "availability": "REACHABLE",
        "title_available": True,
        "hops": [{"url": "https://example.org/", "status": 200, "ip": "93.184.216.34", "location": None}],
        "final_url": "https://example.org/",
        "status_code": 200,
        "content_type": "text/html",
        "is_html": True,
        "served_executable": False,
        "content_truncated": False,
        "tls": {
            "handshake": "ok",
            "tls_version": "TLSv1.3",
            "cipher": "TLS_AES_128_GCM_SHA256",
            "subject_cn": "example.org",
            "issuer": "Some CA",
            "not_after": "Jan 01 00:00:00 2028 GMT",
            "hostname": "example.org",
        },
        "dns": {"host": "example.org", "validated_ips": ["93.184.216.34"], "error": None},
        "redirect_scheme_downgrade": False,
        "page": {
            "title": "Example Page",
            "text_length": 900,
            "text_sample": "hello world",
            "forms": 0,
            "form_actions": [],
            "password_fields": 0,
            "sensitive_inputs": 0,
            "links": 3,
            "external_scripts": 0,
            "iframes": [],
            "meta_refresh": None,
            "http_images": 0,
        },
        "elapsed_ms": 100.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Deterministic SSRF / destination validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip_value",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.4.9",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.3.7",       # CGNAT
        "198.18.0.9",       # benchmark
        "::1",
        "fe80::1",
        "fc00::5",
    ],
)
def test_unsafe_ip_ranges_blocked(ip_value: str):
    assert _ip_is_unsafe(ipaddress.ip_address(ip_value)) is True


def test_public_ip_allowed():
    assert _ip_is_unsafe(ipaddress.ip_address("93.184.216.34")) is False


def test_inspector_blocks_localhost_without_contact():
    result = inspect_website("http://localhost:8000/")
    assert result["reachable"] is False
    assert result["blocked"] is not None
    assert "unsafe_destination" in result["blocked"]


def test_engine_unverifiable_on_ssrf_attempt():
    assessment = eng.decide(
        _model(raw_prediction=1),
        _base_inspection(blocked="unsafe_destination:169.254.169.254", reachable=False),
    )
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT
    assert any("forbidden" in line or "Refused" in line for line in assessment["evidence"])


def test_private_dns_name_never_contacted():
    result = inspect_website("http://10.0.0.5/admin")
    assert result["blocked"] == "unsafe_destination:10.0.0.5"


# ---------------------------------------------------------------------------
# Engine unit tests (synthetic inspection data)
# ---------------------------------------------------------------------------


def test_safe_on_full_benign_inspection():
    assessment = eng.decide(_model(raw_prediction=1), _base_inspection())
    assert assessment["verdict"] == eng.VERDICT_SAFE
    assert any("overridden" in line for line in assessment["evidence"])
    assert assessment["signals"]["r2_raw_decision"] == 1  # preserved, not decisive


def test_password_form_over_http_is_high_risk():
    page = {"password_fields": 1, "sensitive_inputs": 0, "title": "Sign in"}
    inspection = _base_inspection(tls=None, page=page)
    assessment = eng.decide(_model(raw_prediction=0, score=0.1), inspection)
    assert assessment["verdict"] == eng.VERDICT_MALICIOUS
    assert any(f["code"] == "password_over_http" for f in assessment["findings"])


def test_brand_impersonation_is_high_risk():
    page = {"password_fields": 0, "sensitive_inputs": 0, "title": "PayPal - Verify Your Account"}
    inspection = _base_inspection(page=page, final_url="https://secure-paypal4321.example/login")
    inspection["dns"]["host"] = "secure-paypal4321.example"
    assessment = eng.decide(_model(raw_prediction=1), inspection)
    assert assessment["verdict"] == eng.VERDICT_MALICIOUS
    assert any(f["code"] == "brand_impersonation" for f in assessment["findings"])


def test_executable_served_is_high_risk():
    inspection = _base_inspection(served_executable=True, is_html=False, page={})
    assessment = eng.decide(_model(raw_prediction=0, score=0.2), inspection)
    assert assessment["verdict"] == eng.VERDICT_MALICIOUS
    assert any(f["code"] == "executable_download" for f in assessment["findings"])


def test_http_entry_is_capped_below_safe():
    inspection = _base_inspection(tls=None, final_url="http://example.org/")
    inspection["hops"][0]["url"] = "http://example.org/"
    assessment = eng.decide(_model(raw_prediction=0, score=0.2), inspection)
    # Unencrypted transport can never be SAFE; benign content stays below HIGH.
    assert assessment["verdict"] == eng.VERDICT_SUSPICIOUS
    assert any(f["code"] == "no_https_entry" for f in assessment["findings"])


def test_unreachable_site_is_unverified():
    inspection = _base_inspection(reachable=False, error="dns_resolution_failed", tls=None, page={})
    assessment = eng.decide(_model(raw_prediction=0, score=0.1), inspection)
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT
    assert not any(line.startswith("Verdict SAFE") for line in assessment["evidence"])


def test_medium_finding_is_suspicious():
    page = _base_inspection()["page"]
    page["meta_refresh"] = "0; url=https://phishy.example.ru/next"
    inspection = _base_inspection(page=page, final_url="https://could-not-be-anything-else.org/")
    assessment = eng.decide(_model(raw_prediction=0, score=0.2), inspection)
    assert assessment["verdict"] == eng.VERDICT_SUSPICIOUS


# ---------------------------------------------------------------------------
# Availability / security separation regression (10 required cases)
# Security classification must never depend on title/body success, and
# availability must come only from transport/status facts.
# ---------------------------------------------------------------------------

_EMPTY_PAGE = {
    "title": None,
    "text_length": 0,
    "text_sample": "",
    "forms": 0,
    "form_actions": [],
    "password_fields": 0,
    "sensitive_inputs": 0,
    "links": 0,
    "external_scripts": 0,
    "iframes": [],
    "meta_refresh": None,
    "http_images": 0,
}


def test_regression_case_normal_200_webpage():
    assessment = eng.decide(
        _model(raw_prediction=0, score=0.05), _base_inspection()
    )
    assert assessment["verdict"] == eng.VERDICT_SAFE
    assert assessment["availability"] == eng.AVAILABILITY_REACHABLE


def test_regression_case_200_without_title():
    inspection = _base_inspection(
        title_available=False,
        availability=eng.AVAILABILITY_TITLE_UNAVAILABLE,
        page=dict(_EMPTY_PAGE),
    )
    # The URL-feature signal still classifies: R2 flag -> SUSPICIOUS.
    flagged = eng.decide(_model(raw_prediction=1, score=0.9), inspection)
    assert flagged["verdict"] == eng.VERDICT_SUSPICIOUS
    assert flagged["availability"] == eng.AVAILABILITY_TITLE_UNAVAILABLE
    # Without the model flag there is no evidence: honest INSUFFICIENT,
    # never a false SAFE.
    benign = eng.decide(_model(raw_prediction=0, score=0.05), inspection)
    assert benign["verdict"] == eng.VERDICT_INSUFFICIENT


def test_regression_case_404_page():
    inspection = _base_inspection(
        status_code=404,
        error="http_status_404",
        is_html=False,
        title_available=False,
        availability=eng.AVAILABILITY_PAGE_NOT_FOUND,
        page=dict(_EMPTY_PAGE),
    )
    # 404 must not stop URL-feature-based classification.
    flagged = eng.decide(_model(raw_prediction=1, score=0.95), inspection)
    assert flagged["verdict"] == eng.VERDICT_SUSPICIOUS
    assert flagged["availability"] == eng.AVAILABILITY_PAGE_NOT_FOUND
    benign = eng.decide(_model(raw_prediction=0, score=0.05), inspection)
    assert benign["verdict"] == eng.VERDICT_INSUFFICIENT
    assert benign["verdict"] != eng.VERDICT_SAFE  # failure never implies safe


def test_regression_case_500_page():
    inspection = _base_inspection(
        status_code=500,
        error="http_status_500",
        is_html=False,
        title_available=False,
        availability=eng.AVAILABILITY_TITLE_UNAVAILABLE,
        page=dict(_EMPTY_PAGE),
    )
    assessment = eng.decide(_model(raw_prediction=0, score=0.1), inspection)
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT
    assert assessment["availability"] == eng.AVAILABILITY_TITLE_UNAVAILABLE
    flagged = eng.decide(_model(raw_prediction=1, score=0.9), inspection)
    assert flagged["verdict"] == eng.VERDICT_SUSPICIOUS


def test_regression_case_slow_response():
    # Slow but successful: response completes, classification unaffected.
    slow_complete = _base_inspection()
    slow_complete["elapsed_ms"] = 9500.0
    assessment = eng.decide(_model(raw_prediction=0, score=0.05), slow_complete)
    assert assessment["verdict"] == eng.VERDICT_SAFE
    assert assessment["availability"] == eng.AVAILABILITY_REACHABLE


def test_regression_case_connection_timeout():
    inspection = _base_inspection(
        reachable=False,
        error="connection_timeout",
        tls=None,
        page={},
        availability=eng.AVAILABILITY_TIMEOUT,
        title_available=False,
        dns={"host": "example.org", "validated_ips": [], "error": None},
        hops=[],
        final_url=None,
    )
    assessment = eng.decide(_model(raw_prediction=0, score=0.1), inspection)
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT
    assert assessment["availability"] == eng.AVAILABILITY_TIMEOUT
    # A failed fetch never turns a flagged URL malicious on its own.
    flagged = eng.decide(_model(raw_prediction=1, score=0.99), inspection)
    assert flagged["verdict"] == eng.VERDICT_INSUFFICIENT


def test_regression_case_dns_failure():
    inspection = _base_inspection(
        reachable=False,
        blocked="dns_resolution_failed",
        tls=None,
        page={},
        availability=eng.AVAILABILITY_DNS_FAILED,
        title_available=False,
        dns={"host": "example.org", "validated_ips": [], "error": "dns_resolution_failed"},
        hops=[],
        final_url=None,
    )
    assessment = eng.decide(_model(raw_prediction=0, score=0.1), inspection)
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT
    assert assessment["availability"] == eng.AVAILABILITY_DNS_FAILED
    flagged = eng.decide(_model(raw_prediction=1, score=0.99), inspection)
    assert flagged["verdict"] == eng.VERDICT_INSUFFICIENT


def test_regression_case_invalid_url():
    inspection = inspect_website("not a valid url at all")
    assert inspection["blocked"] is not None
    assert inspection["reachable"] is False
    assert inspection["availability"] == eng.AVAILABILITY_UNKNOWN
    assessment = eng.decide(_model(raw_prediction=1, score=0.99), inspection)
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT


def test_regression_case_suspicious_path_on_valid_domain():
    # R2 flags a suspicious-looking path; full benign live inspection of
    # the destination overrides the string model — with explanation.
    inspection = _base_inspection(
        final_url="https://example.org/checkout/pay?id=88192",
    )
    inspection["dns"]["host"] = "example.org"
    assessment = eng.decide(_model(raw_prediction=1, score=0.97), inspection)
    assert assessment["verdict"] == eng.VERDICT_SAFE
    assert assessment["availability"] == eng.AVAILABILITY_REACHABLE
    assert any("overridden" in line for line in assessment["evidence"])


def test_regression_case_harmless_path_on_valid_domain():
    assessment = eng.decide(
        _model(raw_prediction=0, score=0.05), _base_inspection()
    )
    assert assessment["verdict"] == eng.VERDICT_SAFE
    assert assessment["availability"] == eng.AVAILABILITY_REACHABLE
    assert assessment["signals"]["full_inspection"] is True


# ---------------------------------------------------------------------------
# Policy layer (regression from previous fix, unchanged behavior)
# ---------------------------------------------------------------------------


def test_youtube_playlist_policy_still_approved():
    policy = evaluate_trusted_url_policy(
        "https://www.youtube.com/playlist?list=PLU6SqdYcYsfJz9FAzbgocIjlkw4NXAar-"
    )
    assert policy.approved is True


def test_lookalike_policy_not_approved():
    policy = evaluate_trusted_url_policy(
        "https://youtube.com.evil-example.com/playlist?list=PLAB"
    )
    assert policy.approved is False


# ---------------------------------------------------------------------------
# Brand detection unit
# ---------------------------------------------------------------------------


def test_repo_slug_not_flagged():
    title = "GitHub - microsoft/vscode: Visual Studio Code"
    assert extract_brand_claim({"title": title}, "github.com") is None


def test_paypal_on_random_host_flagged():
    title = "PayPal - Secure Login"
    assert extract_brand_claim({"title": title}, "paypal-secure-pop.example") == "paypal"


# ---------------------------------------------------------------------------
# Instagram regression (false-positive fix)
# ---------------------------------------------------------------------------


def test_instagram_brand_on_official_hosts_not_flagged():
    """Every legitimate instagram.com host must be in the official set."""
    from backend.inspector import BRAND_OFFICIAL_HOSTS

    official = BRAND_OFFICIAL_HOSTS["instagram"]
    for host in (
        "instagram.com",
        "www.instagram.com",
        "m.instagram.com",
        "help.instagram.com",
    ):
        assert host in official
        assert extract_brand_claim({"title": "Instagram"}, host) is None


def test_instagram_brand_on_lookalike_host_flagged():
    assert extract_brand_claim({"title": "Instagram - Log in"}, "instagram-login.example.com") == "instagram"
    assert extract_brand_claim({"title": "Instagram"}, "instagram.com.evil.com") == "instagram"


# Instagram structural policy ------------------------------------------------


def _policy(url: str):
    from backend.url_policy import evaluate_trusted_url_policy

    return evaluate_trusted_url_policy(url)


def test_instagram_policy_homepage_and_profile():
    assert _policy("https://www.instagram.com/").approved is True
    assert _policy("https://www.instagram.com/").platform == "instagram"
    assert _policy("https://www.instagram.com/instagram/").approved is True
    assert _policy("https://www.instagram.com/instagram/").url_kind == "profile"
    assert _policy("https://instagram.com/natgeo/?hl=en").approved is True


def test_instagram_policy_post_reel_share_igsh():
    post = _policy("https://www.instagram.com/p/C1ABCdefghI/?igsh=MTQzZT0abcde-fg")
    assert post.approved is True and post.url_kind == "post"
    reel = _policy("https://www.instagram.com/reel/DEFGHIJKlmn/?igsh=MTQzZT0abcde-fg")
    assert reel.approved is True and reel.url_kind == "reel"
    share = _policy("https://instagram.com/shares/AAAA-token_-123?igsh=MTc8m")
    assert share.approved is True and share.url_kind == "share"
    story = _policy("https://www.instagram.com/stories/natgeo/345678912/")
    assert story.approved is True and story.url_kind == "story"


def test_instagram_policy_rejects_lookalikes_and_deceptive_params():
    assert _policy("https://instagram.com.evil.com/instagram/").approved is False
    assert _policy("https://instagram.com.evil.com/instagram/").platform is None
    assert _policy("https://instagram-login.example.com/p/C1ABCdefghI/").approved is False
    assert _policy("http://www.instagram.com/instagram/").rejection == "https_required"
    assert _policy("https://www.instagram.com/p/C1ABCdefghI/?igsh=javascript:alert(1)").approved is False
    assert _policy("https://www.instagram.com/p/C1ABCdefghI/?random=xyz").rejection == "unexpected_query_parameter"


# ---------------------------------------------------------------------------
# Live integration (network required)
# ---------------------------------------------------------------------------


KNOWN_LEGITIMATE = pytest.mark.skipif(
    not _network_available(), reason="network unavailable"
)


@KNOWN_LEGITIMATE
@pytest.mark.parametrize(
    "url",
    [
        "https://chatgpt.com/",
        "https://github.com/",
        "https://www.youtube.com/",
        "https://www.youtube.com/playlist?list=PLU6SqdYcYsfJz9FAzbgocIjlkw4NXAar-",
        "https://github.com/microsoft/vscode",
        "https://www.instagram.com/",
        "https://www.instagram.com/instagram/",
    ],
)
def test_real_legitimate_sites_safe(url: str):
    inspection = inspect_website(url)
    assert inspection["reachable"] is True, inspection.get("error")
    assessment = eng.decide(_model(), inspection)
    assert assessment["verdict"] in (eng.VERDICT_SAFE, eng.VERDICT_SUSPICIOUS), assessment["evidence"]
    if assessment["verdict"] == eng.VERDICT_SUSPICIOUS:
        assert assessment["signals"]["finding_counts"]["high"] == 0


@KNOWN_LEGITIMATE
@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/torvalds",           # profile (deep-owned path)
        "https://en.wikipedia.org/wiki/Phishing",  # developer/wiki deep path
        "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",  # cloud docs
        "https://store.steampowered.com/",       # gaming store
    ],
)
def test_real_platform_and_deep_paths_not_malicious(url: str):
    inspection = inspect_website(url)
    if not inspection["reachable"]:
        pytest.skip(f"destination temporarily unreachable: {inspection.get('error')}")
    assert inspection["blocked"] is None, inspection["blocked"]
    assessment = eng.decide(_model(), inspection)
    assert assessment["verdict"] != eng.VERDICT_MALICIOUS, assessment["evidence"]
    assert assessment["signals"]["finding_counts"]["high"] == 0


@KNOWN_LEGITIMATE
def test_real_short_link_redirect_chain_revalidated():
    """youtu.be short link redirects to www.youtube.com; every hop's
    destination must be DNS-validated and the final page inspected."""
    inspection = inspect_website("https://youtu.be/dQw4w9WgXcQ")
    if not inspection["reachable"]:
        pytest.skip(f"destination temporarily unreachable: {inspection.get('error')}")
    assert inspection["final_url"] is not None
    assessment = eng.decide(_model(), inspection)
    assert assessment["verdict"] != eng.VERDICT_MALICIOUS, assessment["evidence"]
    assert assessment["signals"]["finding_counts"]["high"] == 0
    assert assessment["availability"] in (
        eng.AVAILABILITY_REACHABLE,
        eng.AVAILABILITY_PAGE_NOT_FOUND,  # if the video id is ever retired
    )


@KNOWN_LEGITIMATE
def test_real_uninspectable_instagram_share_not_flagged_malicious():
    """A post/reel code the inspector cannot fully parse (login wall, anti-bot
    or 404) must never become MALICIOUS on inspection failure alone."""
    from backend import risk_engine as eng

    inspection = inspect_website("https://www.instagram.com/reel/abcdefghijk/")
    assessment = eng.decide(_model(), inspection)
    assert assessment["verdict"] != eng.VERDICT_MALICIOUS
    assert not any(
        f["code"] == "brand_impersonation" for f in assessment["findings"]
    )


# Instagram false-positive regression (reported live bug URL) -----------------

@KNOWN_LEGITIMATE
@pytest.mark.parametrize(
    "url",
    [
        "https://www.instagram.com/",                   # homepage
        "https://www.instagram.com/khushiroy.31/",      # exact reported regression URL
        "https://www.instagram.com/instagram/",         # official profile
        "https://www.instagram.com/p/C1ABCdefghI/",     # post URL format
        "https://www.instagram.com/reel/DEFGHIJKlmn/",  # reel URL format
    ],
)
def test_real_instagram_urls_not_flagged_by_inspection(url: str):
    inspection = inspect_website(url)
    assessment = eng.decide(_model(), inspection)
    # An anti-bot page, login wall, JS shell or thin body is never proof
    # of malware: uncertainty is preserved instead.
    assert assessment["verdict"] != eng.VERDICT_MALICIOUS, assessment["evidence"]
    assert not any(
        f["code"] == "brand_impersonation" for f in assessment["findings"]
    )
    assert assessment["signals"]["finding_counts"]["high"] == 0


@KNOWN_LEGITIMATE
def test_real_instagram_profile_khushiroy_regression():
    """Exact reported URL. When Instagram serves the real profile page
    (HTTP 200 with actual content), the verdict must be SAFE — the R2
    string score (0.99+) is an intentional deep-path false positive."""
    from backend.url_policy import evaluate_trusted_url_policy

    inspection = inspect_website("https://www.instagram.com/khushiroy.31/")
    assert evaluate_trusted_url_policy(
        "https://www.instagram.com/khushiroy.31/"
    ).approved is True
    full_page = (
        inspection["reachable"]
        and inspection["status_code"] == 200
        and (inspection.get("page") or {}).get("text_length", 0) > 0
        and not inspection.get("error")
    )
    if full_page:
        assessment = eng.decide(_model(), inspection)
        assert assessment["verdict"] == eng.VERDICT_SAFE, assessment["evidence"]
        assert assessment["availability"] == eng.AVAILABILITY_REACHABLE
        assert assessment["signals"]["r2_raw_decision"] == 1  # score preserved, not decisive


def test_instagram_impersonating_host_is_malicious():
    """An Instagram-branded title on a LOOKALIKE host is phishing, not a
    false positive: only the exact official hostnames are trusted."""
    inspection = _base_inspection(
        final_url="https://instagram.com.evil.com/khushiroy.31/",
        title_available=True,
        availability=eng.AVAILABILITY_REACHABLE,
    )
    inspection["dns"]["host"] = "instagram.com.evil.com"
    inspection["page"]["title"] = "Instagram - Log in to continue"
    assessment = eng.decide(_model(raw_prediction=0, score=0.1), inspection)
    assert assessment["verdict"] == eng.VERDICT_MALICIOUS
    assert any(f["code"] == "brand_impersonation" for f in assessment["findings"])


def test_instagram_antibot_shell_never_safe_or_malicious():
    """Instagram's JS shell (title 'Instagram', zero text) on the official
    host: uncertainty preserved — never SAFE on thin evidence, never
    MALICIOUS on inspection failure alone."""
    shell = dict(_EMPTY_PAGE)
    shell["title"] = "Instagram"
    inspection = _base_inspection(
        final_url="https://www.instagram.com/khushiroy.31/",
        title_available=False,
        availability=eng.AVAILABILITY_TITLE_UNAVAILABLE,
        page=shell,
    )
    inspection["dns"]["host"] = "www.instagram.com"
    flagged = eng.decide(_model(raw_prediction=1, score=0.9), inspection)
    assert flagged["verdict"] in (
        eng.VERDICT_SUSPICIOUS,
        eng.VERDICT_INSUFFICIENT,
    )
    benign = eng.decide(_model(raw_prediction=0, score=0.05), inspection)
    assert benign["verdict"] == eng.VERDICT_INSUFFICIENT


@KNOWN_LEGITIMATE
@pytest.mark.parametrize(
    "url,reason",
    [
        ("https://nonexistent-domain-abcxyz.invalid/", "dns"),
        ("http://nonexistent-domain-abcxyz123.invalid/", "dns"),
    ],
)
def test_inaccessible_sites_unverified(url: str, reason: str):
    inspection = inspect_website(url)
    assert inspection["reachable"] is False
    assert inspection["blocked"] == "dns_resolution_failed"
    assessment = eng.decide(_model(), inspection)
    assert assessment["verdict"] == eng.VERDICT_INSUFFICIENT
    assert assessment["availability"] == eng.AVAILABILITY_DNS_FAILED
