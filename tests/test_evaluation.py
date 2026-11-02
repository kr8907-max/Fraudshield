"""
FraudShield — QA evaluation suite (synthetic, deterministic)
============================================================

Permanent evaluation bank that measures the layered scanner's behaviour on
crafted inspection evidence:

* legitimate bank    -> expect SAFE or (SUSPICIOUS with zero high findings)
* malicious bank     -> expect MALICIOUS or (SUSPICIOUS; never SAFE)
* unverified bank    -> expect INSUFFICIENT EVIDENCE (never SAFE/MALICIOUS)
* edge/security bank -> SSRF, rebinding, punycode, credentials, redirects

Everything here is SYNTHETIC (crafts inspection dicts and calls the risk
engine directly). False positives/detections measured here are therefore
on the engine logic, not on live websites; live measurements are reported
separately in reports/ and tests marked with the KNOWN_LEGITIMATE guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import risk_engine as eng  # noqa: E402
from backend.inspector import _hostname_flags, _ip_is_unsafe, _validate_destination  # noqa: E402
import ipaddress  # noqa: E402


# ---------------------------------------------------------------------------
# Inspection dict factory
# ---------------------------------------------------------------------------

def _tls(cn: str = "example.org") -> dict:
    return {
        "handshake": "ok",
        "tls_version": "TLSv1.3",
        "cipher": "TLS_AES_128_GCM_SHA256",
        "subject_cn": cn,
        "issuer": "Some CA",
        "not_after": "Jan 01 00:00:00 2028 GMT",
        "hostname": cn,
    }


def _site(
    host: str,
    title: str = "Welcome",
    text_length: int = 800,
    *,
    password_fields: int = 0,
    sensitive_inputs: int = 0,
    forms: int = 0,
    form_actions: list | None = None,
    meta_refresh: str | None = None,
    served_executable: bool = False,
    tls: dict | None = "default",
    scheme: str = "https",
    host_flags: list | None = None,
    iframes: list | None = None,
) -> dict:
    if tls == "default":
        tls = _tls(host) if scheme == "https" else None
    return {
        "requested_url": f"{scheme}://{host}/",
        "reachable": True,
        "blocked": None,
        "error": None,
        "availability": "REACHABLE" if text_length and title else "TITLE_UNAVAILABLE",
        "title_available": bool(title),
        "host_flags": host_flags or [],
        "hops": [{"url": f"{scheme}://{host}/", "status": 200, "ip": "203.0.113.7", "location": None}],
        "final_url": f"{scheme}://{host}/",
        "status_code": 200,
        "content_type": "text/html",
        "is_html": True,
        "served_executable": served_executable,
        "content_truncated": False,
        "tls": tls,
        "dns": {"host": host, "validated_ips": ["203.0.113.7"], "error": None},
        "redirect_scheme_downgrade": False,
        "page": {
            "title": title,
            "text_length": text_length,
            "text_sample": "login password form" if password_fields else "example page content",
            "forms": forms,
            "form_actions": form_actions or [],
            "password_fields": password_fields,
            "sensitive_inputs": sensitive_inputs,
            "links": 5,
            "external_scripts": 0,
            "iframes": iframes or [],
            "meta_refresh": meta_refresh,
            "http_images": 0,
        },
    }


def _model_flagged() -> dict:
    return {"raw_prediction": 1, "model_score": 0.95, "policy": {"approved": False}}


def _model_benign() -> dict:
    return {"raw_prediction": 0, "model_score": 0.05, "policy": {"approved": False}}


# ---------------------------------------------------------------------------
# 1. Legitimate bank — measured false-positive rate must stay low
# ---------------------------------------------------------------------------

LEGITIMATE_BANK = [
    # social platforms / profile & share links
    _site("www.instagram.com", "Khushi (@khushiroy.31) • Instagram photos and videos"),
    _site("www.instagram.com", None, 0, tls=_tls("www.instagram.com")),  # JS shell
    _site("www.youtube.com", "Home — YouTube"),
    _site("chatgpt.com", "ChatGPT: Everything you need"),
    _site("x.com", "Home / X"),
    _site("www.reddit.com", "Reddit - Dive into anything"),
    # github repos & profiles
    _site("github.com", "GitHub - microsoft/vscode: Visual Studio Code"),
    _site("github.com", "torvalds - GitHub"),
    # video & playlists
    _site("www.youtube.com", "Playlist: Python tutorials — YouTube"),
    _site("youtu.be", "The [OFFICIAL] video — YouTube"),
    # cloud & developer platforms
    _site("gitlab.com", "Dashboard · GitLab"),
    _site("developer.mozilla.org", "MDN Web Docs"),
    _site("docs.python.org", "3.14 Documentation"),
    _site("gist.github.com", "Awesome tip — gists"),
    # e-commerce / banking-style
    _site("www.amazon.com", "Amazon.com. All Departments", sensitive_inputs=2),
    _site("www.paypal.com", "Log in to your PayPal account", password_fields=1),
    _site("secure.chase.com", "Chase Bank - Sign in", password_fields=1),
    # deep paths & query params
    _site("www.instagram.com", "Profile • Instagram", forms=1),
    _site("www.github.com", "Repository · Pull requests"),
    _site("en.wikipedia.org", "Phishing - Wikipedia"),
]


@pytest.mark.parametrize("inspection", LEGITIMATE_BANK, ids=lambda i: i["final_url"])
def test_legitimate_never_malicious(inspection: dict):
    assessment = eng.decide(_model_flagged(), inspection)
    assert assessment["verdict"] not in (eng.VERDICT_MALICIOUS,), (inspection["final_url"], assessment["findings"])
    assert assessment["signals"]["finding_counts"]["high"] == 0, assessment["findings"]


@pytest.mark.parametrize("inspection", LEGITIMATE_BANK, ids=lambda i: i["final_url"])
def test_legitimate_with_benign_model_verdicts_safe(inspection: dict):
    assessment = eng.decide(_model_benign(), inspection)
    allowable = {eng.VERDICT_SAFE, eng.VERDICT_SUSPICIOUS, eng.VERDICT_INSUFFICIENT}
    if inspection["page"].get("password_fields") and inspection.get("tls") is None:
        allowable.add(eng.VERDICT_MALICIOUS)  # password-over-http remains high
    assert assessment["verdict"] in allowable


def test_legitimate_false_positive_rate():
    """Engine false-positive rate on the legitimate bank must be 0."""
    fp = 0
    for inspection in LEGITIMATE_BANK:
        assessment = eng.decide(_model_flagged(), inspection)
        if assessment["verdict"] == eng.VERDICT_MALICIOUS or assessment["signals"]["finding_counts"]["high"] > 0:
            fp += 1
    rate = fp / len(LEGITIMATE_BANK)
    assert rate == 0.0, f"false positives: {fp}/{len(LEGITIMATE_BANK)}"


# ---------------------------------------------------------------------------
# 2. Malicious bank — measured detection rate
# ---------------------------------------------------------------------------

def _phishy(host: str, title: str = "Welcome", **page_kwargs) -> dict:
    return _site(host, title, **page_kwargs)


MALICIOUS_BANK = [
    # brand impersonation (lookalike/sound-alike domains)
    _phishy("paypa1-secure-login.xyz", "PayPal - Verify Your Account", password_fields=1),
    _phishy("secure-paypal4321.example", "PayPal account limited — log in", forms=1),
    _phishy("instagram-login.example.com", "Instagram - Log in to continue", password_fields=1),
    _phishy("apple-id-verify.top", "Apple ID — Sign in required"),
    _phishy("github-login.ru", "GitHub Login: Sign in to GitHub", password_fields=1),
    # deceptive subdomains
    _phishy("paypal.com.verify-account.xyz", "PayPal Security Center", password_fields=1),
    _phishy("apple.com.unusualactivity.net", "Apple — Confirm your billing"),
    _phishy("youtube.com.renew.watchfree.info", "YouTube - Watch instantly"),
    # fake login / credential harvesting over unencrypted transport
    _site("secure-login.example.net", "Bank of Example — Online Banking", password_fields=1, scheme="http"),
    _site("ebanking-secure.example", "Login to eBanking", password_fields=1, scheme="http"),
    # homograph / punycode (page content otherwise benign, host is evasive)
    _phishy("xn--80ak6aa92e.com", "Secure Account Login", password_fields=1, host_flags=["punycode_host"]),
    _phishy("аpple.com", "Apple", host_flags=["non_ascii_host"]),  # Cyrillic 'а'
    # malicious / executable indicators
    _site("cdn-free-software.example", "Download your antivirus license", served_executable=True, scheme="http"),
    _site("updates-patch.example", "Office update", served_executable=True),
    # deceptive redirects
    _site("login-secure.example", "Please wait, verifying you…",
          meta_refresh="0; url=https://phishy.example.ru/next", password_fields=1),
    _site("promo.example", "Redirecting to your bank…",
          meta_refresh="0; url=http://credentials.example.net/form"),
]


@pytest.mark.parametrize("inspection", MALICIOUS_BANK, ids=lambda i: i["final_url"])
def test_malicious_never_safe(inspection: dict):
    assessment = eng.decide(_model_benign(), inspection)
    assert assessment["verdict"] != eng.VERDICT_SAFE, (inspection["final_url"], assessment["findings"])
    assert assessment["verdict"] in (eng.VERDICT_MALICIOUS, eng.VERDICT_SUSPICIOUS), (
        inspection["final_url"], assessment["findings"]
    )


@pytest.mark.parametrize("inspection", MALICIOUS_BANK, ids=lambda i: i["final_url"])
def test_malicious_gets_high_or_medium_signal(inspection: dict):
    assessment = eng.decide(_model_benign(), inspection)
    counts = assessment["signals"]["finding_counts"]
    assert counts["high"] + counts["medium"] > 0, (inspection["final_url"], assessment["findings"])


def test_malicious_detection_rate():
    detection = 0
    for inspection in MALICIOUS_BANK:
        assessment = eng.decide(_model_benign(), inspection)
        counts = assessment["signals"]["finding_counts"]
        if assessment["verdict"] == eng.VERDICT_MALICIOUS or counts["high"] + counts["medium"] > 0:
            detection += 1
    rate = detection / len(MALICIOUS_BANK)
    assert rate >= 0.9, f"detected: {detection}/{len(MALICIOUS_BANK)}"


# ---------------------------------------------------------------------------
# 3. Unverified bank — failures never produce a safety or risk claim
# ---------------------------------------------------------------------------

UNVERIFIED_BANK = [
    {"reachable": False, "blocked": "unsafe_destination:169.254.169.254", "error": None,
     "host_flags": [], "availability": "UNKNOWN", "hops": [], "final_url": None,
     "status_code": None, "content_type": None, "is_html": False, "served_executable": False,
     "content_truncated": False, "tls": None, "dns": None, "redirect_scheme_downgrade": False,
     "page": {}},
    {"reachable": False, "blocked": None, "error": "connection_timeout", "host_flags": [],
     "availability": "TIMEOUT", "hops": [], "final_url": None, "status_code": None,
     "content_type": None, "is_html": False, "served_executable": False,
     "content_truncated": False, "tls": None, "dns": {"host": "example.org", "validated_ips": [],
     "error": None}, "redirect_scheme_downgrade": False, "page": {}},
    {"reachable": False, "blocked": "dns_resolution_failed", "error": None, "host_flags": [],
     "availability": "DNS_FAILED", "hops": [], "final_url": None, "status_code": None,
     "content_type": None, "is_html": False, "served_executable": False,
     "content_truncated": False, "tls": None, "dns": None, "redirect_scheme_downgrade": False,
     "page": {}},
    {"reachable": False, "blocked": None, "error": "connection_failed", "host_flags": [],
     "availability": "CONNECTION_FAILED", "hops": [], "final_url": None, "status_code": None,
     "content_type": None, "is_html": False, "served_executable": False,
     "content_truncated": False, "tls": None, "dns": None, "redirect_scheme_downgrade": False,
     "page": {}},
    # reachable but a bare JS shell the model could not flag
    {"reachable": True, "blocked": None, "error": None, "host_flags": [],
     "availability": "TITLE_UNAVAILABLE", "title_available": False,
     "hops": [{"url": "https://example.org/", "status": 200, "ip": "1.1.1.1", "location": None}],
     "final_url": "https://example.org/", "status_code": 200, "content_type": "text/html",
     "is_html": True, "served_executable": False, "content_truncated": False, "tls": _tls(),
     "dns": {"host": "example.org", "validated_ips": ["1.1.1.1"], "error": None},
     "redirect_scheme_downgrade": False,
     "page": {"title": None, "text_length": 0, "text_sample": "", "forms": 0,
              "password_fields": 0, "sensitive_inputs": 0, "iframes": [], "meta_refresh": None,
              "http_images": 0}},
]


@pytest.mark.parametrize("inspection", UNVERIFIED_BANK, ids=range(len(UNVERIFIED_BANK)))
def test_failures_and_shells_are_never_claims(inspection: dict):
    flagged = eng.decide(_model_flagged(), inspection)
    assert flagged["verdict"] in (eng.VERDICT_SUSPICIOUS, eng.VERDICT_INSUFFICIENT), flagged["findings"]
    benign_view = eng.decide(_model_benign(), inspection)
    assert benign_view["verdict"] == eng.VERDICT_INSUFFICIENT, benign_view["findings"]


# ---------------------------------------------------------------------------
# 4. Security bank — SSRF, rebinding shapes, punycode, credentials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        "127.0.0.1", "10.0.0.5", "172.16.4.9", "192.168.1.1",
        "169.254.169.254", "100.64.3.7", "198.18.0.9", "255.255.255.255",
        "::1", "fe80::1", "fc00::5", "::ffff:127.0.0.1",
        "::ffff:10.0.0.5", "::ffff:169.254.169.254",
    ],
)
def test_ip_blocklist_including_mapped_ipv6(value: str):
    ip = ipaddress.ip_address(value)
    assert _ip_is_unsafe(ip) is True, value


@pytest.mark.parametrize(
    "host,expected",
    [
        ("xn--80ak6aa92e.com", ["punycode_host"]),
        ("аpple.com", ["non_ascii_host"]),
        ("example.org", []),
        ("xn--e1aybc.xn--p1ai", ["punycode_host"]),
        ("www.мир-пример.рф", ["non_ascii_host"]),
    ],
)
def test_hostname_deception_flags(host: str, expected: list):
    assert _hostname_flags(host) == expected


def test_punycode_host_is_suspicious_not_trusted():
    inspection = _site(
        "xn--80ak6aa92e.com", "Welcome to our store", host_flags=["punycode_host"],
    )
    assessment = eng.decide(_model_benign(), inspection)
    assert assessment["verdict"] == eng.VERDICT_SUSPICIOUS
    assert any(f["code"] == "deceptive_hostname" for f in assessment["findings"])


def test_rebinding_shape_private_ip_on_second_hop_is_blocked():
    """A host whose resolution returns a private IP is refused before contact."""
    result = _validate_destination("10.0.0.9", 80)
    assert result["error"].startswith("unsafe_destination")


def test_getaddrinfo_os_error_never_crashes(monkeypatch):
    import backend.inspector as mod

    def explode(host, port, proto=None):
        raise OSError("resolver failure platform error")

    monkeypatch.setattr(mod.socket, "getaddrinfo", explode)
    assert _validate_destination("example.org", 443) == {"error": "dns_resolution_failed"}


def _patch_dns(monkeypatch, ips: list):
    """Replace real DNS validation with a fixed validated-address list and
    disable hostname-deception flags for geometry-only tests."""
    import backend.inspector as mod

    monkeypatch.setattr(
        mod,
        "_validate_destination",
        lambda host, port: {"ips": ips} if ips else {"error": "no_resolvable_address"},
    )
    monkeypatch.setattr(mod, "_hostname_flags", lambda host: [])


def test_multi_ip_fallback_tries_next_address(monkeypatch):
    """When the first validated IP refuses the connection, the inspector
    falls back to the next validated address instead of giving up."""
    import backend.inspector as mod

    attempts: list = []

    class FakeResponse:
        status = 200

        def getheader(self, name):
            if name == "Location":
                return None
            if name == "Content-Type":
                return "text/html"
            if name == "Content-Length":
                return "5"
            return None

        def read(self, n):
            return b""

    class FakeConn:
        def __init__(self, host, ip, *a, **k):
            self.host, self.ip = host, ip
            self._tls_info = {"peer_cert": {}}
            attempts.append(ip)

        def request(self, *a, **k):
            if self.ip == "203.0.113.7":
                raise ConnectionRefusedError()

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    _patch_dns(monkeypatch, ["203.0.113.7", "203.0.113.8"])
    monkeypatch.setattr(mod, "_HTTPConnection", FakeConn)
    monkeypatch.setattr(mod, "_HTTPSConnection", FakeConn)

    result = mod.inspect_website("http://dual-stack.example/path?q=1")
    assert result["reachable"] is True
    assert result["status_code"] == 200
    assert attempts == ["203.0.113.7", "203.0.113.8"], attempts


def test_multi_ip_all_addresses_failing_reports_failure(monkeypatch):
    import backend.inspector as mod

    class FakeConn:
        def __init__(self, host, ip, *a, **k):
            self._tls_info = {"peer_cert": {}}

        def request(self, *a, **k):
            raise ConnectionRefusedError()

        def getresponse(self):
            raise ConnectionRefusedError()

        def close(self):
            pass

    _patch_dns(monkeypatch, ["203.0.113.7", "203.0.113.8"])
    monkeypatch.setattr(mod, "_HTTPConnection", FakeConn)
    monkeypatch.setattr(mod, "_HTTPSConnection", FakeConn)
    result = mod.inspect_website("http://dead.example/")
    assert result["reachable"] is False
    assert result["error"] == "connection_failed"
    assert result["availability"] == eng.AVAILABILITY_CONNECTION_FAILED


def test_tls_error_stops_immediately_without_ip_hopping(monkeypatch):
    """TLS failures are host-level: the inspector must not retry the other
    addresses (certificate problem would recur; report it honestly)."""
    import ssl

    import backend.inspector as mod

    attempts: list = []

    class FakeConn:
        def __init__(self, host, ip, *a, **k):
            attempts.append(ip)
            self._tls_info = {"peer_cert": {}}

        def request(self, *a, **k):
            raise ssl.SSLError("certificate verify failed")

        def getresponse(self):
            raise AssertionError()

        def close(self):
            pass

    _patch_dns(monkeypatch, ["203.0.113.7", "203.0.113.8"])
    monkeypatch.setattr(mod, "_HTTPConnection", FakeConn)
    monkeypatch.setattr(mod, "_HTTPSConnection", FakeConn)
    result = mod.inspect_website("https://tls-broken.example/")
    assert result["reachable"] is False
    assert result["error"] == "tls_certificate_error"
    assert len(attempts) == 1, attempts


# ---------------------------------------------------------------------------
# 5. Evidence-vs-verdict consistency property tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inspection", LEGITIMATE_BANK + MALICIOUS_BANK + UNVERIFIED_BANK)
def test_raw_r2_is_preserved_and_never_decides(inspection: dict):
    for model in (_model_flagged(), _model_benign()):
        assessment = eng.decide(model, inspection)
        assert assessment["signals"]["r2_raw_decision"] == model["raw_prediction"]
        assert assessment["signals"]["r2_model_score"] == model["model_score"]


@pytest.mark.parametrize("inspection", UNVERIFIED_BANK)
def test_availability_in_unverified_bank_matches_taxonomy(inspection: dict):
    assessment = eng.decide(_model_benign(), inspection)
    assert assessment["availability"] in {
        eng.AVAILABILITY_REACHABLE, eng.AVAILABILITY_PAGE_NOT_FOUND,
        eng.AVAILABILITY_TIMEOUT, eng.AVAILABILITY_CONNECTION_FAILED,
        eng.AVAILABILITY_DNS_FAILED, eng.AVAILABILITY_TITLE_UNAVAILABLE,
        eng.AVAILABILITY_UNKNOWN,
    }
