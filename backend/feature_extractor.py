"""
FraudShield - URL feature extraction
====================================

This module turns a URL string into a fixed-length vector of numbers.
The machine-learning model cannot read a URL directly, so we describe
the URL with measurable facts (length, dots, keywords, ...) and let the
model learn which combinations tend to look risky.

Key rule:
    FEATURE_NAMES defines the ONE official order of features.
    Training and prediction both build vectors with this same order.
    Never reorder or rename these entries without retraining the model.

This module uses only the Python standard library (no pandas).
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# 1. Feature contract (shared by training and prediction)
# ---------------------------------------------------------------------------

# The exact order used everywhere. A model trained with one order will give
# wrong answers if prediction uses a different order.
FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "hostname_length",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "num_query_marks",
    "num_equal_signs",
    "has_at_symbol",
    "uses_https",
    "has_ip_host",
    "num_subdomains",
    "suspicious_keyword_count",
)

# Words often used by phishing / fake-login pages.
# Matching is a plain case-insensitive substring check, which is simple and
# good enough for a college-level project. It can produce false positives
# (e.g. "accountant" contains "account"); the model decides how much weight
# to give this feature.
SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login",
    "signin",
    "verify",
    "account",
    "update",
    "secure",
    "password",
    "bank",
    "confirm",
    "payment",
    "security",
    "wallet",
    "authentication",
    "bonus",
    "prize",
)

# Symbols counted by "num_special_chars". We deliberately leave out the
# characters that already have their own feature (dot, hyphen, slash,
# question mark, equals sign, at sign) so this number means something
# different from the others.
SPECIAL_CHARACTERS = frozenset("!$%^&*()_+|~`{}[]:;\"'<>,%#&")

# "http://", "https://", "ftp://", ... at the start of the string.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

# Catches scheme-like prefixes written WITHOUT slashes, e.g. "javascript:"
# or "mailto:user@example.com". Prefixes containing a dot are treated as
# bare host:port input ("example.com:8080/path"), not as a scheme.
_LOOSE_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")

# Catches invalid-looking numeric hosts such as 999.999.999.999, which
# ipaddress rejects but which still look like an IP address to a user.
_IPV4_LIKE_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# ---------------------------------------------------------------------------
# 2. Small internal helpers
# ---------------------------------------------------------------------------


def _empty_features() -> dict[str, int]:
    """Return a full feature dict of zeros (used for unusable input)."""
    return {name: 0 for name in FEATURE_NAMES}


def _split_url(url: str):
    """
    Split the URL into parts, tolerating URLs written without a scheme.

    Returns (parsed, hostname, scheme, explicit_scheme) or None if the
    input cannot be treated as a normal http(s) web address.

    "explicit_scheme" is False for input like "example.com/login", where we
    added "https://" only to make parsing possible. In that case we do NOT
    claim the URL uses HTTPS, because the user never wrote https://.
    """
    text = url.strip()
    if not text:
        return None

    if _SCHEME_RE.match(text):
        candidate = text
        explicit_scheme = True
    else:
        # Refuse non-web inputs such as "javascript:alert(1)" that have a
        # scheme-like prefix but no "//". A dot in the prefix means it is
        # probably a bare host with a port ("example.com:8080/path").
        loose_match = _LOOSE_SCHEME_RE.match(text)
        if loose_match and "." not in loose_match.group(1):
            return None

        # Allows bare input such as "example.example/login" to be parsed.
        candidate = "https://" + text
        explicit_scheme = False

    try:
        parsed = urlsplit(candidate)
        # Some malformed hosts raise when .hostname is read.
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return None

    scheme = parsed.scheme.lower()

    # Only score normal web URLs. Inputs like "javascript:alert(1)" or
    # "data:text/html,..." are refused and treated as unusable.
    if explicit_scheme and scheme not in ("http", "https"):
        return None

    # Reject empty hosts and hosts containing spaces ("not a url").
    if not hostname or " " in hostname:
        return None

    return parsed, hostname, scheme, explicit_scheme


def _is_ip_host(hostname: str) -> int:
    """Return 1 if the hostname is an IP address (v4, v6, or IP-like), else 0."""
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        # e.g. 999.999.999.999 - not valid, but still "looks like" an IP.
        return 1 if _IPV4_LIKE_RE.match(hostname) else 0


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------


def extract_features(url: str) -> dict[str, int]:
    """
    Turn one URL into a dictionary of integer features.

    This function never raises for bad input: an empty, malformed, or
    non-web URL simply returns all zeros, so training data with a few bad
    rows cannot crash the pipeline. The API layer is still responsible for
    rejecting empty user input with a friendly message.
    """
    if not isinstance(url, str):
        return _empty_features()

    split = _split_url(url)
    if split is None:
        return _empty_features()

    _parsed, hostname, scheme, explicit_scheme = split

    # Use the cleaned text for counting so leading/trailing spaces do not
    # inflate url_length.
    full_url = url.strip()
    lowered_url = full_url.lower()

    # HTTPS only counts when the user actually wrote "https://".
    # Bare input got an internal "https://" prefix just for parsing, so it
    # must NOT be reported as secure - we simply do not know its scheme.
    uses_https = int(explicit_scheme and scheme == "https")

    has_ip_host = _is_ip_host(hostname)

    # A hostname like "a.b.example.com" has 2 subdomains ("a" and "b").
    # IP addresses and single-label hosts such as "localhost" get 0.
    # Note: multi-part public suffixes (e.g. "co.uk") are counted simply,
    # which is an acceptable simplification for this project.
    num_subdomains = 0 if has_ip_host else max(hostname.count(".") - 1, 0)

    features = {
        "url_length": len(full_url),
        "hostname_length": len(hostname),
        "num_dots": full_url.count("."),
        "num_hyphens": full_url.count("-"),
        "num_slashes": full_url.count("/"),
        "num_digits": sum(char.isdigit() for char in full_url),
        "num_special_chars": sum(char in SPECIAL_CHARACTERS for char in full_url),
        "num_query_marks": full_url.count("?"),
        "num_equal_signs": full_url.count("="),
        "has_at_symbol": int("@" in full_url),
        "uses_https": uses_https,
        "has_ip_host": has_ip_host,
        "num_subdomains": num_subdomains,
        # Number of DISTINCT suspicious words found, not total occurrences.
        "suspicious_keyword_count": sum(
            1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered_url
        ),
    }

    # Rebuild in FEATURE_NAMES order and force plain ints. This is the
    # guarantee that the returned dict always matches the training order.
    return {name: int(features[name]) for name in FEATURE_NAMES}


def extract_feature_vector(url: str) -> list[int]:
    """
    Same information as extract_features(), but as a list of numbers in
    FEATURE_NAMES order - the shape scikit-learn expects.
    """
    features = extract_features(url)
    return [features[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# 4. Safe local self-test (no model, no network, no API)
# ---------------------------------------------------------------------------

# All domains/IPs below are reserved for documentation/testing:
#   .example  -> RFC 2606 reserved, never a real website
#   192.0.2.0/24 -> RFC 5737 TEST-NET-1, never routable
_TEST_URLS: tuple[tuple[str, str], ...] = (
    ("Plain HTTPS URL", "https://example.com"),
    ("HTTP login page", "http://example.com/login"),
    ("No scheme at all", "example.example/verify-account"),
    ("Many warning signals", "http://secure-login.example/account/update?user=1&token=abc"),
    ("URL with @ symbol", "http://user@example.example/confirm"),
    ("Reserved IP host", "http://192.0.2.10/login"),
    ("Deep subdomains", "https://a.b.c.example.example/path"),
    ("Empty string", ""),
    ("Only spaces", "   "),
    ("Not a URL", "not a url"),
    ("Non-web scheme", "javascript:alert(1)"),
    ("Invalid numeric host", "http://999.999.999.999/pay"),
)


def _print_table() -> None:
    """Print a compact table so features can be eyeballed during review."""
    headers = (
        "url",
        "len",
        "host",
        "dot",
        "hyph",
        "slsh",
        "dig",
        "spec",
        "?",
        "=",
        "@",
        "https",
        "ip",
        "sub",
        "kw",
    )
    widths = (46, 4, 5, 4, 5, 5, 4, 5, 3, 3, 3, 6, 3, 4, 3)

    def row(values: tuple[object, ...]) -> str:
        cells = []
        for value, width in zip(values, widths):
            text = str(value)
            if len(text) > width:
                text = text[: width - 1] + "~"
            cells.append(text.ljust(width))
        return " ".join(cells).rstrip()

    print(row(headers))
    print("-" * (sum(widths) + len(widths) - 1))

    for _label, test_url in _TEST_URLS:
        shown_url = test_url if test_url else "(empty)"
        f = extract_features(test_url)
        print(
            row(
                (
                    shown_url,
                    f["url_length"],
                    f["hostname_length"],
                    f["num_dots"],
                    f["num_hyphens"],
                    f["num_slashes"],
                    f["num_digits"],
                    f["num_special_chars"],
                    f["num_query_marks"],
                    f["num_equal_signs"],
                    f["has_at_symbol"],
                    f["uses_https"],
                    f["has_ip_host"],
                    f["num_subdomains"],
                    f["suspicious_keyword_count"],
                )
            )
        )


def _run_self_test() -> int:
    """Run sanity checks. Returns 0 when every check passes, 1 otherwise."""
    checks: list[tuple[str, bool]] = []

    # Contract checks: names are unique and complete.
    checks.append(("FEATURE_NAMES has no duplicates", len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)))
    checks.append(("FEATURE_NAMES is not empty", len(FEATURE_NAMES) == 14))

    # Every case, including bad input, must return exactly the official keys
    # in the official order, with non-negative integer values.
    keys_ok = True
    types_ok = True
    for _label, test_url in _TEST_URLS:
        features = extract_features(test_url)
        if tuple(features.keys()) != FEATURE_NAMES:
            keys_ok = False
        if any(not isinstance(value, int) or value < 0 for value in features.values()):
            types_ok = False
    checks.append(("Every result matches FEATURE_NAMES order", keys_ok))
    checks.append(("Every value is a non-negative int", types_ok))

    # Determinism: same input must always produce the same output.
    sample = "http://secure-login.example/account/update?user=1"
    checks.append(
        ("Same input gives same output", extract_features(sample) == extract_features(sample))
    )

    # Behaviour checks tied to the feature definitions.
    https_case = extract_features("https://example.com/path")
    http_case = extract_features("http://example.com/path")
    bare_case = extract_features("example.example/path")
    ip_case = extract_features("http://192.0.2.10/login")
    empty_case = extract_features("")
    bad_scheme_case = extract_features("javascript:alert(1)")
    subdomain_case = extract_features("https://a.b.example.example/path")
    keyword_case = extract_features("https://example.com/LOGIN/verify")

    checks.append(("HTTPS detected", https_case["uses_https"] == 1))
    checks.append(("HTTP marked as not HTTPS", http_case["uses_https"] == 0))
    checks.append(("Missing scheme does not claim HTTPS", bare_case["uses_https"] == 0))
    checks.append(("Bare URL is still parsed", bare_case["hostname_length"] == len("example.example")))
    checks.append(("IP host detected", ip_case["has_ip_host"] == 1))
    checks.append(("IP host has zero subdomains", ip_case["num_subdomains"] == 0))
    checks.append(("Empty URL is all zeros", all(value == 0 for value in empty_case.values())))
    checks.append(("Non-web scheme is all zeros", all(value == 0 for value in bad_scheme_case.values())))
    checks.append(("Two subdomains counted", subdomain_case["num_subdomains"] == 2))
    checks.append(("Keywords are case-insensitive", keyword_case["suspicious_keyword_count"] >= 2))
    checks.append(("Keyword count is distinct", keyword_case["suspicious_keyword_count"] == 2))
    checks.append(("Vector matches dict order", extract_feature_vector(sample) == [extract_features(sample)[name] for name in FEATURE_NAMES]))

    failed = 0
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        if not passed:
            failed += 1

    print()
    print(f"{len(checks) - failed}/{len(checks)} checks passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    print("FraudShield feature extractor self-test")
    print("(All test domains are reserved and non-live.)")
    print()
    _print_table()
    print()
    raise SystemExit(_run_self_test())
