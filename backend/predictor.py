"""
FraudShield - prediction engine
===============================

This module is the SINGLE prediction interface for the selected FraudShield
model. A later FastAPI layer should call :func:`predict_url` (or construct one
:class:`FraudShieldPredictor` at startup) instead of touching the model file.

Current model
-------------
The promoted artifact is **Experiment C Winner R2** (14 canonical features,
100 trees, ``max_depth=24``, ``min_samples_leaf=5``). Promotion copied the
Experiment C artifact; it did not retrain or tune anything.

Score semantics
---------------
``model_score``
    ``RandomForestClassifier.predict_proba(...)[:, 1]`` for the malicious
    class. This is a forest score, **not a calibrated probability**.

``risk_score``
    A 0-100 display transform of ``model_score`` (``model_score * 100``).
    It exists for UI presentation and is not a probability either.

``prediction``
    The model's actual binary decision at the 0.5 boundary (0 benign,
    1 malicious). This is the threshold used for all Experiment C metrics.

``risk_level``
    A separate presentation policy, deliberately different from the model's
    0.5 decision boundary:

        model_score < 0.35                  -> SAFE
        0.35 <= model_score < 0.65          -> SUSPICIOUS
        model_score >= 0.65                 -> HIGH RISK

    These UI bands were **not** statistically optimized. A score of 0.55 is
    therefore ``prediction=1`` (malicious at 0.5) but ``risk_level=SUSPICIOUS``.

Explanations
------------
``reasons`` describe observable lexical URL characteristics (IP host, '@',
security-related words, many subdomains, etc.). They are deterministic and
are **not** model attribution: they do not claim that a feature caused the
RandomForest decision, and no SHAP values are fabricated. HTTPS is never
treated as proof of safety, and HTTP is never treated as automatic malice.

Known limitation discovered in Stage 5
--------------------------------------
Experiment C's external holdout covered homepage-form URLs only
(``https://<domain>`` and ``https://www.<domain>``). Local checks show that
many ordinary legitimate URLs **with paths or query strings** are scored
HIGH RISK by this model. The model is therefore not validated for
unrestricted browser URL scoring yet. Stage 6 must not treat this predictor
as production-ready for deep-path URLs without further benign path-form
coverage and evaluation.

Safety
------
Every URL is plain text. This module never opens sockets, resolves DNS,
performs WHOIS/TLS lookups, downloads pages, calls external APIs, or executes
URL content. It only uses :func:`backend.feature_extractor.extract_features`.

Invalid input
-------------
``predict_url`` never crashes on ordinary bad user input. For empty,
malformed, non-web, or non-string input it returns a structured result with
``valid=False`` and **no fabricated score**.

Run:
    python backend/predictor.py --self-test
    python backend/predictor.py --benchmark
    python backend/predictor.py "https://www.google.com"
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np

# Allow "python backend/predictor.py" from any working directory, while also
# working as "from backend.predictor import predict_url".
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.feature_extractor import (  # noqa: E402  (path setup must run first)
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    extract_features,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "fraudshield_model.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "model" / "model_metadata.json"

# Presentation bands (NOT the model's 0.5 decision boundary).
SAFE_BELOW = 0.35
HIGH_RISK_AT_OR_ABOVE = 0.65
CLASSIFICATION_THRESHOLD = 0.5

MAX_URL_LENGTH = 8192

RISK_SAFE = "SAFE"
RISK_SUSPICIOUS = "SUSPICIOUS"
RISK_HIGH = "HIGH RISK"
RISK_UNKNOWN = "UNKNOWN"

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")
_LOOSE_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")

RISK_POLICY = {
    "safe_below": SAFE_BELOW,
    "high_risk_at_or_above": HIGH_RISK_AT_OR_ABOVE,
    "classification_threshold": CLASSIFICATION_THRESHOLD,
    "note": (
        "risk_level bands are presentation policy, not statistically optimized "
        "thresholds; prediction uses the model's 0.5 decision boundary"
    ),
}

SCORE_NOTE = (
    "model_score is the RandomForest malicious-class score, not a calibrated "
    "probability; risk_score is a 0-100 display transform"
)


class ModelCompatibilityError(RuntimeError):
    """Raised when the model and its metadata do not satisfy the feature contract."""


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _invalid_reason(url: object) -> tuple[str, str] | None:
    """Return (error_code, message) for unusable input, else None."""
    if not isinstance(url, str):
        return (
            "not_a_string",
            f"URL must be a string, received {type(url).__name__}.",
        )

    text = url.strip()
    if not text:
        return ("empty_input", "URL is empty or contains only whitespace.")

    if len(text) > MAX_URL_LENGTH:
        return (
            "url_too_long",
            f"URL exceeds the {MAX_URL_LENGTH}-character input limit.",
        )

    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        return ("invalid_characters", "URL contains control characters.")

    scheme_match = _SCHEME_RE.match(text)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme not in ("http", "https"):
            return (
                "non_web_scheme",
                f"Scheme '{scheme}://' is not an http or https web URL.",
            )
    else:
        loose_match = _LOOSE_SCHEME_RE.match(text)
        if loose_match and "." not in loose_match.group(1):
            return (
                "non_web_scheme",
                f"'{loose_match.group(1)}:' is not an http or https web URL.",
            )

    return None


# ---------------------------------------------------------------------------
# Explanation helpers (lexical observations, not model attribution)
# ---------------------------------------------------------------------------


def _matched_keywords(url: str) -> list[str]:
    lowered = url.lower()
    return [keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered]


def _build_reasons(url: str, features: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    keywords = _matched_keywords(url)

    if features["has_ip_host"]:
        reasons.append(
            "URL uses an IP address as the host instead of a normal domain name."
        )

    if features["has_at_symbol"]:
        reasons.append(
            "URL contains an '@' symbol, which can obscure the real destination."
        )

    if keywords:
        shown = ", ".join(keywords[:4])
        if len(keywords) == 1:
            reasons.append(
                f"URL contains a security/account-related term: {shown}."
            )
        else:
            reasons.append(
                f"URL contains multiple security/account-related terms: {shown}."
            )

    if features["num_subdomains"] >= 3:
        reasons.append(
            f"URL has an unusually large number of subdomains "
            f"({features['num_subdomains']})."
        )

    if not features["has_ip_host"] and features["num_digits"] >= 8:
        reasons.append(
            f"URL contains many digits ({features['num_digits']})."
        )

    if features["num_special_chars"] >= 5:
        reasons.append(
            f"URL contains many special characters "
            f"({features['num_special_chars']})."
        )

    if features["url_length"] >= 100:
        reasons.append(
            f"URL is unusually long ({features['url_length']} characters)."
        )

    if features["num_hyphens"] >= 4:
        reasons.append(
            f"URL contains many hyphens ({features['num_hyphens']})."
        )

    if features["num_query_marks"] >= 2 or features["num_equal_signs"] >= 3:
        reasons.append("URL contains a complex query string.")

    if features["uses_https"] == 0:
        reasons.append(
            "URL does not use HTTPS (unencrypted transport); this observation "
            "alone does not determine risk."
        )

    if not reasons:
        return ["No strong suspicious lexical indicators were detected."]

    # Keep the response short and stable: the list order above is the priority.
    return reasons[:4]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class FraudShieldPredictor:
    """Loads one model/metadata pair and predicts URLs as text."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.is_file():
            raise ModelCompatibilityError(f"Model file not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise ModelCompatibilityError(
                f"Model metadata not found: {self.metadata_path}"
            )

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.model = joblib.load(self.model_path)

        # Single-row inference is dramatically faster with one worker:
        # n_jobs=-1 adds thread-pool overhead that dominates a 1x14 row.
        # This is a runtime setting only; predictions and the artifact are
        # unchanged.
        if getattr(self.model, "n_jobs", 1) != 1:
            self.model.n_jobs = 1

        # Metadata is authoritative for active feature order. Falling back to
        # the canonical list is only for older metadata that predates the field.
        active = self.metadata.get("active_feature_names") or list(FEATURE_NAMES)
        if not isinstance(active, (list, tuple)) or not all(
            isinstance(name, str) for name in active
        ):
            raise ModelCompatibilityError(
                "metadata 'active_feature_names' must be a list of strings"
            )
        self.active_feature_names: tuple[str, ...] = tuple(active)

        self._validate_compatibility()

    def _validate_compatibility(self) -> None:
        names = self.active_feature_names
        if not names:
            raise ModelCompatibilityError("No active feature names configured.")
        if len(set(names)) != len(names):
            raise ModelCompatibilityError("Active feature names contain duplicates.")

        unknown = [name for name in names if name not in FEATURE_NAMES]
        if unknown:
            raise ModelCompatibilityError(
                f"Active features are not in the canonical extractor: {unknown}"
            )

        metadata_count = self.metadata.get("active_feature_count")
        if metadata_count is not None and int(metadata_count) != len(names):
            raise ModelCompatibilityError(
                f"metadata active_feature_count={metadata_count} but "
                f"active_feature_names has {len(names)} entries"
            )

        if not hasattr(self.model, "predict_proba"):
            raise ModelCompatibilityError("Model does not expose predict_proba().")
        if int(getattr(self.model, "n_features_in_", -1)) != len(names):
            raise ModelCompatibilityError(
                f"Model expects {getattr(self.model, 'n_features_in_', '?')} "
                f"features but metadata lists {len(names)}."
            )

        classes = [int(value) for value in getattr(self.model, "classes_", [])]
        if classes != [0, 1]:
            raise ModelCompatibilityError(
                f"Expected model.classes_ == [0, 1], got {classes}."
            )
        self._malicious_index = classes.index(1)

    # -- public API --------------------------------------------------------

    def predict(self, url: str) -> dict:
        """
        Score one URL as text.

        Valid input returns ``valid=True`` and a model score. Invalid input
        returns ``valid=False`` with an error code and no fabricated score.
        This method never performs network access.
        """
        invalid = _invalid_reason(url)
        if invalid is not None:
            code, message = invalid
            shown = url if isinstance(url, str) else repr(url)
            return self._error_result(shown, code, message)

        self._assert_model_loaded()
        text = url.strip()
        features = extract_features(text)
        if features["url_length"] == 0 or features["hostname_length"] == 0:
            return self._error_result(
                url,
                "invalid_url",
                "Input could not be parsed as an http(s) URL or bare web address.",
            )

        # Build the vector by NAME in the metadata's exact order. This is the
        # only place feature order is established, so a model can never be fed
        # a silently reordered vector.
        vector = np.asarray(
            [[features[name] for name in self.active_feature_names]],
            dtype=np.float32,
        )

        # One forest traversal: scikit-learn's predict() itself calls
        # predict_proba() and takes argmax, so reading the row once and taking
        # argmax is exactly the model's binary decision (ties -> class 0).
        proba = self.model.predict_proba(vector)[0]
        model_score = float(proba[self._malicious_index])
        prediction = int(np.argmax(proba))

        if model_score < SAFE_BELOW:
            risk_level = RISK_SAFE
        elif model_score < HIGH_RISK_AT_OR_ABOVE:
            risk_level = RISK_SUSPICIOUS
        else:
            risk_level = RISK_HIGH

        return {
            "url": url,
            "valid": True,
            "prediction": prediction,
            "prediction_label": "malicious" if prediction == 1 else "benign",
            "risk_level": risk_level,
            "risk_score": round(model_score * 100.0, 2),
            "model_score": round(model_score, 6),
            "reasons": _build_reasons(text, features),
            "features": features,
            "model": {
                "name": self.metadata.get("model_name"),
                "variant": self.metadata.get("variant"),
                "feature_count": len(self.active_feature_names),
            },
            "risk_policy": RISK_POLICY,
            "score_note": SCORE_NOTE,
        }

    def _error_result(self, original: str, code: str, message: str) -> dict:
        return {
            "url": original,
            "valid": False,
            "error": code,
            "message": message,
            "prediction": None,
            "prediction_label": None,
            "risk_level": RISK_UNKNOWN,
            "risk_score": None,
            "model_score": None,
            "reasons": [],
            "features": {},
            "model": {
                "name": self.metadata.get("model_name"),
                "variant": self.metadata.get("variant"),
                "feature_count": len(self.active_feature_names),
            },
            "risk_policy": RISK_POLICY,
            "score_note": SCORE_NOTE,
        }

    def _assert_model_loaded(self) -> None:
        if self.model is None:
            raise ModelCompatibilityError("Model is not loaded.")


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_PREDICTOR: FraudShieldPredictor | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_predictor() -> FraudShieldPredictor:
    """Return the process-wide predictor, loading it exactly once."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_PREDICTOR is None:
                _DEFAULT_PREDICTOR = FraudShieldPredictor()
    return _DEFAULT_PREDICTOR


def reset_default_predictor() -> None:
    """Drop the cached predictor (useful for tests after file replacement)."""
    global _DEFAULT_PREDICTOR
    with _DEFAULT_LOCK:
        _DEFAULT_PREDICTOR = None


def predict_url(url: str) -> dict:
    """Predict one URL with the process-wide promoted model."""
    return get_default_predictor().predict(url)


# ---------------------------------------------------------------------------
# Self-test and benchmark
# ---------------------------------------------------------------------------

LEGITIMATE_URLS = (
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.python.org",
)

SUSPICIOUS_URLS = (
    "http://secure-login.example/verify/account",
    "http://192.0.2.123/login/verify/password",
    "http://account-security-update.example/confirm/password",
    "http://bank-login-verification.example/account/secure",
)

ROBUSTNESS_CASES = (
    ("example.com", True, None),
    ("https://example.com/path?a=1", True, None),
    ("", False, "empty_input"),
    ("   ", False, "empty_input"),
    ("not a url", False, "invalid_url"),
    ("javascript:alert(1)", False, "non_web_scheme"),
    ("ftp://example.com/file", False, "non_web_scheme"),
)


def _print_result_row(result: dict) -> None:
    if result["valid"]:
        print(
            f"    {result['risk_level']:<10} pred={result['prediction']} "
            f"model_score={result['model_score']:.4f} "
            f"risk_score={result['risk_score']:.2f}  {result['url']}"
        )
    else:
        print(
            f"    {result['risk_level']:<10} error={result['error']:<18} "
            f"{result['url']!r}"
        )


def _run_self_test() -> int:
    print("FraudShield predictor self-test")
    print("-" * 72)

    started = time.perf_counter()
    predictor = get_default_predictor()
    load_seconds = time.perf_counter() - started
    print(f"Model loaded once in {load_seconds:.3f}s")
    print(f"Model name    : {predictor.metadata.get('model_name')}")
    print(f"Variant       : {predictor.metadata.get('variant')}")
    print(f"Active features ({len(predictor.active_feature_names)}):")
    print("  " + ", ".join(predictor.active_feature_names))

    if predictor.active_feature_names != FEATURE_NAMES:
        print("\nWARNING: active feature order differs from canonical FEATURE_NAMES.")
    else:
        print("Feature order : matches canonical FEATURE_NAMES exactly")
    print()

    checks: list[tuple[str, bool, str]] = []

    def check(label: str, passed: bool, detail: str = "") -> None:
        checks.append((label, bool(passed), detail))

    print("Legitimate sanity URLs:")
    for url in LEGITIMATE_URLS:
        result = predictor.predict(url)
        _print_result_row(result)
        check(
            f"legit SAFE: {url}",
            result["valid"] and result["risk_level"] == RISK_SAFE,
            f"got {result['risk_level']} / score {result['model_score']}",
        )

    print("\nSuspicious synthetic URLs:")
    for url in SUSPICIOUS_URLS:
        result = predictor.predict(url)
        _print_result_row(result)
        check(
            f"suspicious HIGH RISK: {url}",
            result["valid"]
            and result["risk_level"] == RISK_HIGH
            and result["prediction"] == 1,
            f"got {result['risk_level']} / pred {result['prediction']}",
        )

    print("\nInput robustness:")
    for url, should_be_valid, expected_error in ROBUSTNESS_CASES:
        result = predictor.predict(url)
        _print_result_row(result)
        if should_be_valid:
            passed = result["valid"]
        else:
            passed = (
                not result["valid"]
                and result["error"] == expected_error
                and result["model_score"] is None
                and result["risk_score"] is None
            )
        check(f"robustness: {url!r}", passed, str(result.get("error") or ""))

    non_string = predictor.predict(None)  # type: ignore[arg-type]
    _print_result_row(non_string)
    check(
        "robustness: non-string input",
        not non_string["valid"]
        and non_string["error"] == "not_a_string"
        and non_string["model_score"] is None,
    )

    print("\nContract checks:")
    sample = predictor.predict("https://example.com/path?a=1")
    check("features contain every canonical name in order", tuple(sample["features"]) == FEATURE_NAMES)
    check(
        "every feature value is an integer",
        all(isinstance(value, int) for value in sample["features"].values()),
    )
    check("model_score in [0, 1]", 0.0 <= sample["model_score"] <= 1.0)
    check("risk_score in [0, 100]", 0.0 <= sample["risk_score"] <= 100.0)

    for url in LEGITIMATE_URLS + SUSPICIOUS_URLS:
        result = predictor.predict(url)
        # Mirror the model's own argmax decision; an exact 0.5 tie resolves
        # to class 0, exactly as scikit-learn's predict() does.
        expected_prediction = int(
            np.argmax([1.0 - result["model_score"], result["model_score"]])
        )
        expected_level = (
            RISK_SAFE
            if result["model_score"] < SAFE_BELOW
            else RISK_SUSPICIOUS
            if result["model_score"] < HIGH_RISK_AT_OR_ABOVE
            else RISK_HIGH
        )
        check(
            f"threshold policy: {url}",
            result["prediction"] == expected_prediction
            and result["risk_level"] == expected_level,
        )

    first = predictor.predict("https://www.google.com")["model_score"]
    second = predictor.predict("https://www.google.com")["model_score"]
    check("deterministic score", first == second)

    print("\nLatency sample (200 local predictions):")
    latency = _measure_latency(predictor, iterations=200)
    print(
        f"    mean={latency['mean_ms']:.3f} ms  median={latency['median_ms']:.3f} ms  "
        f"p95={latency['p95_ms']:.3f} ms  min={latency['min_ms']:.3f} ms  "
        f"max={latency['max_ms']:.3f} ms"
    )

    failed = 0
    print("\nChecks:")
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failed += 1
        suffix = f" ({detail})" if detail and not passed else ""
        print(f"  [{status}] {label}{suffix}")

    total = len(checks)
    print(f"\n{total - failed}/{total} checks passed.")
    return 0 if failed == 0 else 1


def _measure_latency(predictor: FraudShieldPredictor, iterations: int = 500) -> dict:
    urls = LEGITIMATE_URLS + SUSPICIOUS_URLS + (
        "example.com",
        "https://example.com/path?a=1",
    )
    samples: list[float] = []
    for index in range(max(1, iterations)):
        url = urls[index % len(urls)]
        started = time.perf_counter()
        predictor.predict(url)
        samples.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "iterations": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _run_benchmark(iterations: int) -> int:
    started = time.perf_counter()
    predictor = get_default_predictor()
    load_seconds = time.perf_counter() - started
    print("FraudShield predictor benchmark")
    print("-" * 72)
    print(f"Model load (first call): {load_seconds:.3f}s")
    print(f"Iterations: {iterations}")

    second_start = time.perf_counter()
    get_default_predictor()
    cached_seconds = time.perf_counter() - second_start
    print(f"Second get_default_predictor(): {cached_seconds * 1000:.3f} ms (cached)")

    latency = _measure_latency(predictor, iterations=iterations)
    print(
        f"Prediction latency: mean={latency['mean_ms']:.3f} ms  "
        f"median={latency['median_ms']:.3f} ms  p95={latency['p95_ms']:.3f} ms  "
        f"min={latency['min_ms']:.3f} ms  max={latency['max_ms']:.3f} ms"
    )
    print(f"Approx throughput: {1000.0 / latency['mean_ms']:.0f} predictions/second")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudShield URL risk predictor")
    parser.add_argument("url", nargs="?", help="URL to score (treated as text only)")
    parser.add_argument("--self-test", action="store_true", help="run deterministic checks")
    parser.add_argument("--benchmark", action="store_true", help="measure local latency")
    parser.add_argument(
        "--iterations", type=int, default=500, help="benchmark iterations (default 500)"
    )
    args = parser.parse_args(argv)

    if args.self_test or args.benchmark:
        status = 0
        if args.self_test:
            status |= _run_self_test()
            print()
        if args.benchmark:
            status |= _run_benchmark(max(1, args.iterations))
        return status

    if args.url is not None:
        print(json.dumps(predict_url(args.url), indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
