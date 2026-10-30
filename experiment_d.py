"""
FraudShield - Stage 5.5, Experiment D
=====================================

Goal: determine whether FraudShield can separate malicious URLs from ordinary
legitimate deep-path / query URLs without domain whitelisting, threshold
gaming, network access, leakage, or destroying malicious recall.

Diagnosis that motivates this experiment (measured, not assumed)
---------------------------------------------------------------
Stage 5 showed that R2 scores ordinary legitimate URLs with paths as HIGH RISK.
A controlled probe found the cause is a *scheme-representation artifact*, not
path shape:

    example.com/a          -> 0.0302  (SAFE)
    http://example.com/a   -> 0.9017  (MALICIOUS)
    https://example.com/a  -> 0.9533  (MALICIOUS)
    https://example.com    -> 0.0148  (SAFE, 2 slashes)
    https://example.com/   -> 0.9328  (MALICIOUS, 3 slashes)

93.2% of benign Kaggle rows are written without a scheme, while 67.3% of
malicious rows carry one. Only 2,263 benign "https + deep" rows existed in
training versus 9,956 malicious, so the model learned the dataset's writing
convention and an ultra-narrow "https + exactly 2 slashes" safe cell.

Intervention (Experiment D)
---------------------------
Add *derived browser-form representations* of real benign Kaggle TRAIN rows
that already contain genuine path/query text and have no explicit scheme.
The hostname/path/query/fragment are preserved byte-for-byte; only the scheme
representation is added. No new path or query string is ever invented.

Each selected source row gets BOTH an https:// and an http:// derived form.
This is deliberate: it prevents HTTPS from becoming a benign label proxy
(the Model B failure) because the identical benign path content appears under
both schemes.

Candidates
----------
D1 : 14 canonical features, R2 architecture, Experiment C training data +
     controlled benign browser-form deep augmentation.
D2 : identical rows, explicit active-feature list with "num_slashes" removed.
     Tests whether num_slashes is a brittle dataset shortcut.
D3 : at most ONE additional single-feature ablation, only if D1/D2 clearly
     implicate another structural shortcut. Optional, decided after D1/D2.

Leakage rules (unchanged from Experiment C)
-------------------------------------------
* Kaggle split reproduced exactly (StratifiedGroupKFold, hostname groups).
* Sanity-brand rows purged from the training side only.
* Tranco train/holdout domains reused verbatim from selection.json.
* Derived rows may only come from TRAIN rows; a held-out hostname can never
  produce a derived training row.
* The deep-URL evaluation set is built from the Kaggle TEST split only.

Safety
------
* Every URL is plain text. No HTTP, DNS, WHOIS, TLS, browser, or external API.
* No pandas. No GridSearchCV.
* model/fraudshield_model.pkl, model/model_metadata.json and all Experiment C
  artifacts are never written. Nothing is promoted by this script.

Run:
    python experiment_d.py
    python experiment_d.py --d3-feature num_dots
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from backend.dataset_loader import load_dataset
from backend.feature_extractor import FEATURE_NAMES, extract_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "data" / "malicious_phish.csv"
TRANCO_CSV = PROJECT_ROOT / "data" / "tranco" / "top-1m.csv"
TRANCO_SELECTION = PROJECT_ROOT / "data" / "tranco" / "selection.json"
R2_MODEL = PROJECT_ROOT / "model" / "experiment_c" / "winner_r2.pkl"
R2_METADATA = PROJECT_ROOT / "model" / "experiment_c" / "winner_r2.metadata.json"

EXPERIMENT_DIR = PROJECT_ROOT / "model" / "experiment_d"
COMPARISON_PATH = EXPERIMENT_DIR / "comparison.json"

# ---------------------------------------------------------------------------
# Predeclared experiment constants (fixed before any training)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
AUGMENTATION_SALT = "fraudshield-d-v1"
AUGMENTATION_CAP_SHARE = 0.08  # cap on added rows as a share of the training set
CLASSIFICATION_THRESHOLD = 0.5

R2_CONFIG = dict(
    n_estimators=100,
    max_depth=24,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

D2_FEATURES = tuple(name for name in FEATURE_NAMES if name != "num_slashes")
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

STRUCTURE_GROUPS = ("homepage", "path", "query", "path+query")
DEEP_GROUPS = ("path", "query", "path+query")
SCHEMES = ("bare", "http", "https")

# Sanity suites: never used for fitting, tuning, or accuracy.
SANITY_BRANDS = frozenset(
    {
        "google",
        "wikipedia",
        "github",
        "microsoft",
        "apple",
        "python",
        "mozilla",
        "reddit",
        "stackoverflow",
        "amazon",
    }
)

LEGIT_SANITY = (
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.python.org",
    "https://www.mozilla.org",
    "https://www.reddit.com",
    "https://stackoverflow.com",
    "https://www.amazon.com",
)

SUSPICIOUS_SANITY = (
    "http://secure-login.example/verify/account",
    "http://192.0.2.123/login/verify/password",
    "http://account-security-update.example/confirm/password",
    "http://bank-login-verification.example/account/secure",
)

FAMOUS_DEEP = (
    ("google search", "https://www.google.com/search?q=fraud"),
    ("wikipedia article", "https://en.wikipedia.org/wiki/Phishing"),
    ("github repository", "https://github.com/microsoft/vscode"),
    ("python downloads", "https://www.python.org/downloads/"),
    ("microsoft windows", "https://www.microsoft.com/en-us/windows"),
    ("bbc news", "https://www.bbc.com/news"),
)

EXTRA_DIAGNOSTICS = (
    ("example.org about", "https://example.org/about"),
    ("example.com path+query", "https://example.com/path?a=1"),
)

SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
LOOSE_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:.2f}%" if whole else "0.00%"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def brand_hit(hostname: str) -> bool:
    return any(label in SANITY_BRANDS for label in hostname.split("."))


def _parse(url: str):
    """Scheme-tolerant parse. Returns SplitResult or None for non-web input."""
    text = url.strip()
    if not text:
        return None
    if SCHEME_RE.match(text):
        candidate = text
    else:
        loose = LOOSE_SCHEME_RE.match(text)
        if loose and "." not in loose.group(1):
            return None
        candidate = "https://" + text
    try:
        return urlsplit(candidate)
    except ValueError:
        return None


def structure_of(url: str) -> str | None:
    parsed = _parse(url)
    if parsed is None:
        return None
    has_path = parsed.path not in ("", "/")
    has_query = parsed.query != ""
    if has_path and has_query:
        return "path+query"
    if has_path:
        return "path"
    if has_query:
        return "query"
    return "homepage"


def scheme_of(url: str) -> str:
    low = url.strip().lower()
    if low.startswith("https://"):
        return "https"
    if low.startswith("http://"):
        return "http"
    return "bare"


def browser_form(url: str, scheme: str = "https") -> str:
    """Add a scheme only when the source has none. Path/query/fragment untouched."""
    text = url.strip()
    if SCHEME_RE.match(text):
        return text
    return f"{scheme}://{text}"


def is_deep(url: str) -> bool:
    return structure_of(url) in DEEP_GROUPS


def suppression_hash(url: str, salt: str = AUGMENTATION_SALT) -> str:
    return hashlib.sha256(f"{url}|{salt}".encode("utf-8")).hexdigest()


def feature_stats(values: np.ndarray) -> dict:
    return {
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
    }


def active_matrix(X: np.ndarray, active_names: tuple[str, ...]) -> np.ndarray:
    if tuple(active_names) == FEATURE_NAMES:
        return X
    return X[:, [FEATURE_INDEX[name] for name in active_names]]


def matrix_from_urls(urls: list[str], active_names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray(
        [[extract_features(url)[name] for name in active_names] for url in urls],
        dtype=np.float32,
    )


def scheme_structure_table(urls, labels) -> dict:
    table = {
        int(label): {scheme: {group: 0 for group in STRUCTURE_GROUPS} for scheme in SCHEMES}
        for label in (0, 1)
    }
    for url, label in zip(urls, labels):
        group = structure_of(url)
        if group is None:
            continue
        table[int(label)][scheme_of(url)][group] += 1
    return table


def format_structure_table(table: dict) -> str:
    lines = []
    for label, name in ((0, "benign"), (1, "malicious")):
        lines.append(f"    {name}")
        for scheme in SCHEMES:
            row = table[label][scheme]
            total = sum(row.values())
            lines.append(
                f"      {scheme:<6} total={total:>8,}  "
                + "  ".join(f"{group}={row[group]:>7,}" for group in STRUCTURE_GROUPS)
            )
    return "\n".join(lines)


def cramers_v(table: dict) -> tuple[float, dict]:
    """Cramer's V for scheme x label (2x3)."""
    scheme_label = {
        0: {scheme: sum(table[0][scheme].values()) for scheme in SCHEMES},
        1: {scheme: sum(table[1][scheme].values()) for scheme in SCHEMES},
    }
    n = sum(sum(row.values()) for row in scheme_label.values())
    if n == 0:
        return 0.0, scheme_label
    row_totals = [sum(scheme_label[0].values()), sum(scheme_label[1].values())]
    col_totals = [scheme_label[0][s] + scheme_label[1][s] for s in SCHEMES]
    chi2 = 0.0
    for row_index, label in enumerate((0, 1)):
        for col_index, scheme in enumerate(SCHEMES):
            expected = row_totals[row_index] * col_totals[col_index] / n
            if expected:
                chi2 += (scheme_label[label][scheme] - expected) ** 2 / expected
    return float(np.sqrt(chi2 / (n * 1))), scheme_label


def univariate_auc(X: np.ndarray, y: np.ndarray, name: str) -> float:
    return float(roc_auc_score(y, X[:, FEATURE_INDEX[name]]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_internal(model, X, y, styles) -> dict:
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predictions, labels=[0, 1]).ravel())

    payload = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": CLASSIFICATION_THRESHOLD,
        "rows": int(len(y)),
        "by_style": {},
    }
    for scheme in SCHEMES:
        bm = (styles == scheme) & (y == 0)
        mm = (styles == scheme) & (y == 1)
        payload["by_style"][scheme] = {
            "benign_count": int(bm.sum()),
            "benign_fp": int(predictions[bm].sum()) if bm.any() else 0,
            "benign_fp_rate": float(predictions[bm].mean()) if bm.any() else 0.0,
            "malicious_count": int(mm.sum()),
            "malicious_recall": float(predictions[mm].mean()) if mm.any() else 0.0,
        }
    return payload


def evaluate_external(model, X, urls, domains) -> dict:
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    by_domain: dict[str, list[int]] = {}
    for position, domain in enumerate(domains):
        by_domain.setdefault(domain, []).append(position)
    domain_safe = [
        all(int(predictions[p]) == 0 for p in positions) for positions in by_domain.values()
    ]
    apex = [i for i, u in enumerate(urls) if not u.startswith("https://www.")]
    www = [i for i, u in enumerate(urls) if u.startswith("https://www.")]

    worst = sorted(
        (
            {
                "domain": domain,
                "max_score": round(max(float(scores[p]) for p in positions), 6),
            }
            for domain, positions in by_domain.items()
        ),
        key=lambda item: item["max_score"],
        reverse=True,
    )[:15]

    return {
        "url_count": len(urls),
        "domain_count": len(by_domain),
        "url_safe_rate": float((predictions == 0).mean()),
        "domain_safe_rate": float(np.mean(domain_safe)),
        "domains_safe": int(np.sum(domain_safe)),
        "domains_unsafe": int(len(domain_safe) - np.sum(domain_safe)),
        "apex_safe_rate": float((predictions[apex] == 0).mean()) if apex else 0.0,
        "www_safe_rate": float((predictions[www] == 0).mean()) if www else 0.0,
        "score_mean": float(scores.mean()),
        "worst_domains": worst,
        "note": (
            "Generated homepage representations of held-out Tranco domains. "
            "Never used for fitting, tuning, or threshold selection."
        ),
    }


def evaluate_deep(model, X, urls, structures, labels, name: str) -> dict:
    """Deep-URL suite. labels all 0 for the benign suite, all 1 for malicious."""
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    label = int(labels[0])
    correct = (predictions == 0) if label == 0 else (predictions == 1)

    def rates(mask):
        count = int(mask.sum())
        if not count:
            return {"count": 0, "rate": None, "mean_score": None}
        return {
            "count": count,
            "rate": round(float(correct[mask].mean()), 6),
            "mean_score": round(float(scores[mask].mean()), 6),
        }

    structures = np.asarray(structures, dtype=object)
    return {
        "suite": name,
        "label": label,
        "rows": len(urls),
        "rate_key": "safe_rate" if label == 0 else "recall",
        "score_key": "false_positive_rate" if label == 0 else "false_negative_rate",
        "overall": {
            "count": len(urls),
            "rate": round(float(correct.mean()), 6),
            "mean_score": round(float(scores.mean()), 6),
        },
        "by_structure": {group: rates(structures == group) for group in DEEP_GROUPS},
        "note": (
            "Held-out Kaggle TEST rows only. Browser-form representation: a scheme "
            "was added only where the source row had none; existing path/query text "
            "is byte-identical. Never used for fitting."
        ),
    }


def evaluate_sanity(model, active_names: tuple[str, ...] = FEATURE_NAMES) -> dict:
    rows = []
    for kind, suite in (("legit", LEGIT_SANITY), ("suspicious", SUSPICIOUS_SANITY)):
        for url in suite:
            X = matrix_from_urls([url], active_names)
            rows.append(
                {
                    "kind": kind,
                    "url": url,
                    "prediction": int(model.predict(X)[0]),
                    "score": round(float(model.predict_proba(X)[0, 1]), 6),
                }
            )
    return {
        "rows": rows,
        "legit_safe_count": sum(1 for r in rows if r["kind"] == "legit" and r["prediction"] == 0),
        "legit_total": len(LEGIT_SANITY),
        "suspicious_flagged_count": sum(
            1 for r in rows if r["kind"] == "suspicious" and r["prediction"] == 1
        ),
        "suspicious_total": len(SUSPICIOUS_SANITY),
    }


def evaluate_diagnostics(model, active_names: tuple[str, ...] = FEATURE_NAMES) -> list[dict]:
    rows = []
    for label, url in FAMOUS_DEEP + EXTRA_DIAGNOSTICS:
        X = matrix_from_urls([url], active_names)
        rows.append(
            {
                "label": label,
                "url": url,
                "kind": "famous" if (label, url) in FAMOUS_DEEP else "extra",
                "prediction": int(model.predict(X)[0]),
                "score": round(float(model.predict_proba(X)[0, 1]), 6),
            }
        )
    return rows


def acceptance_gates(internal, external, sanity, deep_benign, deep_malicious, leakage_ok) -> dict:
    """Predeclared gates. Do not modify after seeing results."""
    checks = {
        "overall_malicious_recall_ge_0.87": internal["recall"] >= 0.87,
        "malicious_https_recall_ge_0.90": internal["by_style"]["https"]["malicious_recall"] >= 0.90,
        "external_tranco_homepage_safe_ge_0.95": external["domain_safe_rate"] >= 0.95,
        "suspicious_synthetic_4_of_4": sanity["suspicious_flagged_count"] == 4,
        "legit_deep_safe_ge_0.90": deep_benign["overall"]["rate"] >= 0.90,
        "malicious_deep_recall_ge_0.90": deep_malicious["overall"]["rate"] >= 0.90,
        "no_leakage": leakage_ok,
    }
    return {
        "checks": checks,
        "passed_count": sum(1 for v in checks.values() if v),
        "total": len(checks),
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Augmentation construction
# ---------------------------------------------------------------------------


def build_augmentation(
    train_indices: np.ndarray,
    urls: list[str],
    labels: np.ndarray,
    cap_rows: int,
) -> dict:
    """
    Deterministic, controlled benign browser-form deep augmentation.

    Eligible: benign TRAIN rows, no explicit scheme, genuine path and/or query.
    Selection: sha256(url|salt) ascending, allocated proportionally by
    structure group. Each selected row yields an https:// and an http:// form.
    """
    eligible: dict[str, list[str]] = {group: [] for group in DEEP_GROUPS}
    for index in train_indices:
        if int(labels[index]) != 0:
            continue
        url = urls[index].strip()
        if SCHEME_RE.match(url):
            continue
        group = structure_of(url)
        if group in DEEP_GROUPS:
            eligible[group].append(url)

    pool_total = sum(len(v) for v in eligible.values())
    for group in DEEP_GROUPS:
        eligible[group].sort(key=suppression_hash)

    # Allocate the source-row budget proportionally across structure groups.
    n_sources = max(1, cap_rows // 2)
    allocations: dict[str, int] = {}
    assigned = 0
    for group in DEEP_GROUPS:
        share = len(eligible[group]) / pool_total if pool_total else 0.0
        count = int(round(n_sources * share))
        count = min(count, len(eligible[group]))
        allocations[group] = count
        assigned += count
    # Deterministic top-up / trim to hit the exact budget.
    order = sorted(DEEP_GROUPS, key=lambda g: (-len(eligible[g]), g))
    cursor = 0
    while assigned < n_sources:
        group = order[cursor % len(order)]
        if allocations[group] < len(eligible[group]):
            allocations[group] += 1
            assigned += 1
        cursor += 1
        if cursor > 10_000:
            break
    while assigned > n_sources:
        group = order[cursor % len(order)]
        if allocations[group] > 1:
            allocations[group] -= 1
            assigned -= 1
        cursor += 1
        if cursor > 20_000:
            break

    source_urls: list[str] = []
    source_groups: dict[str, int] = {}
    for group in DEEP_GROUPS:
        chosen = eligible[group][: allocations[group]]
        source_groups[group] = len(chosen)
        source_urls.extend(chosen)

    derived_https = [browser_form(u, "https") for u in source_urls]
    derived_http = [browser_form(u, "http") for u in source_urls]

    digest = hashlib.sha256("\n".join(source_urls).encode("utf-8")).hexdigest()

    return {
        "spec": {
            "salt": AUGMENTATION_SALT,
            "selection_rule": (
                "sha256(source_url + '|' + salt) ascending, allocated proportionally "
                "by structure group (path / query / path+query)"
            ),
            "eligibility": (
                "benign Kaggle TRAIN rows only (post sanity purge), no explicit scheme, "
                "genuine path and/or query text"
            ),
            "derived_forms": "https://<source> and http://<source>",
            "preserved_verbatim": "hostname, path, query, fragment",
            "invented_content": "none - only the scheme representation is added",
            "originals_retained": True,
            "eligible_pool": {
                "total": pool_total,
                **{group: len(eligible[group]) for group in DEEP_GROUPS},
            },
            "selected_source_rows": len(source_urls),
            "selected_by_structure": source_groups,
            "derived_rows_total": len(derived_https) + len(derived_http),
            "derived_rows_https": len(derived_https),
            "derived_rows_http": len(derived_http),
            "source_selection_sha256": digest,
            "dual_scheme_rationale": (
                "Adding https PATHS alone would make HTTPS a benign label proxy again "
                "(the Model B failure). Adding the identical benign content under both "
                "schemes keeps the scheme uninformative for the augmented rows."
            ),
            "not_observed_urls": True,
        },
        "derived_urls": derived_https + derived_http,
        "derived_labels": np.zeros(len(derived_https) + len(derived_http), dtype=np.int8),
        "source_urls": source_urls,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_forest(X, y, config):
    model = RandomForestClassifier(**config)
    started = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - started


def run_variant(label, active_names, X_train_full, y_train, suites, context):
    print(f"\n[{label.upper()}] fitting {len(active_names)} features, "
          f"{R2_CONFIG['n_estimators']} trees, {len(y_train):,} rows...")
    X_train = active_matrix(X_train_full, active_names)
    model, seconds = fit_forest(X_train, y_train, R2_CONFIG)
    del X_train

    if model.n_features_in_ != len(active_names):
        raise SystemExit(
            f"STOP: {label} expects {model.n_features_in_} features but the active "
            f"list has {len(active_names)}."
        )

    internal = evaluate_internal(
        model,
        active_matrix(suites["X_test"], active_names),
        suites["y_test"],
        suites["styles_test"],
    )
    external = evaluate_external(
        model,
        active_matrix(suites["X_external"], active_names),
        suites["external_urls"],
        suites["external_domains"],
    )
    deep_benign = evaluate_deep(
        model,
        active_matrix(suites["X_deep_benign"], active_names),
        suites["deep_benign_urls"],
        suites["deep_benign_structures"],
        suites["deep_benign_labels"],
        "legitimate held-out deep URLs (browser form)",
    )
    deep_benign_original = evaluate_deep(
        model,
        active_matrix(suites["X_deep_benign_original"], active_names),
        suites["deep_benign_original_urls"],
        suites["deep_benign_original_structures"],
        suites["deep_benign_labels"],
        "legitimate held-out deep URLs (original form)",
    )
    deep_malicious = evaluate_deep(
        model,
        active_matrix(suites["X_deep_malicious"], active_names),
        suites["deep_malicious_urls"],
        suites["deep_malicious_structures"],
        suites["deep_malicious_labels"],
        "malicious held-out deep URLs (browser form)",
    )
    deep_malicious_original = evaluate_deep(
        model,
        active_matrix(suites["X_deep_malicious_original"], active_names),
        suites["deep_malicious_original_urls"],
        suites["deep_malicious_original_structures"],
        suites["deep_malicious_labels"],
        "malicious held-out deep URLs (original form)",
    )
    sanity = evaluate_sanity(model, active_names)
    diagnostics = evaluate_diagnostics(model, active_names)
    gates = acceptance_gates(
        internal, external, sanity, deep_benign, deep_malicious, context["leakage"]["ok"]
    )
    importances = {
        name: round(float(value), 6)
        for name, value in sorted(
            zip(active_names, model.feature_importances_.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = EXPERIMENT_DIR / f"{label}.pkl"
    joblib.dump(model, model_path, compress=3)

    reloaded = joblib.load(model_path)
    reloaded.n_jobs = 1
    probe = active_matrix(suites["X_test"][:8], active_names)
    if not np.array_equal(reloaded.predict(probe), model.predict(probe)):
        raise SystemExit(f"STOP: saved {label} does not reproduce predictions.")
    size_mb = model_path.stat().st_size / (1024 * 1024)

    metadata = {
        "model_name": f"FraudShield Experiment D - {label.upper()}",
        "model_type": "RandomForestClassifier",
        "status": "experimental - NOT approved for deployment; R2 remains promoted",
        "experiment": "D",
        "variant": label,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "active_feature_names": list(active_names),
        "active_feature_count": len(active_names),
        "canonical_feature_names": list(FEATURE_NAMES),
        "hyperparameters": {**R2_CONFIG, "n_jobs": -1},
        "provenance": context["provenance"],
        "augmentation": context["augmentation"]["spec"],
        "training_rows": context["training_rows"],
        "training_class_counts": context["training_class_counts"],
        "training_scheme_table": context["training_scheme_table"],
        "training_structure_table": context["training_structure_table"],
        "pre_training_gate": context["pretrain_gate"],
        "leakage_audit": context["leakage"],
        "threshold_policy": (
            "Classification uses the model's own 0.5 decision boundary. The 0.35/0.65 "
            "UI bands are unchanged and were not used for any evaluation here."
        ),
        "metrics_internal_test": internal,
        "external_holdout": external,
        "deep_url_evaluation": {
            "legitimate_browser_form": deep_benign,
            "legitimate_original_form": deep_benign_original,
            "malicious_browser_form": deep_malicious,
            "malicious_original_form": deep_malicious_original,
        },
        "sanity_suite": sanity,
        "diagnostics": diagnostics,
        "acceptance_gates": gates,
        "feature_importances": importances,
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "model_size_mb": round(size_mb, 2),
        "training_seconds": round(seconds, 2),
        "note": (
            "Deep-URL suites are browser-form representations of held-out Kaggle TEST "
            "rows, not observed browsing traffic. predict_proba is a model score, not a "
            "calibrated probability."
        ),
    }
    metadata_path = EXPERIMENT_DIR / f"{label}.metadata.json"
    write_json(metadata_path, metadata)

    print(
        f"    acc={internal['accuracy']:.4f} prec={internal['precision']:.4f} "
        f"rec={internal['recall']:.4f} f1={internal['f1']:.4f} auc={internal['roc_auc']:.4f}"
    )
    print(
        f"    https malicious recall={internal['by_style']['https']['malicious_recall']:.4f}  "
        f"https benign FP={internal['by_style']['https']['benign_fp_rate']:.4f}"
    )
    print(
        f"    legit deep safe={deep_benign['overall']['rate']:.4f}  "
        f"malicious deep recall={deep_malicious['overall']['rate']:.4f}  "
        f"tranco domain safe={external['domain_safe_rate']:.4f}"
    )
    print(f"    gates {gates['passed_count']}/{gates['total']}  size={size_mb:.2f} MB  "
          f"time={seconds:.1f}s")

    return {
        "label": label,
        "model": model,
        "path": model_path,
        "metadata_path": metadata_path,
        "active_names": active_names,
        "internal": internal,
        "external": external,
        "deep_benign": deep_benign,
        "deep_benign_original": deep_benign_original,
        "deep_malicious": deep_malicious,
        "deep_malicious_original": deep_malicious_original,
        "sanity": sanity,
        "diagnostics": diagnostics,
        "gates": gates,
        "importances": importances,
        "size_mb": size_mb,
        "seconds": seconds,
    }


def summary_row(name, result) -> dict:
    internal = result["internal"]
    return {
        "model": name,
        "accuracy": round(internal["accuracy"], 6),
        "precision": round(internal["precision"], 6),
        "recall": round(internal["recall"], 6),
        "f1": round(internal["f1"], 6),
        "roc_auc": round(internal["roc_auc"], 6),
        "tn": internal["tn"],
        "fp": internal["fp"],
        "fn": internal["fn"],
        "tp": internal["tp"],
        "malicious_https_recall": round(internal["by_style"]["https"]["malicious_recall"], 6),
        "benign_https_fp_rate": round(internal["by_style"]["https"]["benign_fp_rate"], 6),
        "external_domain_safe_rate": round(result["external"]["domain_safe_rate"], 6),
        "legit_deep_safe_rate": round(result["deep_benign"]["overall"]["rate"], 6),
        "legit_deep_fp_rate": round(1.0 - result["deep_benign"]["overall"]["rate"], 6),
        "malicious_deep_recall": round(result["deep_malicious"]["overall"]["rate"], 6),
        "legit_sanity_safe": result["sanity"]["legit_safe_count"],
        "suspicious_flagged": result["sanity"]["suspicious_flagged_count"],
        "model_size_mb": round(result["size_mb"], 2),
        "gates_passed": result["gates"]["passed_count"],
        "gates_total": result["gates"]["total"],
        "all_gates_passed": result["gates"]["passed"],
        "active_features": len(result["active_names"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudShield Experiment D")
    parser.add_argument(
        "--d3-feature",
        default=None,
        help="one additional canonical feature to ablate for D3 (optional)",
    )
    args = parser.parse_args(argv)

    print("=" * 84)
    print("FraudShield Stage 5.5 - Experiment D (deep-URL generalization)")
    print("=" * 84)
    print("All URLs are text only. No HTTP/DNS/WHOIS/TLS/browser/external API access.")

    for required in (DATASET_PATH, TRANCO_CSV, TRANCO_SELECTION, R2_MODEL, R2_METADATA):
        if not required.is_file():
            print(f"STOP: required file missing: {required}")
            return 1

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # -- A. Verify inputs ---------------------------------------------------
    print("\n[A] Input verification (read-only)")
    dataset_hash = file_sha256(DATASET_PATH)
    tranco_hash = file_sha256(TRANCO_CSV)
    selection = json.loads(TRANCO_SELECTION.read_text(encoding="utf-8"))
    r2_meta = json.loads(R2_METADATA.read_text(encoding="utf-8"))
    if r2_meta["provenance"]["kaggle_sha256"] != dataset_hash:
        print("STOP: Kaggle dataset differs from the Experiment C provenance hash.")
        return 1
    if r2_meta["provenance"]["tranco_sha256"] != tranco_hash:
        print("STOP: Tranco CSV differs from the Experiment C provenance hash.")
        return 1
    print(f"    Kaggle SHA-256 : {dataset_hash}  (matches Experiment C)")
    print(f"    Tranco SHA-256 : {tranco_hash}  (matches Experiment C)")
    print(f"    R2 model SHA-256: {file_sha256(R2_MODEL)}")

    # -- B. Load and reproduce the Experiment C split -----------------------
    print("\n[B] Loading Kaggle and reproducing the Stage 4 / Experiment C split...")
    started = time.perf_counter()
    bundle = load_dataset(
        DATASET_PATH, label_map=train.SOURCE_TO_BINARY, url_column="url", label_column="type"
    )
    X_all = np.asarray(bundle.X, dtype=np.float32)
    y_all = np.asarray(bundle.y, dtype=np.int8)
    urls = bundle.urls
    del bundle
    print(f"    Usable rows: {len(urls):,} ({time.perf_counter() - started:.1f}s)")

    train_idx, test_idx, groups = train.split_by_hostname(urls, y_all)
    if len(train_idx) != r2_meta["provenance"]["kaggle_train_rows_before_purge"]:
        print("STOP: split size differs from Experiment C.")
        return 1

    keep = np.asarray([not brand_hit(groups[i]) for i in train_idx], dtype=bool)
    train_purged = train_idx[keep]
    if len(train_purged) != r2_meta["provenance"]["kaggle_train_rows_after_purge"]:
        print("STOP: purged training size differs from Experiment C.")
        return 1
    print(f"    train {len(train_purged):,} / test {len(test_idx):,} "
          f"(identical to Experiment C)")

    kaggle_train_hosts = set(groups[train_purged])
    kaggle_test_hosts = set(groups[test_idx])
    kaggle_train_urls = {urls[i] for i in train_purged}
    kaggle_test_urls = {urls[i] for i in test_idx}

    # -- C. Root-cause diagnostics -----------------------------------------
    print("\n[C] Root-cause data analysis (structural distribution)")
    full_struct = scheme_structure_table(urls, y_all)
    print("    FULL DATASET, scheme x structure x class:")
    print(format_structure_table(full_struct))

    print("\n    Feature distributions, benign vs malicious (full dataset):")
    distributions = {}
    for name in ("num_slashes", "url_length", "num_special_chars",
                 "num_query_marks", "num_equal_signs"):
        distributions[name] = {}
        for label in (0, 1):
            distributions[name][label] = feature_stats(X_all[y_all == label, FEATURE_INDEX[name]])
        b, m = distributions[name][0], distributions[name][1]
        print(f"      {name:<20} benign mean={b['mean']:>8.3f} p90={b['p90']:>6.1f} | "
              f"malicious mean={m['mean']:>8.3f} p90={m['p90']:>6.1f}")

    train_struct = scheme_structure_table(
        [urls[i] for i in train_purged], y_all[train_purged]
    )
    test_struct = scheme_structure_table([urls[i] for i in test_idx], y_all[test_idx])

    # -- D. Held-out deep-URL evaluation sets (built before any training) ----
    print("\n[D] Building deep-URL evaluation sets from the Kaggle TEST split only")
    deep_benign_idx = [i for i in test_idx if y_all[i] == 0 and is_deep(urls[i])]
    deep_malicious_idx = [i for i in test_idx if y_all[i] == 1 and is_deep(urls[i])]

    deep_benign_urls = [browser_form(urls[i]) for i in deep_benign_idx]
    deep_benign_original_urls = [urls[i] for i in deep_benign_idx]
    deep_benign_structures = [structure_of(urls[i]) for i in deep_benign_idx]

    deep_malicious_urls = [browser_form(urls[i]) for i in deep_malicious_idx]
    deep_malicious_original_urls = [urls[i] for i in deep_malicious_idx]
    deep_malicious_structures = [structure_of(urls[i]) for i in deep_malicious_idx]

    deep_benign_domains = {groups[i] for i in deep_benign_idx}
    print(f"    legitimate deep sources : {len(deep_benign_idx):,} rows / "
          f"{len(deep_benign_domains):,} hostnames")
    for group in DEEP_GROUPS:
        print(f"      {group:<11} {deep_benign_structures.count(group):>7,}")
    print(f"    malicious deep sources  : {len(deep_malicious_idx):,} rows")
    for group in DEEP_GROUPS:
        print(f"      {group:<11} {deep_malicious_structures.count(group):>7,}")

    # -- E. Controlled augmentation ----------------------------------------
    print("\n[E] Controlled benign browser-form deep augmentation")
    base_train_rows = len(train_purged) + len(selection["train_domains"]) + len(
        selection["www_train_domains"]
    )
    cap_rows = int(AUGMENTATION_CAP_SHARE * base_train_rows)
    augmentation = build_augmentation(train_purged, urls, y_all, cap_rows)
    spec = augmentation["spec"]
    print(f"    eligible pool          : {spec['eligible_pool']['total']:,}")
    print(f"      path={spec['eligible_pool']['path']:,}  "
          f"query={spec['eligible_pool']['query']:,}  "
          f"path+query={spec['eligible_pool']['path+query']:,}")
    print(f"    selected source rows   : {spec['selected_source_rows']:,}")
    print(f"    derived rows (https)   : {spec['derived_rows_https']:,}")
    print(f"    derived rows (http)    : {spec['derived_rows_http']:,}")
    print(f"    derived rows total     : {spec['derived_rows_total']:,} "
          f"({pct(spec['derived_rows_total'], base_train_rows)} of base training rows)")
    print(f"    cap ({AUGMENTATION_CAP_SHARE:.0%} of base)    : {cap_rows:,}")
    print(f"    source selection SHA   : {spec['source_selection_sha256']}")

    # -- F. Assemble training data -----------------------------------------
    print("\n[F] Assembling training data (Experiment C + controlled augmentation)")
    tranco_train_urls = [f"https://{d}" for d in selection["train_domains"]]
    tranco_train_urls += [f"https://www.{d}" for d in selection["www_train_domains"]]

    X_kaggle = X_all[train_purged]
    X_tranco = matrix_from_urls(tranco_train_urls)
    X_aug = matrix_from_urls(augmentation["derived_urls"])

    X_train_full = np.vstack([X_kaggle, X_tranco, X_aug]).astype(np.float32, copy=False)
    y_train = np.concatenate(
        [
            y_all[train_purged],
            np.zeros(len(X_tranco), dtype=np.int8),
            augmentation["derived_labels"],
        ]
    )
    train_urls = (
        [urls[i] for i in train_purged] + tranco_train_urls + augmentation["derived_urls"]
    )
    del X_kaggle, X_tranco, X_aug

    train_scheme_struct = scheme_structure_table(train_urls, y_train)
    print(format_structure_table(train_scheme_struct))
    print(f"    Total training rows: {len(y_train):,}")

    # -- G. Pre-training distribution gate ---------------------------------
    print("\n[G] Pre-training distribution gate (computed BEFORE fitting)")
    v_final, scheme_label = cramers_v(train_scheme_struct)
    v_baseline, _ = cramers_v(
        scheme_structure_table([urls[i] for i in train_purged] , y_all[train_purged])
    )
    auc_https = univariate_auc(X_train_full, y_train, "uses_https")
    https_b = scheme_label[0]["https"]
    https_m = scheme_label[1]["https"]
    https_share = https_m / max(1, https_b + https_m)
    added_share = len(augmentation["derived_labels"]) / len(y_train)

    checks = {
        "uses_https_univariate_auc_0.45_to_0.55": 0.45 <= auc_https <= 0.55,
        "cramers_v_not_worse_than_baseline_by_0.02": v_final <= v_baseline + 0.02,
        "augmentation_le_10_percent_of_training": added_share <= 0.10,
        "dual_scheme_balanced_within_augmentation": (
            spec["derived_rows_https"] == spec["derived_rows_http"]
        ),
    }
    print(f"    uses_https univariate AUC : {auc_https:.4f}  (gate 0.45-0.55)")
    print(f"    Cramer's V scheme x label : {v_final:.4f}  (baseline {v_baseline:.4f})")
    print(f"    https malicious share     : {https_share:.4f}  "
          f"(benign {https_b:,} / malicious {https_m:,})")
    print(f"    augmentation share        : {added_share:.4f}  (cap 0.10)")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    pretrain = {
        "uses_https_univariate_auc": round(auc_https, 6),
        "cramers_v_final": round(v_final, 6),
        "cramers_v_baseline": round(v_baseline, 6),
        "https_benign_count": https_b,
        "https_malicious_count": https_m,
        "https_malicious_share": round(https_share, 6),
        "augmentation_share_of_training": round(added_share, 6),
        "scheme_table": scheme_label,
        "checks": checks,
        "passed": all(checks.values()),
        "note": (
            "HTTPS must not become a label proxy. The derived rows are added under "
            "BOTH schemes so the scheme carries no information for them."
        ),
    }
    if not pretrain["passed"]:
        print("\nSTOP: the construction creates a scheme shortcut. No model was trained.")
        write_json(EXPERIMENT_DIR / "pretrain_gate_failed.json", pretrain)
        return 1

    # -- H. Leakage audit ---------------------------------------------------
    print("\n[H] Leakage audit")
    derived_hosts = {_parse(u).hostname or "" for u in augmentation["derived_urls"]}
    train_hosts_all = kaggle_train_hosts | derived_hosts | {
        _parse(u).hostname or "" for u in tranco_train_urls
    }
    leakage = {
        "ok": True,
        "exact_train_test_hostname_overlap": len(train_hosts_all & kaggle_test_hosts),
        "deep_eval_hosts_in_training": len(deep_benign_domains & train_hosts_all),
        "deep_eval_rows_from_test_split": all(i in set(test_idx.tolist()) for i in deep_benign_idx),
        "derived_urls_equal_to_test_urls": len(set(augmentation["derived_urls"]) & kaggle_test_urls),
        "derived_urls_equal_to_train_urls": len(set(augmentation["derived_urls"]) & kaggle_train_urls),
        "derived_sources_in_test_split": sum(
            1 for u in augmentation["source_urls"] if u in kaggle_test_urls
        ),
        "augmentation_sources_all_benign_train": True,
        "famous_urls_in_training": len(
            {u for _l, u in FAMOUS_DEEP + EXTRA_DIAGNOSTICS} & set(train_urls)
        ),
        "sanity_brand_hosts_in_training": sum(1 for h in train_hosts_all if brand_hit(h)),
        "tranco_holdout_in_train": len(
            set(selection["holdout_domains"]) & set(selection["train_domains"])
        ),
        "derived_rows": spec["derived_rows_total"],
        "derived_source_rows": spec["selected_source_rows"],
        "note": (
            "Derived rows come only from benign TRAIN rows, so by construction no "
            "held-out hostname can produce a derived training row. The deep-URL suites "
            "are drawn exclusively from the Kaggle TEST split."
        ),
    }
    # Confirm every augmentation source row really is a benign TRAIN row.
    train_url_set = {urls[i] for i in train_purged}
    benign_train_set = {urls[i] for i in train_purged if y_all[i] == 0}
    leakage["derived_sources_all_in_train"] = all(
        u in train_url_set for u in augmentation["source_urls"]
    )
    leakage["derived_sources_all_benign"] = all(
        u in benign_train_set for u in augmentation["source_urls"]
    )
    leakage["ok"] = (
        leakage["exact_train_test_hostname_overlap"] == 0
        and leakage["deep_eval_hosts_in_training"] == 0
        and leakage["deep_eval_rows_from_test_split"]
        and leakage["derived_urls_equal_to_test_urls"] == 0
        and leakage["derived_sources_in_test_split"] == 0
        and leakage["famous_urls_in_training"] == 0
        and leakage["sanity_brand_hosts_in_training"] == 0
        and leakage["tranco_holdout_in_train"] == 0
        and leakage["derived_sources_all_in_train"]
        and leakage["derived_sources_all_benign"]
    )
    for key in (
        "exact_train_test_hostname_overlap",
        "deep_eval_hosts_in_training",
        "derived_urls_equal_to_test_urls",
        "derived_sources_in_test_split",
        "famous_urls_in_training",
        "sanity_brand_hosts_in_training",
        "tranco_holdout_in_train",
    ):
        print(f"    {key:<42}: {leakage[key]}")
    print(f"    [{'PASS' if leakage['ok'] else 'FAIL'}] leakage audit")
    if not leakage["ok"]:
        print("STOP: leakage detected. No model was trained.")
        write_json(EXPERIMENT_DIR / "leakage_failed.json", leakage)
        return 1

    # -- I. Shared suites and context --------------------------------------
    external_urls: list[str] = []
    external_domains: list[str] = []
    for domain in selection["holdout_domains"]:
        external_urls.extend([f"https://{domain}", f"https://www.{domain}"])
        external_domains.extend([domain, domain])

    X_deep_benign = matrix_from_urls(deep_benign_urls)
    X_deep_benign_original = matrix_from_urls(deep_benign_original_urls)
    X_deep_malicious = matrix_from_urls(deep_malicious_urls)
    X_deep_malicious_original = matrix_from_urls(deep_malicious_original_urls)

    suites = {
        "X_test": X_all[test_idx],
        "y_test": y_all[test_idx],
        "styles_test": np.asarray([scheme_of(urls[i]) for i in test_idx], dtype=object),
        "X_external": matrix_from_urls(external_urls),
        "external_urls": external_urls,
        "external_domains": external_domains,
        "X_deep_benign": X_deep_benign,
        "deep_benign_urls": deep_benign_urls,
        "deep_benign_original_urls": deep_benign_original_urls,
        "deep_benign_structures": deep_benign_structures,
        "deep_benign_original_structures": deep_benign_structures,
        "deep_benign_labels": np.zeros(len(deep_benign_urls), dtype=np.int8),
        "X_deep_benign_original": X_deep_benign_original,
        "X_deep_malicious": X_deep_malicious,
        "deep_malicious_urls": deep_malicious_urls,
        "deep_malicious_original_urls": deep_malicious_original_urls,
        "deep_malicious_structures": deep_malicious_structures,
        "deep_malicious_original_structures": deep_malicious_structures,
        "deep_malicious_labels": np.ones(len(deep_malicious_urls), dtype=np.int8),
        "X_deep_malicious_original": X_deep_malicious_original,
    }

    class_counts = {
        "benign": int((y_train == 0).sum()),
        "malicious": int((y_train == 1).sum()),
    }
    context = {
        "provenance": {
            "kaggle_dataset": DATASET_PATH.name,
            "kaggle_sha256": dataset_hash,
            "kaggle_train_rows_after_purge": int(len(train_purged)),
            "kaggle_test_rows_fixed": int(len(test_idx)),
            "tranco_csv": TRANCO_CSV.name,
            "tranco_sha256": tranco_hash,
            "tranco_train_domains": len(selection["train_domains"]),
            "tranco_holdout_domains": len(selection["holdout_domains"]),
            "tranco_selection_salt": selection["salt"],
            "reused_experiment_c_selection": True,
            "augmentation": spec,
            "deep_evaluation": {
                "legit_source_rows": len(deep_benign_idx),
                "legit_hostnames": len(deep_benign_domains),
                "malicious_source_rows": len(deep_malicious_idx),
                "representation": (
                    "browser form: https:// added only where the source row had no "
                    "scheme; path/query text is byte-identical to the held-out source"
                ),
                "never_used_for_fitting": True,
            },
            "distribution_statistics": distributions,
            "full_structure_table": full_struct,
            "train_structure_table_base": train_struct,
            "test_structure_table": test_struct,
        },
        "augmentation": augmentation,
        "training_rows": int(len(y_train)),
        "training_class_counts": class_counts,
        "training_scheme_table": scheme_label,
        "training_structure_table": train_scheme_struct,
        "pretrain_gate": pretrain,
        "leakage": leakage,
    }

    # -- J. Baseline R2 on identical suites --------------------------------
    print("\n[I] Experiment C R2 - same evaluation suites (reference)")
    r2_model = joblib.load(R2_MODEL)
    r2_model.n_jobs = 1
    r2 = {
        "internal": evaluate_internal(
            r2_model, active_matrix(suites["X_test"], FEATURE_NAMES), suites["y_test"],
            suites["styles_test"],
        ),
        "external": evaluate_external(
            r2_model, active_matrix(suites["X_external"], FEATURE_NAMES),
            external_urls, external_domains,
        ),
        "deep_benign": evaluate_deep(
            r2_model, active_matrix(suites["X_deep_benign"], FEATURE_NAMES),
            deep_benign_urls, deep_benign_structures, suites["deep_benign_labels"],
            "legitimate held-out deep URLs (browser form)",
        ),
        "deep_benign_original": evaluate_deep(
            r2_model, active_matrix(suites["X_deep_benign_original"], FEATURE_NAMES),
            deep_benign_original_urls, deep_benign_structures, suites["deep_benign_labels"],
            "legitimate held-out deep URLs (original form)",
        ),
        "deep_malicious": evaluate_deep(
            r2_model, active_matrix(suites["X_deep_malicious"], FEATURE_NAMES),
            deep_malicious_urls, deep_malicious_structures, suites["deep_malicious_labels"],
            "malicious held-out deep URLs (browser form)",
        ),
        "deep_malicious_original": evaluate_deep(
            r2_model, active_matrix(suites["X_deep_malicious_original"], FEATURE_NAMES),
            deep_malicious_original_urls, deep_malicious_structures,
            suites["deep_malicious_labels"],
            "malicious held-out deep URLs (original form)",
        ),
        "sanity": evaluate_sanity(r2_model, FEATURE_NAMES),
        "diagnostics": evaluate_diagnostics(r2_model, FEATURE_NAMES),
        "size_mb": R2_MODEL.stat().st_size / (1024 * 1024),
        "active_names": FEATURE_NAMES,
    }
    r2["gates"] = acceptance_gates(
        r2["internal"], r2["external"], r2["sanity"], r2["deep_benign"],
        r2["deep_malicious"], leakage["ok"],
    )
    print(f"    legit deep safe={r2['deep_benign']['overall']['rate']:.4f}  "
          f"malicious deep recall={r2['deep_malicious']['overall']['rate']:.4f}  "
          f"gates {r2['gates']['passed_count']}/{r2['gates']['total']}")

    # -- K. Train candidates ------------------------------------------------
    d1 = run_variant("d1", FEATURE_NAMES, X_train_full, y_train, suites, context)
    d2 = run_variant("d2", D2_FEATURES, X_train_full, y_train, suites, context)

    results = {"r2": r2, "d1": d1, "d2": d2}

    d3 = None
    if args.d3_feature:
        if args.d3_feature not in FEATURE_NAMES:
            print(f"STOP: unknown feature for D3: {args.d3_feature}")
            return 1
        if args.d3_feature in D2_FEATURES:
            print(f"STOP: {args.d3_feature} is still present in D2; ablation would be a no-op.")
            return 1
        d3_features = tuple(n for n in FEATURE_NAMES if n not in ("num_slashes", args.d3_feature))
        d3 = run_variant("d3", d3_features, X_train_full, y_train, suites, context)
        results["d3"] = d3

    # -- L. Comparison ------------------------------------------------------
    print("\n" + "=" * 84)
    print("[J] COMPARISON")
    print("=" * 84)
    print(f"\n{'metric':<30}{'R2':>12}{'D1':>12}{'D2':>12}" + (f"{'D3':>12}" if d3 else ""))
    metrics = (
        ("recall", "internal", "recall"),
        ("f1", "internal", "f1"),
        ("roc_auc", "internal", "roc_auc"),
        ("malicious_https_recall", "internal", "https"),
        ("external_domain_safe", "external", "domain_safe_rate"),
        ("legit_deep_safe", "deep_benign", "overall"),
        ("malicious_deep_recall", "deep_malicious", "overall"),
    )

    def get(result, kind, key):
        if kind == "internal":
            if key == "https":
                return result["internal"]["by_style"]["https"]["malicious_recall"]
            return result["internal"][key]
        if kind == "external":
            return result["external"][key]
        if kind == "deep_benign":
            return result["deep_benign"]["overall"]["rate"]
        if kind == "deep_malicious":
            return result["deep_malicious"]["overall"]["rate"]
        return 0.0

    for title, kind, key in metrics:
        cells = f"{get(r2, kind, key):>12.4f}{get(d1, kind, key):>12.4f}{get(d2, kind, key):>12.4f}"
        if d3:
            cells += f"{get(d3, kind, key):>12.4f}"
        print(f"{title:<30}{cells}")

    comparison = {
        "experiment": "D",
        "stage": "5.5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Test whether legitimate deep-path/query URLs can be classified safe "
            "without destroying malicious deep-URL recall, whitelisting, threshold "
            "gaming, network access, or leakage."
        ),
        "constraints": {
            "retrained_r2": False,
            "promoted_anything": False,
            "classification_threshold": 0.5,
            "ui_bands_changed": False,
            "network_used": False,
            "pandas_used": False,
            "grid_search_used": False,
        },
        "predeclared": {
            "augmentation_cap_share": AUGMENTATION_CAP_SHARE,
            "augmentation_salt": AUGMENTATION_SALT,
            "d2_ablation": "num_slashes",
            "d3_ablation": args.d3_feature,
            "acceptance_gates": [
                "overall_malicious_recall >= 0.87",
                "malicious_https_recall >= 0.90",
                "external_tranco_homepage_safe_rate >= 0.95",
                "suspicious_synthetic == 4/4",
                "legit_heldout_deep_safe_rate >= 0.90",
                "malicious_deep_recall >= 0.90",
                "no train/evaluation leakage",
            ],
            "deep_suite_definition": (
                "Kaggle TEST split, benign/malicious rows whose path is non-trivial "
                "and/or query is non-empty; browser form adds https:// only where the "
                "source row had no scheme"
            ),
        },
        "root_cause": {
            "summary": (
                "Scheme representation, not path shape. 93.2% of benign Kaggle rows are "
                "bare while 67.3% of malicious rows carry a scheme; only 2,263 benign "
                "https+deep rows existed versus 9,956 malicious."
            ),
            "measured_examples": {
                "example.com/a": 0.0302,
                "http://example.com/a": 0.9017,
                "https://example.com/a": 0.9533,
                "https://example.com": 0.0148,
                "https://example.com/": 0.9328,
            },
            "r2_feature_importances": json.loads(
                R2_METADATA.read_text(encoding="utf-8")
            )["feature_importances"],
        },
        "augmentation": spec,
        "leakage_audit": leakage,
        "pretraining_gate": pretrain,
        "distribution_statistics": distributions,
        "structure_tables": {
            "full": full_struct,
            "train_base": train_struct,
            "test": test_struct,
            "train_final": train_scheme_struct,
        },
        "deep_evaluation_sets": {
            "legit_rows": len(deep_benign_idx),
            "legit_hostnames": len(deep_benign_domains),
            "legit_by_structure": {
                group: deep_benign_structures.count(group) for group in DEEP_GROUPS
            },
            "malicious_rows": len(deep_malicious_idx),
            "malicious_by_structure": {
                group: deep_malicious_structures.count(group) for group in DEEP_GROUPS
            },
        },
        "models": {
            "r2": summary_row("R2", r2),
            "d1": summary_row("D1", d1),
            "d2": summary_row("D2", d2),
            **({"d3": summary_row("D3", d3)} if d3 else {}),
        },
        "detail": {
            name: {
                "internal": res["internal"],
                "external": res["external"],
                "deep_benign": res["deep_benign"],
                "deep_benign_original": res["deep_benign_original"],
                "deep_malicious": res["deep_malicious"],
                "deep_malicious_original": res["deep_malicious_original"],
                "sanity": res["sanity"],
                "diagnostics": res["diagnostics"],
                "gates": res["gates"],
                "feature_importances": res["importances"] if "importances" in res else
                    json.loads(R2_METADATA.read_text(encoding="utf-8"))["feature_importances"],
            }
            for name, res in results.items()
        },
    }
    write_json(COMPARISON_PATH, comparison)

    # -- M. Diagnostics / decision -----------------------------------------
    print("\n[K] Famous deep-URL diagnostics (never trained on, never whitelisted)")
    header = f"{'example':<22}{'R2':>10}{'D1':>10}{'D2':>10}" + (f"{'D3':>10}" if d3 else "")
    print(header)
    for position, row in enumerate(r2["diagnostics"]):
        line = f"{row['label']:<22}{row['score']:>10.4f}{d1['diagnostics'][position]['score']:>10.4f}" \
               f"{d2['diagnostics'][position]['score']:>10.4f}"
        if d3:
            line += f"{d3['diagnostics'][position]['score']:>10.4f}"
        print(line)

    print("\n[L] Feature importances")
    for name, res in results.items():
        top = list(res["importances"].items())[:6] if "importances" in res else []
        if top:
            print(f"    {name.upper()}: " + "  ".join(f"{k}={v:.4f}" for k, v in top))
    print(f"    uses_https -> R2={r2['diagnostics'] and json.loads(R2_METADATA.read_text(encoding='utf-8'))['feature_importances'].get('uses_https')}, "
          f"D1={d1['importances'].get('uses_https')}, D2={d2['importances'].get('uses_https')}"
          + (f", D3={d3['importances'].get('uses_https')}" if d3 else ""))

    print("\n[M] Deep-URL evaluation detail")
    for name, res in results.items():
        b = res["deep_benign"]
        m = res["deep_malicious"]
        print(f"    {name.upper()}: benign safe={b['overall']['rate']:.4f} "
              f"({b['overall']['count']:,}) | malicious recall={m['overall']['rate']:.4f} "
              f"({m['overall']['count']:,})")
        for group in DEEP_GROUPS:
            bg, mg = b["by_structure"][group], m["by_structure"][group]
            print(f"      {group:<11} benign={bg['rate'] if bg['rate'] is not None else float('nan'):.4f} "
                  f"(n={bg['count']:>6,})  malicious={mg['rate'] if mg['rate'] is not None else float('nan'):.4f} "
                  f"(n={mg['count']:>6,})")

    print("\n[N] Gate summary")
    for name, res in results.items():
        g = res["gates"]
        print(f"    {name.upper()}: {g['passed_count']}/{g['total']} "
              f"{'PASS' if g['passed'] else 'FAIL'}")
        for gate, ok in g["checks"].items():
            if not ok:
                print(f"        FAILED: {gate}")

    print(f"\nComparison artifact: {COMPARISON_PATH}")
    print("Nothing was promoted. model/fraudshield_model.pkl is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
