"""
FraudShield - Stage 6, Experiment S1
====================================

Question: can a scheme-blind, field-separated semantic lexical representation
distinguish benign from malicious deep HTTPS URLs better than the 14 scalar
counts, without destroying malicious detection?

Design (approved Option 1, fixed before running)
------------------------------------------------
Scheme-blind preprocessing: the http:// / https:// prefix is stripped before
ANY feature is computed and is never a feature. An https:// URL and the same
URL written bare therefore produce identical feature rows by construction.

Representation (candidate S1), four blocks hstacked:
  H  hostname          char_wb 3-5 grams, max_features=30,000, min_df=3
  P  path              alphanumeric word tokens, 1-2 grams, max=50,000, min_df=5
  Q  query             alphanumeric word tokens (keys and values), 1 grams,
                       max=20,000, min_df=5
  V  suspicious vocab  45 deterministic counts: the 15 canonical
                       SUSPICIOUS_KEYWORDS counted separately in host, path and
                       query (15 x 3). Not TF-IDF.

Model: LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs",
max_iter=400, random_state=42). One candidate. No tuning, no threshold change,
no grid search. If ConvergenceWarning is raised, max_iter is increased to 1000
and nothing else changes.

Training data: identical to Experiment C/D (503,901 purged Kaggle rows +
12,905 Tranco generated homepage rows = 516,806). NO augmentation.

Evaluation: identical held-out suites and gates as Experiment D, so numbers are
directly comparable. R2 reference numbers are read from the existing
Experiment D artifacts (source of truth), not recomputed.

Safety: text only. No HTTP/DNS/WHOIS/TLS/browser/external API. No pandas.

Writes only model/experiment_six/. Never touches R2, Experiment C/D, backend/.

Run:
    python experiment_six.py
    python experiment_six.py --preflight-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from backend.dataset_loader import load_dataset
from backend.feature_extractor import FEATURE_NAMES, SUSPICIOUS_KEYWORDS

OUT_DIR = PROJECT_ROOT / "model" / "experiment_six"
R2_PROVENANCE_META = PROJECT_ROOT / "model" / "model_metadata.json"
DATASET_PATH = PROJECT_ROOT / "data" / "malicious_phish.csv"
TRANCO_CSV = PROJECT_ROOT / "data" / "tranco" / "top-1m.csv"
TRANCO_SELECTION = PROJECT_ROOT / "data" / "tranco" / "selection.json"
EXPERIMENT_D_COMPARISON = PROJECT_ROOT / "model" / "experiment_d" / "comparison.json"
EXPERIMENT_D_SEPARABILITY = PROJECT_ROOT / "model" / "experiment_d" / "deep_separability.json"

RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5

# Predeclared vectoriser configuration. Not tuned.
HOST_CONFIG = dict(
    analyzer="char_wb", ngram_range=(3, 5), max_features=30_000, min_df=3,
    sublinear_tf=True, dtype=np.float32, lowercase=True,
)
PATH_CONFIG = dict(
    analyzer="word", ngram_range=(1, 2), max_features=50_000, min_df=5,
    sublinear_tf=True, dtype=np.float32, lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
)
QUERY_CONFIG = dict(
    analyzer="word", ngram_range=(1, 1), max_features=20_000, min_df=5,
    sublinear_tf=True, dtype=np.float32, lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
)
MODEL_CONFIG = dict(
    class_weight="balanced", C=1.0, solver="lbfgs", max_iter=400, random_state=RANDOM_STATE,
)

SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
LOOSE_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")
TOKEN_RE = re.compile(r"[a-z0-9]+")

DEEP_GROUPS = ("path", "query", "path+query")
SCHEMES = ("bare", "http", "https")

LEGIT_SANITY = (
    "https://www.google.com", "https://www.wikipedia.org", "https://github.com",
    "https://www.microsoft.com", "https://www.apple.com", "https://www.python.org",
    "https://www.mozilla.org", "https://www.reddit.com", "https://stackoverflow.com",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe_config(config: dict) -> dict:
    """Vectoriser configs hold numpy dtypes, which json cannot serialise."""
    return {key: (value.__name__ if isinstance(value, type) else value)
            for key, value in config.items()}


def pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:.2f}%" if whole else "0.00%"


def brand_hit(hostname: str, brands=frozenset({
    "google", "wikipedia", "github", "microsoft", "apple",
    "python", "mozilla", "reddit", "stackoverflow", "amazon",
})) -> bool:
    return any(label in brands for label in hostname.split("."))


def split_fields(url: str):
    """
    Scheme-blind field split.

    The scheme is deliberately discarded and replaced with a constant dummy so
    urlsplit can parse host/path/query. It is never returned as a feature.
    """
    text = url.strip()
    if not text:
        return None
    if SCHEME_RE.match(text):
        candidate = text
    else:
        loose = LOOSE_SCHEME_RE.match(text)
        if loose and "." not in loose.group(1):
            return None
        candidate = "http://" + text
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if not host or " " in host:
        return None
    return host, parsed.path or "", parsed.query or ""


def field_texts(url: str):
    """Return (host_text, path_token_text, query_token_text) or None."""
    fields = split_fields(url)
    if fields is None:
        return None
    host, path, query = fields
    return (
        host,
        " ".join(TOKEN_RE.findall(path.lower())),
        " ".join(TOKEN_RE.findall(query.lower())),
    )


def vocab_vector(url: str) -> list[int]:
    fields = split_fields(url)
    if fields is None:
        return [0] * (3 * len(SUSPICIOUS_KEYWORDS))
    values: list[int] = []
    for text in fields:
        lowered = text.lower()
        for keyword in SUSPICIOUS_KEYWORDS:
            values.append(lowered.count(keyword))
    return values


def structure_of(url: str) -> str | None:
    fields = split_fields(url)
    if fields is None:
        return None
    _host, path, query = fields
    has_path = path not in ("", "/")
    has_query = query != ""
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


def browser_form(url: str) -> str:
    text = url.strip()
    return text if SCHEME_RE.match(text) else "https://" + text


def is_deep(url: str) -> bool:
    return structure_of(url) in DEEP_GROUPS


# ---------------------------------------------------------------------------
# Metrics (same definitions as Experiment D)
# ---------------------------------------------------------------------------


def metrics_block(y: np.ndarray, pred: np.ndarray, score: np.ndarray, styles: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    payload = {
        "accuracy": float((pred == y).mean()),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "roc_auc": float(roc_auc_score(y, score)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": CLASSIFICATION_THRESHOLD,
        "rows": int(len(y)),
        "by_style": {},
    }
    for scheme in SCHEMES:
        bm = (styles == scheme) & (y == 0)
        mm = (styles == scheme) & (y == 1)
        payload["by_style"][scheme] = {
            "benign_count": int(bm.sum()),
            "benign_fp": int(pred[bm].sum()) if bm.any() else 0,
            "benign_fp_rate": float(pred[bm].mean()) if bm.any() else 0.0,
            "malicious_count": int(mm.sum()),
            "malicious_recall": float(pred[mm].mean()) if mm.any() else 0.0,
        }
    return payload


def deep_block(pred: np.ndarray, score: np.ndarray, structures, label: int) -> dict:
    structures = np.asarray(structures, dtype=object)
    correct = (pred == 0) if label == 0 else (pred == 1)

    def agg(mask):
        n = int(mask.sum())
        if not n:
            return {"count": 0, "rate": None, "mean_score": None}
        return {
            "count": n,
            "rate": round(float(correct[mask].mean()), 6),
            "mean_score": round(float(score[mask].mean()), 6),
        }

    return {
        "label": label,
        "rows": int(len(pred)),
        "overall": agg(np.ones(len(pred), dtype=bool)),
        "by_structure": {g: agg(structures == g) for g in DEEP_GROUPS},
    }


def external_block(pred, score, urls, domains) -> dict:
    by_domain: dict[str, list[int]] = {}
    for pos, domain in enumerate(domains):
        by_domain.setdefault(domain, []).append(pos)
    safe = [all(int(pred[p]) == 0 for p in positions) for positions in by_domain.values()]
    apex = [i for i, u in enumerate(urls) if not u.startswith("https://www.")]
    www = [i for i, u in enumerate(urls) if u.startswith("https://www.")]
    return {
        "url_count": len(urls),
        "domain_count": len(by_domain),
        "url_safe_rate": float((pred == 0).mean()),
        "domain_safe_rate": float(np.mean(safe)),
        "domains_safe": int(np.sum(safe)),
        "domains_unsafe": int(len(safe) - np.sum(safe)),
        "apex_safe_rate": float((pred[apex] == 0).mean()) if apex else 0.0,
        "www_safe_rate": float((pred[www] == 0).mean()) if www else 0.0,
        "score_mean": float(score.mean()),
    }


def separability(benign_scores: np.ndarray, malicious_scores: np.ndarray) -> dict:
    benign_scores = np.asarray(benign_scores, dtype=np.float64)
    malicious_scores = np.asarray(malicious_scores, dtype=np.float64)
    if benign_scores.size == 0 or malicious_scores.size == 0:
        return {"roc_auc": None, "best_balanced_safe": None,
                "best_balanced_recall": None, "both_0.90_reachable": False,
                "note": "empty class"}
    scores = np.concatenate([benign_scores, malicious_scores])
    labels = np.concatenate([
        np.zeros(len(benign_scores), dtype=np.int8),
        np.ones(len(malicious_scores), dtype=np.int8),
    ])
    auc = float(roc_auc_score(labels, scores))
    # Exact sweep over every observed score in O(n log n): safe = P(benign < t),
    # recall = P(malicious >= t). Same definition as Experiment D, just vectorised.
    thresholds = np.unique(scores)
    benign_sorted = np.sort(benign_scores)
    malicious_sorted = np.sort(malicious_scores)
    safe = np.searchsorted(benign_sorted, thresholds, side="left") / len(benign_sorted)
    recall = (len(malicious_sorted)
              - np.searchsorted(malicious_sorted, thresholds, side="left")) / len(malicious_sorted)
    balanced = np.minimum(safe, recall)
    best_index = int(np.argmax(balanced))
    return {
        "roc_auc": round(auc, 6),
        "best_balanced_safe": round(float(safe[best_index]), 6),
        "best_balanced_recall": round(float(recall[best_index]), 6),
        "best_balanced_min": round(float(balanced[best_index]), 6),
        "best_balanced_threshold": round(float(thresholds[best_index]), 6),
        "both_0.90_reachable": bool(balanced[best_index] >= 0.90),
        "note": "Threshold sweep is diagnostic only; deployed threshold stays 0.5.",
    }


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------


class SemanticURLModel:
    """TF-IDF field blocks + suspicious-vocabulary counts + logistic regression."""

    def __init__(self, host_vec, path_vec, query_vec, classifier, config: dict):
        self.host_vec = host_vec
        self.path_vec = path_vec
        self.query_vec = query_vec
        self.classifier = classifier
        self.config = config

    def transform(self, urls: list[str]):
        hosts, paths, queries = [], [], []
        vocab_rows = []
        for url in urls:
            texts = field_texts(url)
            if texts is None:
                hosts.append("")
                paths.append("")
                queries.append("")
            else:
                hosts.append(texts[0])
                paths.append(texts[1])
                queries.append(texts[2])
            vocab_rows.append(vocab_vector(url))
        X = hstack(
            [
                self.host_vec.transform(hosts),
                self.path_vec.transform(paths),
                self.query_vec.transform(queries),
                csr_matrix(np.asarray(vocab_rows, dtype=np.float32)),
            ],
            format="csr",
        )
        return X

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(urls))

    def predict(self, urls: list[str]) -> np.ndarray:
        return self.classifier.predict(self.transform(urls))


def block_bounds(host_vec, path_vec, query_vec, total_features: int) -> dict[str, tuple[int, int]]:
    host_end = len(host_vec.vocabulary_)
    path_end = host_end + len(path_vec.vocabulary_)
    query_end = path_end + len(query_vec.vocabulary_)
    return {
        "host": (0, host_end),
        "path": (host_end, path_end),
        "query": (path_end, query_end),
        "vocab": (query_end, total_features),
    }


def block_contributions(matrix, classifier, bounds: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    """Exact additive logit contribution of each feature block (linear model)."""
    coef = classifier.coef_.ravel()
    parts = {name: np.asarray(matrix[:, low:high] @ coef[low:high]).ravel()
             for name, (low, high) in bounds.items()}
    parts["intercept"] = np.full(matrix.shape[0], float(classifier.intercept_[0]))
    parts["total"] = sum(parts.values())
    return parts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudShield Stage 6 Experiment S1")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run audits and feature-equality checks, then stop before fitting")
    parser.add_argument("--report-only", action="store_true",
                        help="load the already-fitted s1.pkl and regenerate artifacts; never trains")
    args = parser.parse_args(argv)
    if args.preflight_only and args.report_only:
        parser.error("choose one of --preflight-only / --report-only, not both")

    print("=" * 86)
    print("FraudShield Stage 6 - Experiment S1 (scheme-blind semantic lexical model)")
    print("=" * 86)
    print("Text-only processing. No HTTP/DNS/WHOIS/TLS/browser/external API access.")
    print("This run writes only model/experiment_six/.")

    for required in (DATASET_PATH, TRANCO_CSV, TRANCO_SELECTION,
                     EXPERIMENT_D_COMPARISON, EXPERIMENT_D_SEPARABILITY):
        if not required.is_file():
            print(f"STOP: required file missing: {required}")
            return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- A. Input verification ---------------------------------------------
    print("\n[A] Input verification (read-only)")
    r2_meta = json.loads(R2_PROVENANCE_META.read_text(encoding="utf-8"))
    dataset_hash = file_sha256(DATASET_PATH)
    tranco_hash = file_sha256(TRANCO_CSV)
    if dataset_hash != r2_meta["provenance"]["kaggle_sha256"]:
        print("STOP: Kaggle CSV differs from the promoted-model provenance hash.")
        return 1
    if tranco_hash != r2_meta["provenance"]["tranco_sha256"]:
        print("STOP: Tranco CSV differs from the promoted-model provenance hash.")
        return 1
    print(f"    Kaggle SHA-256 : {dataset_hash}")
    print(f"    Tranco SHA-256 : {tranco_hash}")
    print(f"    Promoted R2    : {file_sha256(PROJECT_ROOT / 'model' / 'fraudshield_model.pkl')}")

    # -- B. Data + split ----------------------------------------------------
    print("\n[B] Loading Kaggle and reproducing the Stage 4 / Experiment C split...")
    started = time.perf_counter()
    bundle = load_dataset(DATASET_PATH, label_map=train.SOURCE_TO_BINARY,
                          url_column="url", label_column="type")
    urls = bundle.urls
    y = np.asarray(bundle.y, dtype=np.int8)
    del bundle
    print(f"    Usable rows: {len(urls):,} ({time.perf_counter() - started:.1f}s)")

    train_idx, test_idx, groups = train.split_by_hostname(urls, y)
    if len(train_idx) != r2_meta["provenance"]["kaggle_train_rows_before_purge"]:
        print("STOP: split size differs from Experiment C.")
        return 1
    keep = np.asarray([not brand_hit(groups[i]) for i in train_idx], dtype=bool)
    train_purged = train_idx[keep]
    if len(train_purged) != r2_meta["provenance"]["kaggle_train_rows_after_purge"]:
        print("STOP: purged training size differs from Experiment C.")
        return 1
    print(f"    train {len(train_purged):,} / test {len(test_idx):,} (identical to C/D)")

    selection = json.loads(TRANCO_SELECTION.read_text(encoding="utf-8"))
    tranco_train_urls = [f"https://{d}" for d in selection["train_domains"]]
    tranco_train_urls += [f"https://www.{d}" for d in selection["www_train_domains"]]
    tranco_holdout_urls: list[str] = []
    tranco_holdout_domains: list[str] = []
    for domain in selection["holdout_domains"]:
        tranco_holdout_urls += [f"https://{domain}", f"https://www.{domain}"]
        tranco_holdout_domains += [domain, domain]

    # -- C. Evaluation index sets (identical to Experiment D) --------------
    print("\n[C] Building held-out evaluation sets (identical index sets to Experiment D)")
    benign_idx = [i for i in test_idx if y[i] == 0 and is_deep(urls[i])]
    malicious_idx = [i for i in test_idx if y[i] == 1 and is_deep(urls[i])]
    benign_urls = [browser_form(urls[i]) for i in benign_idx]
    malicious_urls = [browser_form(urls[i]) for i in malicious_idx]
    benign_struct = [structure_of(urls[i]) for i in benign_idx]
    malicious_struct = [structure_of(urls[i]) for i in malicious_idx]
    benign_https_idx = [j for j, u in enumerate(benign_urls) if u.startswith("https://")]
    malicious_https_idx = [j for j, u in enumerate(malicious_urls) if u.startswith("https://")]
    benign_domains = {groups[i] for i in benign_idx}

    print(f"    legitimate deep : {len(benign_urls):,} rows / {len(benign_domains):,} hostnames")
    print(f"    malicious deep  : {len(malicious_urls):,} rows")
    print(f"    https-only deep : benign {len(benign_https_idx):,} / malicious {len(malicious_https_idx):,}")

    # Subtype map (from the raw CSV; labels only, never features).
    url_types: dict[str, set[str]] = {}
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            u = (row.get("url") or "").strip()
            t = (row.get("type") or "").strip().lower()
            if u and t:
                url_types.setdefault(u, set()).add(t)
    malicious_subtype = []
    for i in malicious_idx:
        types = url_types.get(urls[i], set())
        malicious_subtype.append("|".join(sorted(types)) if types else "unknown")
    subtype_counts = {k: malicious_subtype.count(k) for k in sorted(set(malicious_subtype))}
    print(f"    malicious subtypes: {subtype_counts}")

    # -- D. Pre-flight audits ----------------------------------------------
    print("\n[D] PRE-FLIGHT AUDITS (must pass before any fitting)")
    failures: list[str] = []

    train_urls_set = {urls[i] for i in train_purged}
    train_hosts = {groups[i] for i in train_purged} | {
        (split_fields(u) or ("",))[0] for u in tranco_train_urls
    }
    test_hosts = {groups[i] for i in test_idx}
    test_host_set = set(test_hosts)
    test_idx_set = set(int(i) for i in test_idx)
    malicious_domains = {groups[i] for i in malicious_idx}

    checks = {
        "kaggle_train_test_hostname_overlap_zero": len(train_hosts & test_hosts),
        "deep_eval_hostnames_in_train_zero": len(benign_domains & train_hosts),
        "deep_malicious_hostnames_in_train_zero": len(malicious_domains & train_hosts),
        "deep_eval_rows_come_from_test_split": int(
            set(int(i) for i in benign_idx).issubset(test_idx_set)
            and set(int(i) for i in malicious_idx).issubset(test_idx_set)),
        "famous_diagnostic_urls_in_train_zero": len(
            {u for _l, u in FAMOUS_DEEP + EXTRA_DIAGNOSTICS} & train_urls_set),
        "sanity_brand_hosts_in_train_zero": sum(1 for h in train_hosts if brand_hit(h)),
        "tranco_holdout_in_train_zero": len(
            set(selection["holdout_domains"]) & set(selection["train_domains"])),
        "no_augmentation_rows": 0,
        "vocab_will_be_fit_on_train_only": 1,
    }
    expect_one = {"deep_eval_rows_come_from_test_split", "vocab_will_be_fit_on_train_only"}
    for name, value in checks.items():
        ok = value == (1 if name in expect_one else 0)
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {value}")
        if not ok:
            failures.append(name)

    # Scheme-blind equality: identical features for original vs browser form.
    sample_positions = list(range(0, len(urls), max(1, len(urls) // 5000)))[:5000]
    field_equal = 0
    vocab_equal = 0
    for pos in sample_positions:
        original = urls[pos]
        derived = browser_form(original)
        if field_texts(original) == field_texts(derived):
            field_equal += 1
        if vocab_vector(original) == vocab_vector(derived):
            vocab_equal += 1
    equality_ok = field_equal == len(sample_positions) == vocab_equal
    print(f"    [{'PASS' if equality_ok else 'FAIL'}] scheme-blind field equality: "
          f"{field_equal}/{len(sample_positions)} field, {vocab_equal}/{len(sample_positions)} vocab")
    if not equality_ok:
        failures.append("scheme_blind_field_equality")
    checks["scheme_blind_sample_size"] = len(sample_positions)

    if failures:
        print(f"\nSTOP: pre-flight audits failed: {failures}. Nothing was fitted.")
        write_json(OUT_DIR / "preflight_failed.json", {
            "failures": failures, "checks": checks,
            "checked_expecting_one": sorted(expect_one),
        })
        return 1
    print("    All pre-flight audits passed.")

    if args.preflight_only:
        print("\n--preflight-only requested; stopping before vectoriser/classifier fitting.")
        return 0

    model_path = OUT_DIR / "s1.pkl"

    if args.report_only:
        # -- E. Report-only: load the already-fitted bundle. NO TRAINING. ---
        print("\n[E] REPORT-ONLY: loading the already-fitted bundle (no fitting, no training)")
        if not model_path.is_file():
            print(f"STOP: fitted bundle not found: {model_path}")
            print("      Cannot regenerate artifacts without a fitted model. Nothing was trained.")
            return 1
        bundle = joblib.load(model_path)
        host_vec = bundle["host_vec"]
        path_vec = bundle["path_vec"]
        query_vec = bundle["query_vec"]
        classifier = bundle["classifier"]
        used_max_iter = int(bundle["config"]["max_iter"])
        size_mb = model_path.stat().st_size / (1024 * 1024)
        vectoriser_seconds = 0.0
        transform_seconds = 0.0
        model_seconds = 0.0
        model = SemanticURLModel(host_vec, path_vec, query_vec, classifier, bundle["config"])
        print(f"    bundle        : {model_path.name}  sha256={file_sha256(model_path)}")
        print(f"    vocab sizes   : host={len(host_vec.vocabulary_):,}  "
              f"path={len(path_vec.vocabulary_):,}  query={len(query_vec.vocabulary_):,}  "
              f"vocab_block={3 * len(SUSPICIOUS_KEYWORDS)}")
        print(f"    hyperparams   : {bundle['config']}  iterations={int(classifier.n_iter_[0])}")
        print(f"    model size    : {size_mb:.2f} MB")

        # The loaded bundle must still satisfy the scheme-blind property.
        equality_positions = list(range(0, len(urls), max(1, len(urls) // 200)))[:200]
        pairs = [(urls[p], browser_form(urls[p])) for p in equality_positions]
        original_rows = model.transform([a for a, _ in pairs])
        derived_rows = model.transform([b for _, b in pairs])
        post_fit_equal = (original_rows != derived_rows).nnz == 0
        print(f"    [{'PASS' if post_fit_equal else 'FAIL'}] post-fit feature equality "
              f"(original vs browser form, {len(pairs)} pairs)")
        if not post_fit_equal:
            print("STOP: scheme-blind equality failed after loading.")
            return 1

    else:
        # -- E. Fit vectorisers on TRAIN ONLY ------------------------------
        print("\n[E] Fitting vectorisers on training text only")
        train_jobs = [(urls[i], y[i]) for i in train_purged] + [
            (u, 0) for u in tranco_train_urls
        ]
        y_train = np.asarray([label for _u, label in train_jobs], dtype=np.int8)
        train_hosts_text, train_paths_text, train_queries_text = [], [], []
        train_vocab = []
        for url, _label in train_jobs:
            texts = field_texts(url)
            if texts is None:
                train_hosts_text.append("")
                train_paths_text.append("")
                train_queries_text.append("")
            else:
                train_hosts_text.append(texts[0])
                train_paths_text.append(texts[1])
                train_queries_text.append(texts[2])
            train_vocab.append(vocab_vector(url))
        print(f"    training rows : {len(train_jobs):,}  "
              f"(benign {int((y_train == 0).sum()):,} / malicious {int((y_train == 1).sum()):,})")

        t0 = time.perf_counter()
        host_vec = TfidfVectorizer(**HOST_CONFIG).fit(train_hosts_text)
        path_vec = TfidfVectorizer(**PATH_CONFIG).fit(train_paths_text)
        query_vec = TfidfVectorizer(**QUERY_CONFIG).fit(train_queries_text)
        vectoriser_seconds = time.perf_counter() - t0
        print(f"    vocab sizes   : host={len(host_vec.vocabulary_):,}  "
              f"path={len(path_vec.vocabulary_):,}  query={len(query_vec.vocabulary_):,}  "
              f"vocab_block={3 * len(SUSPICIOUS_KEYWORDS)}  "
              f"total={len(host_vec.vocabulary_) + len(path_vec.vocabulary_) + len(query_vec.vocabulary_) + 3 * len(SUSPICIOUS_KEYWORDS):,}")

        t0 = time.perf_counter()
        X_train = hstack([
            host_vec.transform(train_hosts_text),
            path_vec.transform(train_paths_text),
            query_vec.transform(train_queries_text),
            csr_matrix(np.asarray(train_vocab, dtype=np.float32)),
        ], format="csr")
        transform_seconds = time.perf_counter() - t0
        print(f"    sparse matrix : {X_train.shape[0]:,} x {X_train.shape[1]:,}  "
              f"nnz={X_train.nnz:,} ({X_train.nnz / X_train.shape[0]:.1f} per row)  "
              f"{X_train.data.nbytes / 1e6:.0f} MB data")
        del train_hosts_text, train_paths_text, train_queries_text, train_vocab, train_jobs

        # Post-fit feature equality on a sample (strong version of the pre-flight check).
        equality_positions = list(range(0, len(urls), max(1, len(urls) // 200)))[:200]
        pairs = [(urls[p], browser_form(urls[p])) for p in equality_positions]
        original_rows = hstack([
            host_vec.transform([field_texts(u)[0] if field_texts(u) else "" for u, _ in pairs]),
            path_vec.transform([field_texts(u)[1] if field_texts(u) else "" for u, _ in pairs]),
            query_vec.transform([field_texts(u)[2] if field_texts(u) else "" for u, _ in pairs]),
            csr_matrix(np.asarray([vocab_vector(u) for u, _ in pairs], dtype=np.float32)),
        ], format="csr")
        derived_rows = hstack([
            host_vec.transform([field_texts(u)[0] if field_texts(u) else "" for _, u in pairs]),
            path_vec.transform([field_texts(u)[1] if field_texts(u) else "" for _, u in pairs]),
            query_vec.transform([field_texts(u)[2] if field_texts(u) else "" for _, u in pairs]),
            csr_matrix(np.asarray([vocab_vector(u) for _, u in pairs], dtype=np.float32)),
        ], format="csr")
        post_fit_equal = (original_rows != derived_rows).nnz == 0
        print(f"    [{'PASS' if post_fit_equal else 'FAIL'}] post-fit feature equality "
              f"(original vs browser form, {len(pairs)} pairs)")
        if not post_fit_equal:
            print("STOP: scheme-blind equality failed after vectorisation.")
            return 1

        # -- F. Fit logistic regression ------------------------------------
        print("\n[F] Fitting LogisticRegression (one candidate, no tuning)")
        used_max_iter = MODEL_CONFIG["max_iter"]
        convergence_warning = False
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier = LogisticRegression(**MODEL_CONFIG).fit(X_train, y_train)
            convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if convergence_warning:
            print(f"    ConvergenceWarning at max_iter={used_max_iter}; "
                  f"refitting with max_iter=1000 (predeclared, design unchanged)")
            used_max_iter = 1000
            classifier = LogisticRegression(
                **{**MODEL_CONFIG, "max_iter": 1000}
            ).fit(X_train, y_train)
        model_seconds = time.perf_counter() - t0
        print(f"    fit time      : {model_seconds:.1f}s  max_iter used={used_max_iter}  "
              f"iterations={int(classifier.n_iter_[0])}")
        del X_train

        model = SemanticURLModel(host_vec, path_vec, query_vec, classifier, MODEL_CONFIG)
        joblib.dump({
            "host_vec": host_vec,
            "path_vec": path_vec,
            "query_vec": query_vec,
            "classifier": classifier,
            "config": {**MODEL_CONFIG, "max_iter": used_max_iter},
            "vectoriser_config": {"host": HOST_CONFIG, "path": PATH_CONFIG, "query": QUERY_CONFIG},
            "representation": "scheme-blind field-separated TF-IDF + suspicious vocabulary counts",
        }, model_path, compress=3)
        size_mb = model_path.stat().st_size / (1024 * 1024)

        reloaded = joblib.load(model_path)
        probe_urls = ["https://www.google.com/search?q=fraud", "http://secure-login.example/verify/account"]
        reloaded_model = SemanticURLModel(
            reloaded["host_vec"], reloaded["path_vec"], reloaded["query_vec"], reloaded["classifier"],
            reloaded["config"],
        )
        if not np.allclose(reloaded_model.predict_proba(probe_urls), model.predict_proba(probe_urls)):
            print("STOP: reloaded bundle does not reproduce predictions.")
            return 1
        print(f"    model size    : {size_mb:.2f} MB  (artifact reload verified)")

    # -- G. Evaluate --------------------------------------------------------
    print("\n[G] Evaluation (identical suites and threshold as Experiment D)")

    test_urls = [urls[i] for i in test_idx]
    test_styles = np.asarray([scheme_of(u) for u in test_urls], dtype=object)
    t0 = time.perf_counter()
    test_score = model.predict_proba(test_urls)[:, 1]
    test_pred = (test_score >= CLASSIFICATION_THRESHOLD).astype(np.int8)
    eval_seconds = time.perf_counter() - t0
    internal = metrics_block(y[test_idx], test_pred, test_score, test_styles)
    print(f"    internal test : acc={internal['accuracy']:.4f} prec={internal['precision']:.4f} "
          f"rec={internal['recall']:.4f} f1={internal['f1']:.4f} auc={internal['roc_auc']:.4f}")

    ext_score = model.predict_proba(tranco_holdout_urls)[:, 1]
    ext_pred = (ext_score >= CLASSIFICATION_THRESHOLD).astype(np.int8)
    external = external_block(ext_pred, ext_score, tranco_holdout_urls, tranco_holdout_domains)
    print(f"    Tranco domain safe: {external['domain_safe_rate']:.4f}")

    # -- G2. Measured evidence for the Tranco result ------------------------
    print("\n[G2] Tranco holdout: measured evidence (no model change, no tuning)")
    n_features = (len(host_vec.vocabulary_) + len(path_vec.vocabulary_)
                  + len(query_vec.vocabulary_) + 3 * len(SUSPICIOUS_KEYWORDS))
    bounds = block_bounds(host_vec, path_vec, query_vec, n_features)
    ext_matrix = model.transform(tranco_holdout_urls)
    parts = block_contributions(ext_matrix, classifier, bounds)
    s1_flag = ext_pred.astype(bool)

    # R2 scored on the exact same 4,000 holdout URLs, read-only, via the
    # canonical prediction interface. Nothing is written to R2.
    from backend.predictor import FraudShieldPredictor  # local import: read-only audit

    r2_predictor = FraudShieldPredictor()
    r2_results = [r2_predictor.predict(u) for u in tranco_holdout_urls]
    r2_scores = np.asarray([r["model_score"] for r in r2_results], dtype=np.float64)
    r2_pred = np.asarray([r["prediction"] for r in r2_results], dtype=np.int8)
    domain_positions: dict[str, list[int]] = {}
    for position, domain in enumerate(tranco_holdout_domains):
        domain_positions.setdefault(domain, []).append(position)
    r2_domain_safe = float(np.mean([
        all(int(r2_pred[p]) == 0 for p in positions)
        for positions in domain_positions.values()
    ]))

    # Host n-gram coverage: how much of each hostname is inside the train vocabulary.
    host_analyzer = host_vec.build_analyzer()
    host_vocabulary = host_vec.vocabulary_

    def coverage_of(sample_urls):
        shares = []
        for u in sample_urls:
            fields = split_fields(u)
            if fields is None:
                continue
            grams = host_analyzer(fields[0])
            if not grams:
                continue
            known = sum(1 for gram in grams if gram in host_vocabulary)
            shares.append(known / len(grams))
        return np.asarray(shares, dtype=np.float64)

    tranco_coverage = coverage_of(tranco_holdout_urls)
    benign_coverage = coverage_of(benign_urls[:8000])
    malicious_coverage = coverage_of(malicious_urls[:8000])

    # How often the dataset writes a "www." hostname, by TRAIN class. This is a
    # measured convention statistic (no fitting, no evaluation leakage).
    www_counts = {"benign_with_www": 0, "benign_total": 0,
                  "malicious_with_www": 0, "malicious_total": 0}
    for index in train_purged:
        fields = split_fields(urls[index])
        if fields is None:
            continue
        has_www = fields[0].startswith("www.")
        if y[index] == 1:
            www_counts["malicious_total"] += 1
            www_counts["malicious_with_www"] += int(has_www)
        else:
            www_counts["benign_total"] += 1
            www_counts["benign_with_www"] += int(has_www)
    www_counts["tranco_train_with_www"] = int(sum(
        1 for u in tranco_train_urls if (split_fields(u) or ("",))[0].startswith("www.")))
    www_counts["benign_share"] = round(
        www_counts["benign_with_www"] / www_counts["benign_total"], 6)
    www_counts["malicious_share"] = round(
        www_counts["malicious_with_www"] / www_counts["malicious_total"], 6)

    coef_full = classifier.coef_.ravel()
    all_names = np.concatenate([
        host_vec.get_feature_names_out(),
        path_vec.get_feature_names_out(),
        query_vec.get_feature_names_out(),
        np.asarray([f"kw:{field}:{kw}" for field in ("host", "path", "query")
                    for kw in SUSPICIOUS_KEYWORDS], dtype=object),
    ])
    flagged_matrix = ext_matrix[s1_flag]
    aggregated = (np.asarray(flagged_matrix.sum(axis=0)).ravel() * coef_full
                  if flagged_matrix.shape[0] else np.zeros(len(coef_full)))
    top_index = np.argsort(aggregated)[::-1][:10]
    n_flagged = int(s1_flag.sum())
    top_features = [
        {"feature": str(all_names[i]),
         "mean_logit_per_flagged_row": round(float(aggregated[i] / n_flagged), 6)}
        for i in top_index if aggregated[i] > 0
    ]

    tranco_diagnosis = {
        "note": (
            "All values below are measured on the existing fitted S1 bundle and the "
            "unchanged 4,000-URL Tranco holdout. No retraining, no threshold change."
        ),
        "s1": {
            "domain_safe_rate": external["domain_safe_rate"],
            "url_safe_rate": external["url_safe_rate"],
            "mean_score": round(float(ext_score.mean()), 6),
            "median_score": round(float(np.median(ext_score)), 6),
            "p10_score": round(float(np.quantile(ext_score, 0.10)), 6),
            "p90_score": round(float(np.quantile(ext_score, 0.90)), 6),
            "flagged_urls": n_flagged,
            "flagged_share": round(float(s1_flag.mean()), 6),
            "share_of_flagged_above_0.65": round(float((ext_score[s1_flag] >= 0.65).mean()), 6)
                                           if n_flagged else None,
        },
        "r2_same_urls": {
            "domain_safe_rate": round(r2_domain_safe, 6),
            "url_safe_rate": round(float((r2_pred == 0).mean()), 6),
            "mean_score": round(float(r2_scores.mean()), 6),
            "median_score": round(float(np.median(r2_scores)), 6),
        },
        "agreement": {
            "both_safe": int(((r2_pred == 0) & ~s1_flag).sum()),
            "r2_safe_s1_malicious": int(((r2_pred == 0) & s1_flag).sum()),
            "r2_malicious_s1_safe": int(((r2_pred == 1) & ~s1_flag).sum()),
            "both_malicious": int(((r2_pred == 1) & s1_flag).sum()),
        },
        "block_logit_mean_flagged": {
            name: round(float(parts[name][s1_flag].mean()), 6) if n_flagged else None
            for name in ("host", "path", "query", "vocab", "intercept", "total")
        },
        "block_logit_mean_safe": {
            name: round(float(parts[name][~s1_flag].mean()), 6) if (~s1_flag).any() else None
            for name in ("host", "path", "query", "vocab", "intercept", "total")
        },
        "path_block_is_zero_on_every_holdout_url": bool(np.all(parts["path"] == 0.0)),
        "query_block_is_zero_on_every_holdout_url": bool(np.all(parts["query"] == 0.0)),
        "top_features_driving_flags": top_features,
        "host_ngram_coverage": {
            "definition": "share of a hostname's char_wb 3-5 grams that exist in the training vocabulary",
            "tranco_holdout_mean": round(float(tranco_coverage.mean()), 6),
            "deep_benign_sample_mean": round(float(benign_coverage.mean()), 6),
            "deep_malicious_sample_mean": round(float(malicious_coverage.mean()), 6),
            "sample_sizes": {"tranco_holdout": int(len(tranco_coverage)),
                             "deep_benign": int(len(benign_coverage)),
                             "deep_malicious": int(len(malicious_coverage))},
        },
        "www_prefix_prevalence_in_training": www_counts,
    }
    print(f"    S1 mean score={tranco_diagnosis['s1']['mean_score']:.4f} "
          f"median={tranco_diagnosis['s1']['median_score']:.4f} "
          f"flagged={n_flagged:,}/{len(ext_score):,}")
    print(f"    R2 same URLs: domain safe={r2_domain_safe:.4f} "
          f"mean score={tranco_diagnosis['r2_same_urls']['mean_score']:.4f}")
    print(f"    logit contribution on flagged rows: {tranco_diagnosis['block_logit_mean_flagged']}")
    print(f"    host n-gram coverage: tranco={tranco_diagnosis['host_ngram_coverage']['tranco_holdout_mean']:.4f} "
          f"deep_benign={tranco_diagnosis['host_ngram_coverage']['deep_benign_sample_mean']:.4f} "
          f"deep_malicious={tranco_diagnosis['host_ngram_coverage']['deep_malicious_sample_mean']:.4f}")
    for row in top_features[:5]:
        print(f"      +{row['mean_logit_per_flagged_row']:.4f}  {row['feature']}")
    print(f"    www. hostname share in TRAIN: benign {www_counts['benign_share']:.4f} "
          f"({www_counts['benign_with_www']:,}/{www_counts['benign_total']:,}) vs malicious "
          f"{www_counts['malicious_share']:.4f} "
          f"({www_counts['malicious_with_www']:,}/{www_counts['malicious_total']:,})")

    benign_score = model.predict_proba(benign_urls)[:, 1]
    benign_pred = (benign_score >= CLASSIFICATION_THRESHOLD).astype(np.int8)
    deep_benign = deep_block(benign_pred, benign_score, benign_struct, 0)
    print(f"    legit deep safe   : {deep_benign['overall']['rate']:.4f}")

    malicious_score = model.predict_proba(malicious_urls)[:, 1]
    malicious_pred = (malicious_score >= CLASSIFICATION_THRESHOLD).astype(np.int8)
    deep_malicious = deep_block(malicious_pred, malicious_score, malicious_struct, 1)
    print(f"    malicious deep rec: {deep_malicious['overall']['rate']:.4f}")

    # Per-subtype malicious deep recall.
    subtype_results = {}
    for subtype in sorted(set(malicious_subtype)):
        mask = np.asarray([s == subtype for s in malicious_subtype], dtype=bool)
        subtype_results[subtype] = {
            "count": int(mask.sum()),
            "recall": round(float(malicious_pred[mask].mean()), 6),
            "mean_score": round(float(malicious_score[mask].mean()), 6),
        }
    for subtype, payload in subtype_results.items():
        print(f"      {subtype:<14} n={payload['count']:>6,} recall={payload['recall']:.4f}")

    https_sep = separability(benign_score[benign_https_idx], malicious_score[malicious_https_idx])
    all_sep = separability(benign_score, malicious_score)
    print(f"    https-only deep ROC-AUC: {https_sep['roc_auc']:.4f}  "
          f"(R2 baseline 0.7572)  best-balanced {https_sep['best_balanced_safe']:.3f}/"
          f"{https_sep['best_balanced_recall']:.3f}")
    print(f"    all-deep ROC-AUC       : {all_sep['roc_auc']:.4f}")

    # Sanity + diagnostics.
    sanity_rows = []
    for kind, suite in (("legit", LEGIT_SANITY), ("suspicious", SUSPICIOUS_SANITY)):
        scores = model.predict_proba(list(suite))[:, 1]
        for url, s in zip(suite, scores):
            sanity_rows.append({"kind": kind, "url": url, "score": round(float(s), 6),
                                "prediction": int(s >= CLASSIFICATION_THRESHOLD)})
    legit_safe = sum(1 for r in sanity_rows if r["kind"] == "legit" and r["prediction"] == 0)
    suspicious_hit = sum(1 for r in sanity_rows if r["kind"] == "suspicious" and r["prediction"] == 1)

    diag_rows = []
    for label, url in FAMOUS_DEEP + EXTRA_DIAGNOSTICS:
        s = float(model.predict_proba([url])[0, 1])
        diag_rows.append({
            "label": label, "url": url,
            "kind": "famous" if (label, url) in FAMOUS_DEEP else "extra",
            "score": round(s, 6), "prediction": int(s >= CLASSIFICATION_THRESHOLD),
        })
    print(f"    sanity        : legit {legit_safe}/10 safe, suspicious {suspicious_hit}/4 flagged")

    # Inference latency (single URL, end-to-end including parsing + vectorising).
    latency_urls = list(LEGIT_SANITY) + list(SUSPICIOUS_SANITY) + benign_urls[:20]
    timings = []
    for index in range(200):
        u = latency_urls[index % len(latency_urls)]
        t = time.perf_counter()
        model.predict_proba([u])
        timings.append((time.perf_counter() - t) * 1000.0)
    ordered = sorted(timings)
    latency = {
        "iterations": len(timings),
        "mean_ms": round(statistics.fmean(timings), 3),
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(ordered[int(round(0.95 * (len(ordered) - 1)))], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }
    print(f"    inference     : mean={latency['mean_ms']:.2f} ms  median={latency['median_ms']:.2f} ms  "
          f"p95={latency['p95_ms']:.2f} ms")

    # Residual scheme recoverability probe (diagnostic only, small sample).
    probe_rng = np.random.default_rng(RANDOM_STATE)
    in_train = np.zeros(len(urls), dtype=bool)
    in_train[train_purged] = True
    probe_pool = probe_rng.choice(len(urls), size=min(60_000, len(urls)), replace=False)
    probe_train = [int(p) for p in probe_pool if in_train[p]][:25_000]
    probe_test = probe_rng.choice(len(test_idx), size=min(15_000, len(test_idx)), replace=False)
    probe_test_positions = [int(test_idx[p]) for p in probe_test]
    probe_y_train = np.asarray([1 if scheme_of(urls[p]) == "https" else 0 for p in probe_train])
    probe_y_test = np.asarray([1 if scheme_of(urls[p]) == "https" else 0 for p in probe_test_positions])
    scheme_probe = None
    if probe_y_train.min() != probe_y_train.max() and probe_y_test.min() != probe_y_test.max():
        Xp_train = model.transform([urls[p] for p in probe_train])
        Xp_test = model.transform([urls[p] for p in probe_test_positions])
        probe_clf = LogisticRegression(max_iter=100, solver="lbfgs", random_state=RANDOM_STATE)
        probe_clf.fit(Xp_train, probe_y_train)
        probe_scores = probe_clf.predict_proba(Xp_test)[:, 1]
        scheme_probe = {
            "purpose": "diagnostic only: can the semantic features reconstruct the written scheme?",
            "train_rows": len(probe_train),
            "test_rows": len(probe_test_positions),
            "https_rate_train": round(float(probe_y_train.mean()), 6),
            "auc": round(float(roc_auc_score(probe_y_test, probe_scores)), 6),
        }
        print(f"    scheme probe  : AUC={scheme_probe['auc']:.4f} "
              f"(diagnostic only; https rate {scheme_probe['https_rate_train']:.3f})")
        del Xp_train, Xp_test, probe_clf, probe_scores

    # -- H. Gates -----------------------------------------------------------
    gate_checks = {
        "overall_malicious_recall_ge_0.87": internal["recall"] >= 0.87,
        "malicious_https_recall_ge_0.90": internal["by_style"]["https"]["malicious_recall"] >= 0.90,
        "external_tranco_homepage_safe_ge_0.95": external["domain_safe_rate"] >= 0.95,
        "suspicious_synthetic_4_of_4": suspicious_hit == 4,
        "legit_deep_safe_ge_0.90": deep_benign["overall"]["rate"] >= 0.90,
        "malicious_deep_recall_ge_0.90": deep_malicious["overall"]["rate"] >= 0.90,
        "no_leakage": True,
    }
    gates = {
        "checks": gate_checks,
        "passed_count": sum(1 for v in gate_checks.values() if v),
        "total": len(gate_checks),
        "passed": all(gate_checks.values()),
    }
    print("\n[H] Acceptance gates (predeclared, unchanged)")
    for name, ok in gate_checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    {gates['passed_count']}/{gates['total']} gates passed")

    # -- H2. Degeneracy diagnostic -----------------------------------------
    print("\n[H2] Degeneracy diagnostic: behaviour when no lexical evidence exists")
    intercept = float(classifier.intercept_[0])
    empty_score = float(1.0 / (1.0 + np.exp(-intercept)))

    def zero_row_profile(subset_urls, subset_scores):
        matrix = model.transform(subset_urls)
        zero = np.asarray(matrix.getnnz(axis=1) == 0)
        return {
            "rows": int(len(subset_urls)),
            "rows_with_no_features": int(zero.sum()),
            "share_with_no_features": round(float(zero.mean()), 6),
            "mean_score_all": round(float(subset_scores.mean()), 6),
            "mean_score_no_features": (round(float(subset_scores[zero].mean()), 6)
                                       if bool(zero.any()) else None),
            "mean_score_with_features": (round(float(subset_scores[~zero].mean()), 6)
                                         if bool((~zero).any()) else None),
        }

    degeneracy = {
        "note": (
            "Diagnostic only, no model change. A URL whose host/path/query tokens are all "
            "outside the training vocabulary produces an all-zero feature row, so its score "
            "is sigmoid(intercept). With class_weight='balanced' the intercept sits on the "
            "0.5 boundary, so an evidence-free URL is decided by a tie, not by evidence."
        ),
        "intercept": round(intercept, 6),
        "score_when_no_features": round(empty_score, 6),
        "decision_when_no_features": int(empty_score >= CLASSIFICATION_THRESHOLD),
        "tranco_holdout": zero_row_profile(tranco_holdout_urls, ext_score),
        "deep_benign": zero_row_profile(benign_urls, benign_score),
        "deep_malicious": zero_row_profile(malicious_urls, malicious_score),
    }
    print(f"    intercept={intercept:.4f}  score with no features={empty_score:.4f}  "
          f"-> decision {'MALICIOUS' if degeneracy['decision_when_no_features'] else 'benign'}")
    for name in ("tranco_holdout", "deep_benign", "deep_malicious"):
        profile = degeneracy[name]
        without = profile["mean_score_no_features"]
        with_ = profile["mean_score_with_features"]
        print(f"    {name:<15} no-feature rows {profile['rows_with_no_features']:>6,}"
              f"/{profile['rows']:<7,} ({profile['share_with_no_features'] * 100:5.2f}%)  "
              f"mean score without={without}  with={with_}")

    # -- I. Artifacts -------------------------------------------------------
    print("\n[I] Writing Stage 6 artifacts")
    total_features = (len(host_vec.vocabulary_) + len(path_vec.vocabulary_)
                      + len(query_vec.vocabulary_) + 3 * len(SUSPICIOUS_KEYWORDS))
    feature_config = {
        "representation": "scheme-blind field-separated TF-IDF + suspicious vocabulary counts",
        "scheme_handling": {
            "scheme_is_a_feature": False,
            "preprocessing": "http(s):// prefix stripped before any feature is computed",
            "equality_check": "features(original) == features(browser_form)",
        },
        "blocks": {
            "host": {"field": "hostname", "config": json_safe_config(HOST_CONFIG),
                     "vocab_size": len(host_vec.vocabulary_), "offset": 0},
            "path": {"field": "path", "config": json_safe_config(PATH_CONFIG),
                     "vocab_size": len(path_vec.vocabulary_),
                     "offset": len(host_vec.vocabulary_)},
            "query": {"field": "query", "config": json_safe_config(QUERY_CONFIG),
                      "vocab_size": len(query_vec.vocabulary_),
                      "offset": len(host_vec.vocabulary_) + len(path_vec.vocabulary_)},
            "suspicious_vocab": {
                "fields": ["host", "path", "query"],
                "keywords": list(SUSPICIOUS_KEYWORDS),
                "count": 3 * len(SUSPICIOUS_KEYWORDS),
                "offset": len(host_vec.vocabulary_) + len(path_vec.vocabulary_)
                          + len(query_vec.vocabulary_),
                "encoding": "raw substring counts per field",
            },
        },
        "total_features": int(total_features),
        "model_config": {**MODEL_CONFIG, "max_iter": used_max_iter},
    }

    d_comparison = json.loads(EXPERIMENT_D_COMPARISON.read_text(encoding="utf-8"))
    d_sep = json.loads(EXPERIMENT_D_SEPARABILITY.read_text(encoding="utf-8"))
    # Experiment D stores lower-case model keys; values are identical either way.
    r2_row = d_comparison["models"]["r2"]

    s1_row = {
        "model": "S1",
        "accuracy": round(internal["accuracy"], 6),
        "precision": round(internal["precision"], 6),
        "recall": round(internal["recall"], 6),
        "f1": round(internal["f1"], 6),
        "roc_auc": round(internal["roc_auc"], 6),
        "tn": internal["tn"], "fp": internal["fp"], "fn": internal["fn"], "tp": internal["tp"],
        "malicious_https_recall": round(internal["by_style"]["https"]["malicious_recall"], 6),
        "benign_https_fp_rate": round(internal["by_style"]["https"]["benign_fp_rate"], 6),
        "external_domain_safe_rate": round(external["domain_safe_rate"], 6),
        "legit_deep_safe_rate": round(deep_benign["overall"]["rate"], 6),
        "legit_deep_fp_rate": round(1.0 - deep_benign["overall"]["rate"], 6),
        "malicious_deep_recall": round(deep_malicious["overall"]["rate"], 6),
        "https_only_deep_roc_auc": https_sep["roc_auc"],
        "legit_sanity_safe": legit_safe,
        "suspicious_flagged": suspicious_hit,
        "model_size_mb": round(size_mb, 2),
        "gates_passed": gates["passed_count"],
        "gates_total": gates["total"],
        "all_gates_passed": gates["passed"],
        "active_features": feature_config["total_features"],
    }

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "internal_test": internal,
        "external_tranco_homepage": external,
        "deep_legitimate": deep_benign,
        "deep_malicious": deep_malicious,
        "deep_malicious_by_subtype": subtype_results,
        "separability": {"https_only_deep": https_sep, "all_deep": all_sep},
        "sanity": {"rows": sanity_rows, "legit_safe": legit_safe,
                   "suspicious_flagged": suspicious_hit},
        "diagnostics": diag_rows,
        "inference_latency": latency,
        "scheme_probe": scheme_probe,
        "degeneracy": degeneracy,
        "tranco_diagnosis": tranco_diagnosis,
        "gates": gates,
        "timings": {
            "mode": "report-only (loaded s1.pkl)" if args.report_only else "fit",
            "vectoriser_fit_seconds": round(vectoriser_seconds, 2),
            "train_transform_seconds": round(transform_seconds, 2),
            "classifier_fit_seconds": round(model_seconds, 2),
            "test_inference_seconds": round(eval_seconds, 2),
            "note": ("Report-only regeneration performs no fitting, so fit timings are 0. "
                     "The original fit run measured 9.6 s for the logistic regression "
                     "(101 lbfgs iterations); see experiment_six_run.log in this artifact directory."),
        },
    }

    comparison = {
        "experiment": "S1",
        "stage": "6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Test whether a scheme-blind, field-separated semantic lexical "
            "representation adds genuine deep-URL discriminative ability over the "
            "14 scalar counts, instead of shifting the operating point."
        ),
        "policy": {
            "candidate_count": 1,
            "tuning": "none; C=1.0, threshold=0.5 fixed before running",
            "threshold_changed": False,
            "augmentation_used": False,
            "network_used": False,
            "pandas_used": False,
            "promoted_anything": False,
        },
        "r2_reference_source": (
            "numbers read from model/experiment_d/comparison.json (detail.r2) and "
            "deep_separability.json; not recomputed"
        ),
        "feature_config": feature_config,
        "models": {"R2": r2_row, "S1": s1_row},
        "headline": {
            "r2_https_only_deep_auc": d_sep["results"]["R2|https-only deep"]["roc_auc"],
            "s1_https_only_deep_auc": https_sep["roc_auc"],
            "auc_delta": round(https_sep["roc_auc"]
                               - d_sep["results"]["R2|https-only deep"]["roc_auc"], 6),
            "r2_legit_deep_safe": r2_row["legit_deep_safe_rate"],
            "s1_legit_deep_safe": s1_row["legit_deep_safe_rate"],
            "r2_malicious_deep_recall": r2_row["malicious_deep_recall"],
            "s1_malicious_deep_recall": s1_row["malicious_deep_recall"],
            "r2_malicious_https_recall": r2_row["malicious_https_recall"],
            "s1_malicious_https_recall": s1_row["malicious_https_recall"],
        },
        "tranco_explanation": {
            "observed": tranco_diagnosis,
            "interpretation": [
                (f"S1's Tranco domain safe rate is {external['domain_safe_rate']:.4f} "
                 f"({external['domains_safe']}/{external['domain_count']} domains). R2, scored on the "
                 f"identical 4,000 URLs, reaches {r2_domain_safe:.4f} with mean score "
                 f"{float(r2_scores.mean()):.4f}, against S1's mean score {float(ext_score.mean()):.4f}."),
                (f"{n_flagged:,} of 4,000 holdout URLs cross the 0.5 boundary. On those rows the host "
                 f"block contributes {float(parts['host'][s1_flag].mean()):+.4f} mean logit, the path and "
                 f"query blocks contribute exactly 0.0000 because the holdout is homepage-only, the "
                 f"keyword block contributes {float(parts['vocab'][s1_flag].mean()):+.4f}, and the "
                 f"intercept is {float(classifier.intercept_[0]):+.4f}."),
                (f"The largest positive contributors on the flagged rows are host character n-grams: "
                 f"{', '.join(repr(t['feature']) for t in top_features[:5])}."),
                (f"The dataset writes a 'www.' hostname in {www_counts['malicious_share'] * 100:.2f}% of "
                 f"malicious training rows but only {www_counts['benign_share'] * 100:.2f}% of benign "
                 f"rows, so a positive www-prefix weight encodes a writing convention of the training "
                 f"data rather than URL semantics."),
                (f"Host n-gram coverage of the Tranco holdout is {float(tranco_coverage.mean()):.4f}, "
                 f"close to the deep-malicious sample ({float(malicious_coverage.mean()):.4f}) and below "
                 f"the deep-benign sample ({float(benign_coverage.mean()):.4f}), so unfamiliar hostnames "
                 f"land near the malicious side of the host block."),
                ("Observed and interpretation are separated deliberately: the numbers above are "
                 "measured on the unchanged fitted bundle; the causal reading is an interpretation."),
            ],
        },
        "interpretation": None,
    }

    # Predeclared interpretation rule (no post-hoc goalpost moving).
    improved_separability = comparison["headline"]["auc_delta"] >= 0.05
    passes = gates["passed"]
    if passes:
        verdict = "PASS: S1 satisfies every predeclared gate."
        recommend = True
    elif improved_separability:
        verdict = (
            "FAIL: S1 adds measurable deep-URL separability but misses at least one "
            "predeclared gate; not promotable."
        )
        recommend = False
    else:
        verdict = (
            "FAIL: S1 does not add meaningful deep-URL separability (AUC delta < 0.05); "
            "the limit is the representation/data, not the operating point."
        )
        recommend = False
    comparison["interpretation"] = {
        "improved_https_only_separability_by_0.05": improved_separability,
        "verdict": verdict,
        "recommend_promotion": recommend,
        "rule": (
            "Predeclared: promotion requires ALL gates. An AUC gain alone is evidence, "
            "not a promotion."
        ),
    }

    metadata = {
        "model_name": "FraudShield Stage 6 Experiment S1",
        "variant": "s1",
        "status": "experimental - NOT approved for deployment; R2 remains promoted",
        "experiment": "S1",
        "representation": feature_config,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "trained_on": {
            "kaggle_train_rows_purged": int(len(train_purged)),
            "tranco_train_rows": len(tranco_train_urls),
            "total": int(len(train_purged) + len(tranco_train_urls)),
            "augmentation": "none",
            "vocabulary_fit": "train only",
        },
        "metrics": s1_row,
        "evaluation": evaluation,
        "comparison_with_r2": comparison["headline"],
        "gates": gates,
        "leakage_audit": checks,
        "note": (
            "Deep-URL suites are browser-form representations of held-out Kaggle TEST "
            "rows. Features are scheme-blind, so browser form and original form are "
            "identical inputs. predict_proba is a model score, not a calibrated "
            "probability."
        ),
    }

    payloads = {
        "feature_config.json": feature_config,
        "evaluation.json": evaluation,
        "comparison.json": comparison,
        "s1.metadata.json": metadata,
    }
    for name, payload in payloads.items():   # fail fast: never write partial artifacts
        json.dumps(payload, ensure_ascii=False)
    for name, payload in payloads.items():
        write_json(OUT_DIR / name, payload)
        print(f"    wrote {name}")

    # -- J. Final report ----------------------------------------------------
    print("\n" + "=" * 86)
    print("[J] SUMMARY")
    print("=" * 86)
    print(f"{'metric':<32}{'R2':>14}{'S1':>14}{'delta':>12}")
    rows = (
        ("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"),
        ("f1", "f1"), ("roc_auc", "roc_auc"),
        ("malicious_https_recall", "malicious_https_recall"),
        ("benign_https_fp_rate", "benign_https_fp_rate"),
        ("tranco_domain_safe_rate", "external_domain_safe_rate"),
        ("legit_deep_safe_rate", "legit_deep_safe_rate"),
        ("malicious_deep_recall", "malicious_deep_recall"),
        ("https_only_deep_roc_auc", "https_only_deep_roc_auc"),
        ("model_size_mb", "model_size_mb"),
    )
    for label, key in rows:
        r2_value = r2_row.get(key)
        s1_value = s1_row.get(key)
        delta = ""
        if isinstance(r2_value, (int, float)) and isinstance(s1_value, (int, float)):
            delta = f"{s1_value - r2_value:+.4f}"
        print(f"{label:<32}{str(r2_value):>14}{str(s1_value):>14}{delta:>12}")
    print(f"{'gates_passed':<32}{str(r2_row['gates_passed'])+'/7':>14}"
          f"{str(s1_row['gates_passed'])+'/7':>14}")

    print("\nMalicious deep recall by subtype:")
    for subtype, payload in subtype_results.items():
        print(f"  {subtype:<14} n={payload['count']:>6,} recall={payload['recall']:.4f}")

    print("\nFamous deep diagnostics:")
    for row in diag_rows:
        print(f"  {row['label']:<22} score={row['score']:.4f} "
              f"{'MALICIOUS' if row['prediction'] else 'benign'}  {row['url']}")

    print(f"\nHTTPS-only deep ROC-AUC: S1={https_sep['roc_auc']:.4f} "
          f"vs R2={d_sep['results']['R2|https-only deep']['roc_auc']:.4f} "
          f"(delta {comparison['headline']['auc_delta']:+.4f})")
    print(f"\nVerdict: {verdict}")
    print(f"Artifacts: {OUT_DIR}")
    print("Nothing was promoted. model/fraudshield_model.pkl is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
