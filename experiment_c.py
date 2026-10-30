"""
FraudShield - Stage 4.6, Experiment C
=====================================

Goal: test whether REAL legitimate-domain coverage (Tranco) fixes the
browser-form benign URL problem WITHOUT recreating the "https => safe"
shortcut that made Model B fail.

Design (agreed in the Stage 4.6 plan):
  * Kaggle split is reproduced exactly from Stage 4 and the Kaggle TEST set is
    never modified, so Model A and Experiment C are compared on identical rows.
  * 10,000 Tranco domains (ranks 1-50,000) become training-only coverage.
  * 2,000 disjoint Tranco domains become an external legitimate-domain holdout.
  * Representations are GENERATED: https://<domain> for all training domains,
    plus https://www.<domain> for a deterministic ~30% subset. They are NOT
    observed browser URLs.
  * C1 uses all 14 canonical features. C2 uses the same rows and the same
    forest but drops "uses_https" at training/inference selection time only.
  * The better feature set (by security/generalization gates) also gets a
    smaller forest, R2, to test whether model size can be reduced safely.

Safety:
  * Every URL is plain text. No HTTP, DNS, WHOIS, TLS, or browser access.
  * data/malicious_phish.csv and data/tranco/top-1m.csv are read-only.
  * No pandas. No GridSearchCV. No deep learning.
  * Baseline and Model B artifacts are never overwritten.
  * No model is promoted by this script.

Run:
    python experiment_c.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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
TRANCO_ZIP = PROJECT_ROOT / "top-1m.csv.zip"
TRANCO_MANIFEST = PROJECT_ROOT / "data" / "tranco" / "tranco_manifest.json"
TRANCO_SELECTION = PROJECT_ROOT / "data" / "tranco" / "selection.json"

BASELINE_MODEL = PROJECT_ROOT / "model" / "fraudshield_model.pkl"
BASELINE_METADATA = PROJECT_ROOT / "model" / "model_metadata.json"

EXPERIMENT_DIR = PROJECT_ROOT / "model" / "experiment_c"
COMPARISON_PATH = EXPERIMENT_DIR / "comparison.json"

# ---------------------------------------------------------------------------
# Experiment constants (fixed before training; do not tune on results)
# ---------------------------------------------------------------------------

RANDOM_STATE = train.RANDOM_STATE  # 42
SELECTION_SALT = "fraudshield-c-v1"
TRANCO_MAX_RANK = 50_000
N_TRAIN_DOMAINS = 10_000
N_HOLDOUT_DOMAINS = 2_000
WWW_PERCENT = 30  # deterministic subset of training domains also gets a www form

# Sanity brands. Matching is per DNS label, so "google.co.uk" and
# "mail.google.com" are excluded but "pineapple.com" is not.
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
SANITY_DOMAINS = frozenset(
    {
        "google.com",
        "wikipedia.org",
        "github.com",
        "microsoft.com",
        "apple.com",
        "python.org",
        "mozilla.org",
        "reddit.com",
        "stackoverflow.com",
        "amazon.com",
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

C2_FEATURES = tuple(name for name in FEATURE_NAMES if name != "uses_https")
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

C1_CONFIG = dict(
    n_estimators=150,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
R2_CONFIG = dict(
    n_estimators=100,
    max_depth=24,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

SCHEMES = ("bare", "http", "https")
_DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ASCII_TLD_RE = re.compile(r"^[a-z]{2,63}$")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.2f}%" if whole else "0.00%"


def style_of(url: str) -> str:
    lowered = url.strip().lower()
    if lowered.startswith("https://"):
        return "https"
    if lowered.startswith("http://"):
        return "http"
    return "bare"


def brand_hit(hostname: str) -> bool:
    return any(label in SANITY_BRANDS for label in hostname.split("."))


def is_valid_domain(domain: str) -> bool:
    """Conservative syntax check; accepts punycode IDN TLDs."""
    if not domain or len(domain) > 253:
        return False
    labels = domain.split(".")
    if len(labels) < 2:
        return False
    if not all(_DOMAIN_LABEL_RE.match(label) for label in labels):
        return False
    tld = labels[-1]
    return bool(_ASCII_TLD_RE.match(tld)) or tld.startswith("xn--")


def domain_selection_hash(domain: str) -> str:
    return hashlib.sha256(f"{domain}|{SELECTION_SALT}".encode("utf-8")).hexdigest()


def wants_www(domain: str) -> bool:
    digest = hashlib.sha256(f"{domain}|{SELECTION_SALT}|www".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < WWW_PERCENT


def active_matrix(X: np.ndarray, active_names: tuple[str, ...]) -> np.ndarray:
    if tuple(active_names) == FEATURE_NAMES:
        return X
    indices = [FEATURE_INDEX[name] for name in active_names]
    return X[:, indices]


def vector_for(url: str, active_names: tuple[str, ...]) -> list[int]:
    features = extract_features(url)
    return [features[name] for name in active_names]


def active_matrix_from_urls(urls: list[str], active_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([vector_for(url, active_names) for url in urls], dtype=np.float32)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def suffix_ancestors(hostname: str):
    """Yield every proper parent suffix of a hostname (hostname itself excluded)."""
    parts = hostname.split(".")
    for index in range(1, len(parts)):
        yield ".".join(parts[index:])


def all_suffixes(hostname: str) -> list[str]:
    parts = hostname.split(".")
    return [".".join(parts[index:]) for index in range(len(parts))]


# ---------------------------------------------------------------------------
# Tranco inspection and selection
# ---------------------------------------------------------------------------


def inspect_tranco() -> dict:
    """Read-only verification of the downloaded Tranco CSV."""
    if not TRANCO_CSV.is_file():
        raise SystemExit(f"STOP: Tranco file not found: {TRANCO_CSV}")

    rows = 0
    field_errors = 0
    other_invalid = 0
    wildcard = 0
    punycode = 0
    duplicates = 0
    seen: set[str] = set()
    domains: list[str] = []

    with TRANCO_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows += 1
            if len(row) != 2:
                field_errors += 1
                continue
            rank_text, domain = row
            try:
                int(rank_text)
            except ValueError:
                field_errors += 1
                continue
            if not domain:
                field_errors += 1
                continue
            if domain in seen:
                duplicates += 1
                continue
            seen.add(domain)
            domains.append(domain)
            if "_" in domain:
                wildcard += 1
            elif "xn--" in domain:
                punycode += 1
            elif not is_valid_domain(domain) and not domain.startswith("www."):
                other_invalid += 1

    info = {
        "rows": rows,
        "unique_domains": len(domains),
        "duplicates": duplicates,
        "field_errors": field_errors,
        "other_invalid_entries": other_invalid,
        "malformed_entries": field_errors + wildcard + other_invalid,
        "wildcard_entries": wildcard,
        "idn_punycode_entries": punycode,
        "www_prefixed_entries": sum(1 for d in domains if d.startswith("www.")),
    }
    return info


def read_tranco_top(max_rank: int) -> list[str]:
    domains: list[str] = []
    with TRANCO_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                continue
            try:
                rank = int(row[0])
            except ValueError:
                continue
            if rank > max_rank:
                break
            domains.append(row[1])
    return domains


def build_selection(kaggle_hosts: set[str]) -> dict:
    """
    Deterministic Tranco selection.

    Conservative overlap rule (no Public Suffix List):
      a domain is excluded when it equals, is a parent of, or is a descendant
      of ANY Kaggle hostname in either split. This prevents generated apex/www
      forms from colliding with Kaggle train or test infrastructure.
    """
    raw = read_tranco_top(TRANCO_MAX_RANK)

    kaggle_suffixes: set[str] = set()
    for host in kaggle_hosts:
        kaggle_suffixes.update(all_suffixes(host))

    stats = {
        "raw_rows_rank_1_to_50000": len(raw),
        "duplicates_removed": 0,
        "invalid_syntax": 0,
        "www_prefixed_excluded": 0,
        "sanity_domain_or_variant_excluded": 0,
        "kaggle_overlap_excluded": 0,
        "parent_child_conflicts_skipped": 0,
    }
    sanity_excluded: list[str] = []
    kaggle_overlap_sample: list[str] = []
    seen: set[str] = set()
    candidates: list[str] = []

    for domain in raw:
        if domain in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(domain)
        if domain.startswith("www."):
            stats["www_prefixed_excluded"] += 1
            continue
        if not is_valid_domain(domain):
            stats["invalid_syntax"] += 1
            continue
        if brand_hit(domain):
            stats["sanity_domain_or_variant_excluded"] += 1
            sanity_excluded.append(domain)
            continue
        # Domain equals / is an ancestor of / is a descendant of a Kaggle host.
        if domain in kaggle_suffixes or any(ancestor in kaggle_hosts for ancestor in suffix_ancestors(domain)):
            stats["kaggle_overlap_excluded"] += 1
            if len(kaggle_overlap_sample) < 50:
                kaggle_overlap_sample.append(domain)
            continue
        candidates.append(domain)

    stats["eligible_candidates"] = len(candidates)
    ordered = sorted(candidates, key=domain_selection_hash)

    # Greedy de-conflict: keep the 12,000 hash-earliest domains that are not
    # parents/children of an already selected domain.
    selected: list[str] = []
    selected_set: set[str] = set()
    selected_suffixes: set[str] = set()
    conflict_examples: list[str] = []
    for domain in ordered:
        if domain in selected_set:
            continue
        has_selected_ancestor = any(
            ancestor in selected_set for ancestor in suffix_ancestors(domain)
        )
        is_selected_ancestor = domain in selected_suffixes
        if has_selected_ancestor or is_selected_ancestor:
            stats["parent_child_conflicts_skipped"] += 1
            if len(conflict_examples) < 25:
                conflict_examples.append(domain)
            continue
        selected.append(domain)
        selected_set.add(domain)
        selected_suffixes.update(all_suffixes(domain))
        if len(selected) == N_TRAIN_DOMAINS + N_HOLDOUT_DOMAINS:
            break

    if len(selected) < N_TRAIN_DOMAINS + N_HOLDOUT_DOMAINS:
        raise SystemExit(
            "STOP: not enough eligible Tranco domains after exclusions: "
            f"{len(selected):,} < {N_TRAIN_DOMAINS + N_HOLDOUT_DOMAINS:,}"
        )

    train_domains = selected[:N_TRAIN_DOMAINS]
    holdout_domains = selected[N_TRAIN_DOMAINS:]
    www_train = [domain for domain in train_domains if wants_www(domain)]

    return {
        "salt": SELECTION_SALT,
        "hash_rule": "sha256(domain + '|' + salt), lexicographic ascending",
        "www_rule": (
            "sha256(domain + '|' + salt + '|www') first 8 hex digits mod 100 < 30"
        ),
        "stats": stats
        | {
            "selected_domains": len(selected),
            "train_domains": len(train_domains),
            "holdout_domains": len(holdout_domains),
            "www_train_variants": len(www_train),
        },
        "sanity_excluded_domains": sorted(sanity_excluded),
        "kaggle_overlap_sample": kaggle_overlap_sample,
        "parent_child_conflict_sample": conflict_examples,
        "train_domains": train_domains,
        "holdout_domains": holdout_domains,
        "www_train_domains": www_train,
    }


# ---------------------------------------------------------------------------
# Distribution / univariate diagnostics
# ---------------------------------------------------------------------------


def scheme_table(styles: np.ndarray, labels: np.ndarray) -> dict:
    table = {0: {scheme: 0 for scheme in SCHEMES}, 1: {scheme: 0 for scheme in SCHEMES}}
    for label, scheme in zip(labels, styles):
        table[int(label)][str(scheme)] += 1
    return table


def cramers_v(table: dict) -> float:
    n = sum(sum(row.values()) for row in table.values())
    if n == 0:
        return 0.0
    row_totals = [sum(table[0].values()), sum(table[1].values())]
    col_totals = [table[0][s] + table[1][s] for s in SCHEMES]
    chi2 = 0.0
    for row_index, label in enumerate((0, 1)):
        for col_index, scheme in enumerate(SCHEMES):
            expected = row_totals[row_index] * col_totals[col_index] / n
            if expected:
                observed = table[label][scheme]
                chi2 += (observed - expected) ** 2 / expected
    return math.sqrt(chi2 / (n * (min(2, 3) - 1)))


def univariate_auc(X: np.ndarray, y: np.ndarray, name: str) -> float:
    return float(roc_auc_score(y, X[:, FEATURE_INDEX[name]]))


def format_scheme_table(table: dict) -> str:
    lines = []
    for label, name in ((0, "benign"), (1, "malicious")):
        total = sum(table[label].values())
        cells = "  ".join(
            f"{scheme}={table[label][scheme]:>8,} ({pct(table[label][scheme], total)})"
            for scheme in SCHEMES
        )
        lines.append(f"    {name:<10} n={total:>8,}  {cells}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_internal(
    model,
    X: np.ndarray,
    y: np.ndarray,
    styles: np.ndarray,
    brand_mask: np.ndarray | None = None,
) -> dict:
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

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
        "by_style": {},
        "sanity_brand_test_subset": None,
    }

    for scheme in SCHEMES:
        benign_mask = (styles == scheme) & (y == 0)
        malicious_mask = (styles == scheme) & (y == 1)
        payload["by_style"][scheme] = {
            "benign_count": int(benign_mask.sum()),
            "benign_fp": int(predictions[benign_mask].sum()) if benign_mask.any() else 0,
            "benign_fp_rate": float(predictions[benign_mask].mean()) if benign_mask.any() else 0.0,
            "benign_mean_score": float(scores[benign_mask].mean()) if benign_mask.any() else 0.0,
            "malicious_count": int(malicious_mask.sum()),
            "malicious_detected": int(predictions[malicious_mask].sum()) if malicious_mask.any() else 0,
            "malicious_recall": float(predictions[malicious_mask].mean()) if malicious_mask.any() else 0.0,
            "malicious_mean_score": float(scores[malicious_mask].mean()) if malicious_mask.any() else 0.0,
        }

    if brand_mask is not None and brand_mask.any():
        subset_y = y[brand_mask]
        subset_pred = predictions[brand_mask]
        payload["sanity_brand_test_subset"] = {
            "rows": int(brand_mask.sum()),
            "benign_rows": int((subset_y == 0).sum()),
            "malicious_rows": int((subset_y == 1).sum()),
            "accuracy": float(accuracy_score(subset_y, subset_pred)),
            "malicious_recall": float(recall_score(subset_y, subset_pred, zero_division=0)),
        }

    return payload


def evaluate_external(
    model,
    X: np.ndarray,
    urls: list[str],
    url_domains: list[str],
    active_names: tuple[str, ...],
) -> dict:
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]

    by_domain: dict[str, list[int]] = {}
    for position, domain in enumerate(url_domains):
        by_domain.setdefault(domain, []).append(position)

    domain_scores = [float(scores[positions].mean()) for positions in by_domain.values()]
    domain_safe = [all(int(predictions[position]) == 0 for position in positions)
                   for positions in by_domain.values()]

    apex_positions = [i for i, url in enumerate(urls) if not url.startswith("https://www.")]
    www_positions = [i for i, url in enumerate(urls) if url.startswith("https://www.")]

    worst = sorted(
        (
            {
                "domain": domain,
                "max_score": round(max(float(scores[p]) for p in positions), 6),
                "mean_score": round(float(scores[positions].mean()), 6),
                "forms": len(positions),
            }
            for domain, positions in by_domain.items()
        ),
        key=lambda item: item["max_score"],
        reverse=True,
    )[:20]

    return {
        "url_count": len(urls),
        "domain_count": len(by_domain),
        "url_safe_rate": float((predictions == 0).mean()),
        "domain_safe_rate": float(np.mean(domain_safe)),
        "domains_safe": int(np.sum(domain_safe)),
        "domains_unsafe": int(len(domain_safe) - np.sum(domain_safe)),
        "apex_safe_rate": float((predictions[apex_positions] == 0).mean()),
        "www_safe_rate": float((predictions[www_positions] == 0).mean()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_p90": float(np.percentile(scores, 90)),
        "domain_score_mean": float(np.mean(domain_scores)),
        "domain_score_median": float(np.median(domain_scores)),
        "domain_score_p90": float(np.percentile(domain_scores, 90)),
        "worst_domains": worst,
        "note": (
            "Generated representations of ranked Tranco domains, not observed browser "
            "URLs. Popularity is not a guarantee of safety."
        ),
    }


def evaluate_sanity(model, active_names: tuple[str, ...]) -> dict:
    rows = []
    for kind, suite in (("legit", LEGIT_SANITY), ("suspicious", SUSPICIOUS_SANITY)):
        for url in suite:
            vector = np.asarray([vector_for(url, active_names)], dtype=np.float32)
            prediction = int(model.predict(vector)[0])
            score = float(model.predict_proba(vector)[0, 1])
            rows.append(
                {
                    "kind": kind,
                    "url": url,
                    "prediction": prediction,
                    "score": round(score, 6),
                }
            )
    legit = [row for row in rows if row["kind"] == "legit"]
    suspicious = [row for row in rows if row["kind"] == "suspicious"]
    return {
        "rows": rows,
        "legit_safe_count": sum(1 for row in legit if row["prediction"] == 0),
        "legit_total": len(legit),
        "suspicious_flagged_count": sum(1 for row in suspicious if row["prediction"] == 1),
        "suspicious_total": len(suspicious),
        "note": "Sanity strings only; never used for fitting, tuning, or accuracy.",
    }


def quality_warnings(metrics: dict) -> dict:
    return {
        "accuracy_below_0.85": metrics["accuracy"] < 0.85,
        "precision_below_0.72": metrics["precision"] < 0.72,
        "f1_below_0.80": metrics["f1"] < 0.80,
        "roc_auc_below_0.93": metrics["roc_auc"] < 0.93,
    }


def acceptance_gates(
    metrics: dict,
    external: dict,
    sanity: dict,
    leakage_ok: bool,
    pretrain_ok: bool,
) -> dict:
    checks = {
        "https_malicious_recall_ge_0.90": metrics["by_style"]["https"]["malicious_recall"] >= 0.90,
        "overall_malicious_recall_ge_0.87": metrics["recall"] >= 0.87,
        "external_domain_safe_ge_0.85": external["domain_safe_rate"] >= 0.85,
        "suspicious_sanity_4_of_4": sanity["suspicious_flagged_count"] == sanity["suspicious_total"],
        "no_leakage_detected": leakage_ok,
        "pre_training_distribution_gate": pretrain_ok,
    }
    return {
        "checks": checks,
        "passed_count": sum(1 for value in checks.values() if value),
        "total": len(checks),
        "passed": all(checks.values()),
        "quality_warnings": quality_warnings(metrics),
        "quality_warning_triggered": any(quality_warnings(metrics).values()),
    }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------


def parent_child_audit(train_hosts: set[str], test_hosts: set[str]) -> dict:
    """Count train/test pairs where one hostname is a parent of the other."""
    train_is_ancestor = 0
    train_ancestor_examples: list[dict] = []
    for host in test_hosts:
        for ancestor in suffix_ancestors(host):
            if ancestor in train_hosts:
                train_is_ancestor += 1
                if len(train_ancestor_examples) < 10:
                    train_ancestor_examples.append({"test_host": host, "train_parent": ancestor})
                break

    test_is_ancestor = 0
    test_ancestor_examples: list[dict] = []
    for host in train_hosts:
        for ancestor in suffix_ancestors(host):
            if ancestor in test_hosts:
                test_is_ancestor += 1
                if len(test_ancestor_examples) < 10:
                    test_ancestor_examples.append({"train_host": host, "test_parent": ancestor})
                break

    return {
        "train_host_is_ancestor_of_test_host": train_is_ancestor,
        "test_host_is_ancestor_of_train_host": test_is_ancestor,
        "examples_train_parent": train_ancestor_examples,
        "examples_test_parent": test_ancestor_examples,
        "note": (
            "Hostname-only grouping without a Public Suffix List. Parent/child "
            "overlaps are reported, not claimed to be zero."
        ),
    }


def leakage_audit(
    kaggle_train_hosts: set[str],
    kaggle_test_hosts: set[str],
    tranco_train_domains: set[str],
    tranco_holdout_domains: set[str],
    train_repr_urls: list[str],
    holdout_repr_urls: list[str],
    kaggle_train_urls: set[str],
    kaggle_test_urls: set[str],
) -> dict:
    train_repr_hosts = {train.hostname_of(url) for url in train_repr_urls}
    holdout_repr_hosts = {train.hostname_of(url) for url in holdout_repr_urls}
    all_train_hosts = kaggle_train_hosts | train_repr_hosts

    exact_host_overlap = sorted(all_train_hosts & kaggle_test_hosts)
    derived_url_in_test = sorted(set(train_repr_urls) & kaggle_test_urls)
    derived_url_in_train = sorted(set(train_repr_urls) & kaggle_train_urls)
    holdout_domains_in_train = sorted(tranco_holdout_domains & tranco_train_domains)
    holdout_hosts_in_train = sorted(holdout_repr_hosts & all_train_hosts)
    sanity_hosts_in_train = sorted(host for host in all_train_hosts if brand_hit(host))
    sanity_domains_in_holds = sorted(SANITY_DOMAINS & (tranco_train_domains | tranco_holdout_domains))
    generated_sanity_urls = sorted(set(LEGIT_SANITY) & set(train_repr_urls))

    parent_child = parent_child_audit(all_train_hosts, kaggle_test_hosts)

    tranco_conflicts = sorted(
        domain
        for domain in tranco_holdout_domains
        if domain in tranco_train_domains
        or any(ancestor in tranco_train_domains for ancestor in suffix_ancestors(domain))
    )

    ok = not (
        exact_host_overlap
        or derived_url_in_test
        or derived_url_in_train
        or holdout_domains_in_train
        or holdout_hosts_in_train
        or sanity_hosts_in_train
        or sanity_domains_in_holds
        or generated_sanity_urls
        or tranco_conflicts
    )

    return {
        "ok": ok,
        "kaggle_train_hosts": len(kaggle_train_hosts),
        "kaggle_test_hosts": len(kaggle_test_hosts),
        "tranco_train_domains": len(tranco_train_domains),
        "tranco_holdout_domains": len(tranco_holdout_domains),
        "train_generated_hosts": len(train_repr_hosts),
        "exact_train_test_hostname_overlap": len(exact_host_overlap),
        "exact_overlap_examples": exact_host_overlap[:10],
        "derived_train_urls_in_kaggle_test": len(derived_url_in_test),
        "derived_train_urls_already_in_kaggle_train": len(derived_url_in_train),
        "holdout_source_domains_in_training_domains": len(holdout_domains_in_train),
        "holdout_representation_hosts_in_training_hosts": len(holdout_hosts_in_train),
        "sanity_brand_hosts_in_training": len(sanity_hosts_in_train),
        "sanity_brand_host_examples": sanity_hosts_in_train[:10],
        "sanity_domains_in_tranco_selection": len(sanity_domains_in_holds),
        "legit_sanity_urls_generated_as_training_rows": len(generated_sanity_urls),
        "tranco_holdout_parent_child_conflicts_with_train": len(tranco_conflicts),
        "derived_urls_in_test_examples": derived_url_in_test[:10],
        "parent_child": parent_child,
        "note": (
            "Tranco domains overlapping any Kaggle hostname (exact, parent, or child) "
            "were excluded before selection. Hostname grouping is not registrable-domain "
            "grouping because no Public Suffix List is used."
        ),
    }


# ---------------------------------------------------------------------------
# Training / artifact helpers
# ---------------------------------------------------------------------------


def fit_forest(X: np.ndarray, y: np.ndarray, config: dict) -> tuple[RandomForestClassifier, float]:
    model = RandomForestClassifier(**config)
    started = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - started
    return model, seconds


def model_metadata(
    label: str,
    active_names: tuple[str, ...],
    config: dict,
    metrics: dict,
    external: dict,
    sanity: dict,
    gates: dict,
    importances: dict,
    model_path: Path,
    size_mb: float,
    training_seconds: float,
    context: dict,
    extra: dict | None = None,
) -> dict:
    metadata = {
        "model_name": f"FraudShield Experiment C - {label.upper()}",
        "model_type": "RandomForestClassifier",
        "status": "experimental - NOT approved for deployment",
        "experiment": "C",
        "variant": label,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "active_feature_names": list(active_names),
        "active_feature_count": len(active_names),
        "canonical_feature_names": list(FEATURE_NAMES),
        "hyperparameters": config,
        "provenance": context["provenance"],
        "training_rows": context["training_rows"],
        "training_class_counts": context["training_class_counts"],
        "training_scheme_table": context["training_scheme_table"],
        "representations": context["representations"],
        "pre_training_gate": context["pretrain_gate"],
        "metrics_internal_test": metrics,
        "external_holdout": external,
        "sanity_suite": sanity,
        "acceptance_gates": gates,
        "feature_importances": importances,
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "model_size_mb": round(size_mb, 2),
        "training_seconds": round(training_seconds, 2),
        "leakage_audit": context["leakage"],
        "note": (
            "Tranco rows are generated representations of ranked legitimate domains, "
            "not observed browser URLs. predict_proba is a model risk score, not a "
            "calibrated real-world probability."
        ),
    }
    if extra:
        metadata.update(extra)
    return metadata


def run_variant(
    label: str,
    active_names: tuple[str, ...],
    config: dict,
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_test_full: np.ndarray,
    y_test: np.ndarray,
    styles_test: np.ndarray,
    brand_mask_test: np.ndarray,
    X_holdout_full: np.ndarray,
    holdout_urls: list[str],
    holdout_url_domains: list[str],
    context: dict,
) -> dict:
    print(f"\n[{label.upper()}] Training {len(active_names)} features, "
          f"{config['n_estimators']} trees...")
    X_train = active_matrix(X_train_full, active_names)
    model, seconds = fit_forest(X_train, y_train, config)
    del X_train

    if model.n_features_in_ != len(active_names):
        raise SystemExit(
            f"STOP: {label} expects {model.n_features_in_} features, "
            f"active list has {len(active_names)}."
        )

    metrics = evaluate_internal(
        model, active_matrix(X_test_full, active_names), y_test, styles_test, brand_mask_test
    )
    external = evaluate_external(
        model, active_matrix(X_holdout_full, active_names), holdout_urls, holdout_url_domains, active_names
    )
    sanity = evaluate_sanity(model, active_names)

    importances = {
        name: round(float(value), 6)
        for name, value in sorted(
            zip(active_names, model.feature_importances_.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }
    gates = acceptance_gates(
        metrics, external, sanity, context["leakage"]["ok"], context["pretrain_gate"]["passed"]
    )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = EXPERIMENT_DIR / f"{label}.pkl"
    joblib.dump(model, model_path, compress=3)
    reloaded = joblib.load(model_path)
    if reloaded.n_features_in_ != len(active_names):
        raise SystemExit(f"STOP: saved {label} failed the feature-count check.")
    probe = active_matrix(X_test_full[:4], active_names)
    if not np.array_equal(reloaded.predict(probe), model.predict(probe)):
        raise SystemExit(f"STOP: saved {label} does not reproduce predictions.")
    size_mb = model_path.stat().st_size / (1024 * 1024)

    metadata = model_metadata(
        label,
        active_names,
        config,
        metrics,
        external,
        sanity,
        gates,
        importances,
        model_path,
        size_mb,
        seconds,
        context,
    )
    metadata_path = EXPERIMENT_DIR / f"{label}.metadata.json"
    write_json(metadata_path, metadata)

    print(
        f"    acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}"
    )
    print(
        f"    https malicious recall={metrics['by_style']['https']['malicious_recall']:.4f}  "
        f"https benign FP={metrics['by_style']['https']['benign_fp_rate']:.4f}"
    )
    print(
        f"    external domain safe={external['domain_safe_rate']:.4f}  "
        f"sanity legit safe={sanity['legit_safe_count']}/10  "
        f"suspicious={sanity['suspicious_flagged_count']}/4"
    )
    print(f"    gates {gates['passed_count']}/{gates['total']}  size={size_mb:.2f} MB  "
          f"time={seconds:.1f}s")
    if label == "c1_r1":
        print(f"    uses_https importance = {importances['uses_https']:.4f} "
              f"(Model A 0.0167, Model B 0.2838)")

    return {
        "label": label,
        "model": model,
        "path": model_path,
        "metadata_path": metadata_path,
        "active_names": active_names,
        "config": config,
        "metrics": metrics,
        "external": external,
        "sanity": sanity,
        "gates": gates,
        "importances": importances,
        "size_mb": size_mb,
        "seconds": seconds,
    }


def summary_row(name: str, result_or_dict: dict) -> dict:
    metrics = result_or_dict.get("metrics", result_or_dict.get("metrics_internal_test"))
    external = result_or_dict.get("external", result_or_dict.get("external_holdout"))
    sanity = result_or_dict.get("sanity", result_or_dict.get("sanity_suite"))
    gates = result_or_dict.get("gates", result_or_dict.get("acceptance_gates"))
    size_mb = result_or_dict.get("size_mb", result_or_dict.get("model_size_mb"))
    return {
        "model": name,
        "accuracy": round(metrics["accuracy"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "f1": round(metrics["f1"], 6),
        "roc_auc": round(metrics["roc_auc"], 6),
        "false_positives": metrics["fp"],
        "false_negatives": metrics["fn"],
        "https_malicious_recall": round(metrics["by_style"]["https"]["malicious_recall"], 6),
        "https_benign_fp_rate": round(metrics["by_style"]["https"]["benign_fp_rate"], 6),
        "external_domain_safe_rate": round(external["domain_safe_rate"], 6),
        "external_url_safe_rate": round(external["url_safe_rate"], 6),
        "legit_sanity_safe": sanity["legit_safe_count"],
        "suspicious_flagged": sanity["suspicious_flagged_count"],
        "model_size_mb": round(float(size_mb), 2),
        "gates_passed": gates["passed_count"] if gates else None,
        "gates_total": gates["total"] if gates else None,
        "all_gates_passed": gates["passed"] if gates else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 80)
    print("FraudShield Stage 4.6 - Experiment C (Tranco legitimate-domain coverage)")
    print("=" * 80)
    print("All URLs are processed as text only. No HTTP/DNS/WHOIS/TLS/browser access.")

    for required in (DATASET_PATH, TRANCO_CSV, BASELINE_MODEL, BASELINE_METADATA):
        if not required.is_file():
            print(f"STOP: required file missing: {required}")
            return 1

    baseline_meta = json.loads(BASELINE_METADATA.read_text(encoding="utf-8"))

    # -- A. Verify inputs ---------------------------------------------------
    print("\n[A] Input verification (read-only)")
    dataset_hash = file_sha256(DATASET_PATH)
    dataset_unchanged = dataset_hash == baseline_meta.get("dataset_sha256")
    print(f"    Kaggle dataset : {DATASET_PATH.name} ({DATASET_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"    Kaggle SHA-256 : {dataset_hash}")
    print(f"    vs Stage 4    : {'UNCHANGED' if dataset_unchanged else 'CHANGED - STOP'}")
    if not dataset_unchanged:
        print("STOP: the Kaggle dataset is not byte-identical to Stage 4.")
        return 1

    tranco_bytes = TRANCO_CSV.stat().st_size
    tranco_hash = file_sha256(TRANCO_CSV)
    tranco_info = inspect_tranco()
    archive_hash = file_sha256(TRANCO_ZIP) if TRANCO_ZIP.is_file() else None
    archive_member_mtime = None
    if TRANCO_ZIP.is_file():
        with zipfile.ZipFile(TRANCO_ZIP) as archive:
            for member in archive.infolist():
                if member.filename.endswith(".csv"):
                    archive_member_mtime = "%04d-%02d-%02dT%02d:%02d:%02d" % member.date_time
                    break

    print(f"    Tranco bytes   : {tranco_bytes:,}")
    print(f"    Tranco SHA-256 : {tranco_hash}")
    print(
        "    Tranco format  : rank,domain | "
        f"rows={tranco_info['rows']:,} | unique={tranco_info['unique_domains']:,} | "
        f"duplicates={tranco_info['duplicates']}"
    )
    print(
        "    Tranco quirks  : "
        f"idn/punycode={tranco_info['idn_punycode_entries']}, "
        f"wildcard/underscore={tranco_info['wildcard_entries']}, "
        f"www-prefixed={tranco_info['www_prefixed_entries']}"
    )

    manifest = {
        "source": "Tranco list (https://tranco-list.eu/) - research-oriented top sites ranking",
        "download_url": "https://tranco-list.eu/top-1m.csv.zip",
        "attribution": (
            "Le Pochat, Van Goethem, Tajalizadehkhoob, Korczynski, Joosen. "
            "Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation (NDSS 2019)."
        ),
        "local_csv_path": str(TRANCO_CSV),
        "csv_sha256": tranco_hash,
        "csv_size_bytes": tranco_bytes,
        "archive_path": str(TRANCO_ZIP) if TRANCO_ZIP.is_file() else None,
        "archive_sha256": archive_hash,
        "archive_member_last_modified": archive_member_mtime,
        "csv_mtime_utc": datetime.fromtimestamp(
            TRANCO_CSV.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "rows": tranco_info["rows"],
        "unique_domains": tranco_info["unique_domains"],
        "rank_range": [1, tranco_info["rows"]],
        "format": "rank,domain (no header; CRLF line endings)",
        "duplicates": tranco_info["duplicates"],
        "malformed_entries": tranco_info["malformed_entries"],
        "malformed_breakdown": {
            "wildcard_or_underscore": tranco_info["wildcard_entries"],
            "idn_punycode_accepted_as_valid": tranco_info["idn_punycode_entries"],
            "other_invalid": tranco_info["other_invalid_entries"] + tranco_info["field_errors"],
        },
        "list_id": None,
        "list_id_note": (
            "No list ID is embedded in top-1m.csv. The authoritative ID is published at "
            "https://tranco-list.eu/top-1m-id; it was NOT fetched because this stage forbids "
            "network access."
        ),
        "verified_read_only": True,
        "network_used": False,
    }
    write_json(TRANCO_MANIFEST, manifest)
    print(f"    Manifest       : {TRANCO_MANIFEST}")

    # -- Load Kaggle and reproduce the Stage 4 split ------------------------
    print("\n[1] Loading Kaggle dataset (read-only) and reproducing the Stage 4 split...")
    started = time.perf_counter()
    bundle = load_dataset(
        DATASET_PATH,
        label_map=train.SOURCE_TO_BINARY,
        url_column="url",
        label_column="type",
    )
    X_all = np.asarray(bundle.X, dtype=np.float32)
    y_all = np.asarray(bundle.y, dtype=np.int8)
    urls = bundle.urls
    report = bundle.report
    del bundle
    print(f"    Usable rows: {len(urls):,} ({time.perf_counter() - started:.1f}s)")

    train_idx, test_idx, groups = train.split_by_hostname(urls, y_all)
    if len(train_idx) != baseline_meta["training_samples"] or len(test_idx) != baseline_meta["testing_samples"]:
        print("STOP: split sizes differ from the Stage 4 baseline.")
        return 1
    print(f"    Split: train {len(train_idx):,} / test {len(test_idx):,} (identical sizes to Stage 4)")

    kaggle_train_hosts_all = set(groups[train_idx])
    kaggle_test_hosts = set(groups[test_idx])
    if kaggle_train_hosts_all & kaggle_test_hosts:
        print("STOP: original Kaggle hostname leakage detected.")
        return 1

    # Purge sanity-brand rows from the Kaggle TRAINING side only.
    keep_train = np.asarray([not brand_hit(groups[index]) for index in train_idx], dtype=bool)
    train_idx_purged = train_idx[keep_train]
    purged_rows = int((~keep_train).sum())
    purged_benign = int((y_all[train_idx][~keep_train] == 0).sum())
    purged_malicious = int((y_all[train_idx][~keep_train] == 1).sum())
    print(
        f"    Sanity purge (training only): {purged_rows:,} rows "
        f"(benign {purged_benign:,}, malicious {purged_malicious:,}); "
        f"train now {len(train_idx_purged):,}"
    )

    kaggle_train_hosts = set(groups[train_idx_purged])
    kaggle_test_urls = {urls[index] for index in test_idx}
    kaggle_train_urls = {urls[index] for index in train_idx_purged}
    brand_mask_test = np.asarray([brand_hit(groups[index]) for index in test_idx], dtype=bool)
    print(
        f"    Sanity-brand rows retained in the fixed Kaggle TEST set: "
        f"{int(brand_mask_test.sum()):,} (reported separately, never removed)"
    )

    # -- B. Tranco selection ------------------------------------------------
    print("\n[2] Deterministic Tranco selection (ranks 1-50,000)...")
    kaggle_hosts_all = kaggle_train_hosts_all | kaggle_test_hosts
    selection = build_selection(kaggle_hosts_all)
    write_json(TRANCO_SELECTION, selection)
    stats = selection["stats"]
    print(
        f"    raw={stats['raw_rows_rank_1_to_50000']:,}  "
        f"invalid={stats['invalid_syntax']:,}  www-prefixed={stats['www_prefixed_excluded']:,}  "
        f"sanity={stats['sanity_domain_or_variant_excluded']:,}  "
        f"kaggle-overlap={stats['kaggle_overlap_excluded']:,}  "
        f"parent/child skipped={stats['parent_child_conflicts_skipped']:,}"
    )
    print(
        f"    Selected: {stats['train_domains']:,} training + "
        f"{stats['holdout_domains']:,} external holdout domains"
    )
    print(f"    Deterministic www training variants: {stats['www_train_variants']:,}")
    print(f"    Selection artifact: {TRANCO_SELECTION}")

    train_domains = selection["train_domains"]
    holdout_domains = selection["holdout_domains"]
    www_train_domains = selection["www_train_domains"]

    # Generated representations (documented as generated, never observed).
    train_repr_urls = [f"https://{domain}" for domain in train_domains]
    train_repr_urls += [f"https://www.{domain}" for domain in www_train_domains]
    holdout_urls: list[str] = []
    holdout_url_domains: list[str] = []
    for domain in holdout_domains:
        holdout_urls.append(f"https://{domain}")
        holdout_url_domains.append(domain)
        holdout_urls.append(f"https://www.{domain}")
        holdout_url_domains.append(domain)

    print(
        f"    Training representations: {len(train_repr_urls):,} "
        f"({len(train_domains):,} apex + {len(www_train_domains):,} www)"
    )
    print(
        f"    Holdout representations: {len(holdout_urls):,} "
        f"({len(holdout_domains):,} domains x apex+www)"
    )

    # -- C. Assemble training data ------------------------------------------
    print("\n[3] Assembling training data...")
    X_kaggle_train = X_all[train_idx_purged]
    y_kaggle_train = y_all[train_idx_purged]
    X_tranco_train = active_matrix_from_urls(train_repr_urls, FEATURE_NAMES)
    X_train = np.vstack([X_kaggle_train, X_tranco_train]).astype(np.float32, copy=False)
    y_train = np.concatenate(
        [y_kaggle_train, np.zeros(len(X_tranco_train), dtype=np.int8)]
    )
    styles_train = np.asarray(
        [style_of(urls[index]) for index in train_idx_purged]
        + [style_of(url) for url in train_repr_urls],
        dtype=object,
    )
    del X_kaggle_train, X_tranco_train, y_kaggle_train

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    styles_test = np.asarray([style_of(urls[index]) for index in test_idx], dtype=object)
    X_holdout = active_matrix_from_urls(holdout_urls, FEATURE_NAMES)

    # Baseline train table (unpurged Kaggle training rows) for comparison.
    baseline_styles = np.asarray([style_of(urls[index]) for index in train_idx], dtype=object)
    baseline_labels = np.asarray([int(value) for value in y_all[train_idx]], dtype=np.int8)
    baseline_table = scheme_table(baseline_styles, baseline_labels)
    del X_all, y_all, baseline_styles, baseline_labels

    table = scheme_table(styles_train, y_train)
    print(format_scheme_table(table))
    print(f"    Total training rows: {len(y_train):,}")

    # -- D. Pre-training distribution gate ----------------------------------
    print("\n[4] Pre-training distribution gate (computed BEFORE fitting)")

    https_benign = table[0]["https"]
    https_malicious = table[1]["https"]
    https_benign_share = https_benign / (https_benign + https_malicious)
    v_final = cramers_v(table)
    v_baseline = cramers_v(baseline_table)
    auc_https = univariate_auc(X_train, y_train, "uses_https")
    auc_report = {
        name: round(univariate_auc(X_train, y_train, name), 6)
        for name in ("uses_https", "num_slashes", "num_dots", "url_length", "num_subdomains")
    }

    checks = {
        "https_benign_share_0.40_to_0.60": 0.40 <= https_benign_share <= 0.60,
        "uses_https_univariate_auc_0.45_to_0.55": 0.45 <= auc_https <= 0.55,
        "cramers_v_not_worse_than_baseline": v_final <= v_baseline + 0.001,
    }
    print(f"    HTTPS benign share     : {https_benign_share:.4f}  (gate 0.40-0.60)")
    print(f"    uses_https univariate AUC: {auc_https:.4f}  (gate 0.45-0.55)")
    print(f"    Cramer's V scheme x label: {v_final:.4f}  (baseline train {v_baseline:.4f})")
    print("    Other univariate AUCs   : " + "  ".join(f"{k}={v:.4f}" for k, v in auc_report.items()))
    for name, passed in checks.items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    pretrain_gate = {
        "https_benign_share": round(https_benign_share, 6),
        "https_benign_count": https_benign,
        "https_malicious_count": https_malicious,
        "cramers_v_final": round(v_final, 6),
        "cramers_v_baseline_train": round(v_baseline, 6),
        "univariate_auc": auc_report,
        "scheme_table": table,
        "checks": checks,
        "passed": all(checks.values()),
        "note": (
            "num_dots/num_subdomains/num_slashes retain high univariate AUC because "
            "malicious Kaggle URLs contain deep paths and subdomains. This is a real "
            "dataset property, not an HTTPS label proxy; the formal gate is the "
            "uses_https AUC plus the HTTPS class balance."
        ),
    }
    if not pretrain_gate["passed"]:
        print("\nSTOP: the proposed construction creates a scheme shortcut. "
              "No model was trained.")
        return 1

    # -- E. Leakage audit ---------------------------------------------------
    print("\n[5] Leakage audit")
    tranco_train_domains = set(train_domains)
    tranco_holdout_domains_set = set(holdout_domains)
    leakage = leakage_audit(
        kaggle_train_hosts,
        kaggle_test_hosts,
        tranco_train_domains,
        tranco_holdout_domains_set,
        train_repr_urls,
        holdout_urls,
        kaggle_train_urls,
        kaggle_test_urls,
    )
    print(f"    exact train/test hostname overlap     : {leakage['exact_train_test_hostname_overlap']}")
    print(f"    derived train URLs inside Kaggle test : {leakage['derived_train_urls_in_kaggle_test']}")
    print(f"    holdout domains inside training       : {leakage['holdout_source_domains_in_training_domains']}")
    print(f"    holdout hosts inside training hosts   : {leakage['holdout_representation_hosts_in_training_hosts']}")
    print(f"    sanity-brand hosts inside training    : {leakage['sanity_brand_hosts_in_training']}")
    print(f"    sanity URLs generated as training rows: {leakage['legit_sanity_urls_generated_as_training_rows']}")
    print(
        f"    parent/child (train parent of test / test parent of train): "
        f"{leakage['parent_child']['train_host_is_ancestor_of_test_host']} / "
        f"{leakage['parent_child']['test_host_is_ancestor_of_train_host']}"
    )
    if not leakage["ok"]:
        print("STOP: leakage detected. No model was trained.")
        return 1

    # -- Shared context -----------------------------------------------------
    train_benign, train_malicious = int((y_train == 0).sum()), int((y_train == 1).sum())
    context = {
        "provenance": {
            "kaggle_dataset": DATASET_PATH.name,
            "kaggle_sha256": dataset_hash,
            "kaggle_original_rows": report.total_rows,
            "kaggle_usable_rows": report.usable_rows,
            "kaggle_train_rows_before_purge": int(len(train_idx)),
            "kaggle_train_rows_after_purge": int(len(train_idx_purged)),
            "kaggle_test_rows_fixed": int(len(test_idx)),
            "sanity_rows_purged_from_train": purged_rows,
            "tranco_csv": TRANCO_CSV.name,
            "tranco_sha256": tranco_hash,
            "tranco_rows": tranco_info["rows"],
            "tranco_selection_salt": SELECTION_SALT,
            "tranco_rank_limit": TRANCO_MAX_RANK,
        },
        "training_rows": int(len(y_train)),
        "training_class_counts": {"benign": train_benign, "malicious": train_malicious},
        "training_scheme_table": table,
        "representations": {
            "train": (
                "Generated representations of ranked legitimate Tranco domains: "
                "https://<domain> for 100% and https://www.<domain> for the "
                "deterministic ~30% subset. These are NOT observed browser URLs."
            ),
            "holdout": (
                "Generated representations of held-out Tranco domains: "
                "https://<domain> and https://www.<domain> for every domain. "
                "Never used for fitting, threshold selection, feature selection, "
                "or hyperparameter tuning."
            ),
            "train_rows": len(train_repr_urls),
            "holdout_rows": len(holdout_urls),
        },
        "pretrain_gate": pretrain_gate,
        "leakage": leakage,
    }

    # -- F. Baseline Model A evaluation -------------------------------------
    print("\n[6] Model A (untouched Stage 4 baseline) - same evaluation suite")
    baseline_model = joblib.load(BASELINE_MODEL)
    if baseline_model.n_features_in_ != len(FEATURE_NAMES):
        print("STOP: baseline feature count mismatch.")
        return 1
    baseline_metrics = evaluate_internal(
        baseline_model, X_test, y_test, styles_test, brand_mask_test
    )
    baseline_external = evaluate_external(
        baseline_model, X_holdout, holdout_urls, holdout_url_domains, FEATURE_NAMES
    )
    baseline_sanity = evaluate_sanity(baseline_model, FEATURE_NAMES)
    print(
        f"    acc={baseline_metrics['accuracy']:.6f} rec={baseline_metrics['recall']:.6f} "
        f"f1={baseline_metrics['f1']:.6f} auc={baseline_metrics['roc_auc']:.6f}"
    )
    if abs(baseline_metrics["accuracy"] - baseline_meta["accuracy"]) > 1e-5:
        print("STOP: baseline metrics did not reproduce; compare split is invalid.")
        return 1
    baseline_size_mb = BASELINE_MODEL.stat().st_size / (1024 * 1024)

    # -- G. C1, C2, R2 ------------------------------------------------------
    c1 = run_variant(
        "c1_r1", FEATURE_NAMES, C1_CONFIG,
        X_train, y_train, X_test, y_test, styles_test, brand_mask_test,
        X_holdout, holdout_urls, holdout_url_domains, context,
    )
    c2 = run_variant(
        "c2_r1", C2_FEATURES, C1_CONFIG,
        X_train, y_train, X_test, y_test, styles_test, brand_mask_test,
        X_holdout, holdout_urls, holdout_url_domains, context,
    )

    # -- H. Winner selection ------------------------------------------------
    def rank_key(result: dict) -> tuple:
        return (
            1 if result["gates"]["passed"] else 0,
            result["gates"]["passed_count"],
            round(result["external"]["domain_safe_rate"], 6),
            round(result["metrics"]["by_style"]["https"]["malicious_recall"], 6),
            round(result["metrics"]["recall"], 6),
            round(result["metrics"]["f1"], 6),
        )

    winner = c1 if rank_key(c1) >= rank_key(c2) else c2

    # Parsimony tie-break: if the two are effectively tied on the security and
    # generalization metrics, prefer C2 (no scheme feature).
    ext_gap = abs(c1["external"]["domain_safe_rate"] - c2["external"]["domain_safe_rate"])
    https_gap = abs(
        c1["metrics"]["by_style"]["https"]["malicious_recall"]
        - c2["metrics"]["by_style"]["https"]["malicious_recall"]
    )
    recall_gap = abs(c1["metrics"]["recall"] - c2["metrics"]["recall"])
    if (
        c1["gates"]["passed"] == c2["gates"]["passed"]
        and ext_gap <= 0.005
        and https_gap <= 0.005
        and recall_gap <= 0.005
    ):
        winner = c2
        tie_break = "parsimony: C1 and C2 effectively tied, so the 13-feature C2 is preferred"
    else:
        tie_break = "higher security/generalization rank"

    print(f"\n[7] Winner: {winner['label'].upper()} ({tie_break})")

    r2 = run_variant(
        "winner_r2",
        winner["active_names"],
        R2_CONFIG,
        X_train, y_train, X_test, y_test, styles_test, brand_mask_test,
        X_holdout, holdout_urls, holdout_url_domains, context,
    )

    # R2 must not materially degrade the winner.
    ext_drop = winner["external"]["domain_safe_rate"] - r2["external"]["domain_safe_rate"]
    https_drop = (
        winner["metrics"]["by_style"]["https"]["malicious_recall"]
        - r2["metrics"]["by_style"]["https"]["malicious_recall"]
    )
    recall_drop = winner["metrics"]["recall"] - r2["metrics"]["recall"]
    r2_acceptable = (
        r2["gates"]["passed"]
        and ext_drop <= 0.02
        and https_drop <= 0.02
        and recall_drop <= 0.01
    )
    print(
        f"    R2 vs winner: external safe delta={-ext_drop:+.4f}, "
        f"https recall delta={-https_drop:+.4f}, overall recall delta={-recall_drop:+.4f}"
    )
    print(f"    R2 acceptable: {r2_acceptable}")

    # -- I. Comparison ------------------------------------------------------
    print("\n[8] Comparison: Model A vs C1 vs C2 vs R2")
    baseline_row = summary_row("A_baseline", {
        "metrics": baseline_metrics,
        "external": baseline_external,
        "sanity": baseline_sanity,
        "size_mb": baseline_size_mb,
        "gates": None,
    })
    rows = [
        baseline_row,
        summary_row("C1_all14", c1),
        summary_row("C2_no_https", c2),
        summary_row("R2_small", r2),
    ]
    headers = (
        ("model", 12), ("acc", 7), ("prec", 7), ("rec", 7), ("f1", 7), ("auc", 7),
        ("FP", 7), ("FN", 7), ("HTTPSrec", 10), ("HTTPSfp", 9),
        ("extSafe", 9), ("10safe", 7), ("4flag", 6), ("MB", 8),
    )
    print("    " + "".join(name.ljust(width) for name, width in headers))
    for row in rows:
        values = (
            row["model"],
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['roc_auc']:.4f}",
            f"{row['false_positives']:,}",
            f"{row['false_negatives']:,}",
            f"{row['https_malicious_recall']:.4f}",
            f"{row['https_benign_fp_rate']:.4f}",
            f"{row['external_domain_safe_rate']:.4f}",
            f"{row['legit_sanity_safe']}/10",
            f"{row['suspicious_flagged']}/4",
            f"{row['model_size_mb']:.1f}",
        )
        print("    " + "".join(str(value).ljust(width) for value, (_, width) in zip(values, headers)))

    print("\n[9] Per-style internal test metrics")
    for label, result in (("A", baseline_row), ("C1", c1), ("C2", c2), ("R2", r2)):
        metrics = (
            baseline_metrics if label == "A"
            else result["metrics"]
        )
        print(f"    {label}:")
        for scheme in SCHEMES:
            entry = metrics["by_style"][scheme]
            print(
                f"      {scheme:<6} benign n={entry['benign_count']:>7,} "
                f"FP={entry['benign_fp']:>6,} ({entry['benign_fp_rate']:.2%})  |  "
                f"malicious n={entry['malicious_count']:>7,} "
                f"recall={entry['malicious_recall']:.2%}"
            )

    print("\n[10] Sanity suite (never used for fitting or accuracy)")
    for result in (c1, c2, r2):
        print(f"    {result['label'].upper()}:")
        for row in result["sanity"]["rows"]:
            tag = "SAFE" if row["prediction"] == 0 else "RISK"
            print(f"      {tag:<5} score={row['score']:.4f}  {row['url']}")
    print("    BASELINE A:")
    for row in baseline_sanity["rows"]:
        tag = "SAFE" if row["prediction"] == 0 else "RISK"
        print(f"      {tag:<5} score={row['score']:.4f}  {row['url']}")

    print("\n[11] Feature importance")
    for label, importances in (
        ("A", baseline_meta["feature_importances"]),
        ("C1", [{"feature": k, "importance": v} for k, v in c1["importances"].items()]),
        ("C2", [{"feature": k, "importance": v} for k, v in c2["importances"].items()]),
        ("R2", [{"feature": k, "importance": v} for k, v in r2["importances"].items()]),
    ):
        print(f"    {label}: " + ", ".join(
            f"{item['feature']}={item['importance']:.4f}" for item in importances
        ))

    # -- J. Decision --------------------------------------------------------
    if not winner["gates"]["passed"]:
        recommendation = "NO CANDIDATE PASSES THE SECURITY/GENERALIZATION GATES - do not promote."
        promote = None
    elif r2_acceptable and r2["size_mb"] < winner["size_mb"]:
        recommendation = (
            f"Promote {r2['label'].upper()} (passes all gates and matches the larger winner "
            "within the stated tolerances while being smaller)."
        )
        promote = r2["label"]
    else:
        recommendation = (
            f"Promote {winner['label'].upper()} (R2 is not an acceptable size reduction)."
        )
        promote = winner["label"]

    c1_https_importance = c1["importances"]["uses_https"]
    if c1_https_importance > 0.15:
        uses_https_verdict = (
            f"uses_https importance is {c1_https_importance:.4f}, well above Model A's 0.0167; "
            "treat it as a warning and prefer C2 unless C1 is clearly better on the holdout."
        )
    else:
        uses_https_verdict = (
            f"uses_https importance is {c1_https_importance:.4f} (Model A 0.0167, Model B 0.2838), "
            "so scheme is not dominating this forest."
        )

    comparison = {
        "experiment": "C",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "experimental - no model promoted",
        "provenance": context["provenance"],
        "training_rows": context["training_rows"],
        "training_class_counts": context["training_class_counts"],
        "training_scheme_table": context["training_scheme_table"],
        "representations": context["representations"],
        "pre_training_gate": pretrain_gate,
        "leakage_audit": leakage,
        "models": {
            "A_baseline": {
                "summary": baseline_row,
                "metrics_internal_test": baseline_metrics,
                "external_holdout": baseline_external,
                "sanity_suite": baseline_sanity,
                "model_path": str(BASELINE_MODEL),
                "feature_names": list(FEATURE_NAMES),
            },
            "C1_all14": {
                "summary": summary_row("C1_all14", c1),
                "metrics_internal_test": c1["metrics"],
                "external_holdout": c1["external"],
                "sanity_suite": c1["sanity"],
                "acceptance_gates": c1["gates"],
                "feature_importances": c1["importances"],
                "active_feature_names": list(c1["active_names"]),
                "hyperparameters": c1["config"],
                "model_path": str(c1["path"]),
                "metadata_path": str(c1["metadata_path"]),
                "model_size_mb": c1["size_mb"],
            },
            "C2_no_https": {
                "summary": summary_row("C2_no_https", c2),
                "metrics_internal_test": c2["metrics"],
                "external_holdout": c2["external"],
                "sanity_suite": c2["sanity"],
                "acceptance_gates": c2["gates"],
                "feature_importances": c2["importances"],
                "active_feature_names": list(c2["active_names"]),
                "hyperparameters": c2["config"],
                "model_path": str(c2["path"]),
                "metadata_path": str(c2["metadata_path"]),
                "model_size_mb": c2["size_mb"],
            },
            "R2_small": {
                "summary": summary_row("R2_small", r2),
                "metrics_internal_test": r2["metrics"],
                "external_holdout": r2["external"],
                "sanity_suite": r2["sanity"],
                "acceptance_gates": r2["gates"],
                "feature_importances": r2["importances"],
                "active_feature_names": list(r2["active_names"]),
                "hyperparameters": r2["config"],
                "model_path": str(r2["path"]),
                "metadata_path": str(r2["metadata_path"]),
                "model_size_mb": r2["size_mb"],
            },
        },
        "comparison_table": rows,
        "decision": {
            "winner_feature_set": winner["label"],
            "feature_set_tie_break": tie_break,
            "uses_https_importance_c1": c1_https_importance,
            "uses_https_verdict": uses_https_verdict,
            "r2_acceptable": r2_acceptable,
            "r2_vs_winner_external_safe_delta": round(-ext_drop, 6),
            "r2_vs_winner_https_recall_delta": round(-https_drop, 6),
            "r2_vs_winner_overall_recall_delta": round(-recall_drop, 6),
            "recommended_model": promote,
            "recommendation": recommendation,
            "promotion_performed": False,
        },
    }
    write_json(COMPARISON_PATH, comparison)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Training rows          : {len(y_train):,} "
          f"(benign {train_benign:,} / malicious {train_malicious:,})")
    print(f"Tranco train domains   : {len(train_domains):,} "
          f"({len(train_repr_urls):,} generated rows)")
    print(f"Tranco holdout domains : {len(holdout_domains):,} "
          f"({len(holdout_urls):,} generated rows)")
    print(f"Pre-training gate      : {'PASS' if pretrain_gate['passed'] else 'FAIL'}")
    print(f"Leakage audit          : {'PASS' if leakage['ok'] else 'FAIL'}")
    for label, result in (("C1", c1), ("C2", c2), ("R2", r2)):
        print(
            f"{label:<22} gates {result['gates']['passed_count']}/{result['gates']['total']} | "
            f"ext-safe {result['external']['domain_safe_rate']:.4f} | "
            f"https-rec {result['metrics']['by_style']['https']['malicious_recall']:.4f} | "
            f"10-site {result['sanity']['legit_safe_count']}/10 | "
            f"{result['size_mb']:.1f} MB"
        )
    print(f"Recommended (NOT promoted): {promote}")
    print(f"Artifacts: {EXPERIMENT_DIR}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
