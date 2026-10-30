"""
FraudShield - Stage 4.5, Experiment A vs B
==========================================

Question: does scheme-aware augmentation of BARE BENIGN URLs remove the
"URL formatting shortcut" without hardcoding any website?

Model A = the saved baseline (model/fraudshield_model.pkl), untouched.
Model B = same split, same hyperparameters, plus training-only variants:
             bare benign URL "example.com/path"
               -> "https://example.com/path"
               -> "https://www.example.com/path"  (only when sensible)

Safety:
  - URLs are text only. No HTTP, DNS, browser, WHOIS, or TLS.
  - data/malicious_phish.csv is read only, never written.
  - Baseline model files are never overwritten.
  - All augmented rows are derived from TRAIN rows only, after splitting.
  - Any derived hostname that exists in the test set is discarded.

Run:
    python experiment_augmentation.py
"""

from __future__ import annotations

import ipaddress
import json
import platform
import re
import sys
import time
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

DATASET_PATH = PROJECT_ROOT / "data" / "malicious_phish.csv"
BASELINE_MODEL = PROJECT_ROOT / "model" / "fraudshield_model.pkl"
BASELINE_METADATA = PROJECT_ROOT / "model" / "model_metadata.json"
EXPERIMENT_MODEL = PROJECT_ROOT / "model" / "fraudshield_model_augmented.pkl"
EXPERIMENT_METADATA = PROJECT_ROOT / "model" / "model_metadata_augmented.json"

SCHEME_AT_START = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

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
FOCUS_FEATURES = (
    "uses_https",
    "num_slashes",
    "url_length",
    "hostname_length",
    "num_subdomains",
)


def style_of(url: str) -> str:
    lowered = url.strip().lower()
    if lowered.startswith("https://"):
        return "https"
    if lowered.startswith("http://"):
        return "http"
    return "bare"


def is_bare(url: str) -> bool:
    return not SCHEME_AT_START.match(url.strip())


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def can_add_www(host: str) -> bool:
    """Conservative: only two-label alphabetic domains, and not already www."""
    if not host or host.startswith("www.") or host.count(".") != 1:
        return False
    if is_ip(host):
        return False
    tld = host.rsplit(".", 1)[1]
    return tld.isalpha() and len(tld) >= 2


def vector_of(url: str) -> list[int]:
    features = extract_features(url)
    return [features[name] for name in FEATURE_NAMES]


def evaluate(model, X: np.ndarray, y: np.ndarray, styles: np.ndarray, name: str) -> dict:
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predictions": predictions,
        "scores": scores,
    }

    by_style: dict[str, dict[str, float | int]] = {}
    for scheme in ("bare", "http", "https"):
        benign_mask = (styles == scheme) & (y == 0)
        malicious_mask = (styles == scheme) & (y == 1)
        by_style[scheme] = {
            "benign_count": int(benign_mask.sum()),
            "benign_fp": int(predictions[benign_mask].sum()) if benign_mask.any() else 0,
            "benign_fp_rate": float(predictions[benign_mask].mean()) if benign_mask.any() else 0.0,
            "malicious_count": int(malicious_mask.sum()),
            "malicious_detected": int(predictions[malicious_mask].sum()) if malicious_mask.any() else 0,
            "malicious_recall": float(predictions[malicious_mask].mean()) if malicious_mask.any() else 0.0,
            "mean_score_benign": float(scores[benign_mask].mean()) if benign_mask.any() else 0.0,
            "mean_score_malicious": float(scores[malicious_mask].mean()) if malicious_mask.any() else 0.0,
        }
    metrics["by_style"] = by_style
    return metrics


def main() -> int:
    print("=" * 76)
    print("FraudShield Stage 4.5 - Experiment A (baseline) vs B (benign augmentation)")
    print("=" * 76)

    for required in (DATASET_PATH, BASELINE_MODEL, BASELINE_METADATA):
        if not required.is_file():
            print(f"Missing required file: {required}")
            return 1

    baseline_meta = json.loads(BASELINE_METADATA.read_text(encoding="utf-8"))
    dataset_hash_now = train.file_sha256(DATASET_PATH)
    dataset_unchanged = dataset_hash_now == baseline_meta.get("dataset_sha256")
    print(f"\nDataset          : {DATASET_PATH.name} (read-only)")
    print(f"Dataset SHA-256  : {'UNCHANGED vs Stage 4' if dataset_unchanged else 'CHANGED!'}")
    if not dataset_unchanged:
        print("STOP: the dataset bytes differ from training; refusing to continue.")
        return 1

    print("\n[1] Loading original dataset (no augmentation yet)...")
    started = time.perf_counter()
    bundle = load_dataset(
        DATASET_PATH,
        label_map=train.SOURCE_TO_BINARY,
        url_column="url",
        label_column="type",
    )
    X_lists, y_lists, urls = bundle.X, bundle.y, bundle.urls
    report = bundle.report
    del bundle
    print(f"    Usable rows: {len(urls):,}  ({time.perf_counter() - started:.1f}s)")

    # --- Same split as Stage 4 ------------------------------------------------
    split_seed = train.RANDOM_STATE
    train_idx, test_idx, groups = train.split_by_hostname(urls, y_lists)
    print(f"    Split: train {len(train_idx):,} / test {len(test_idx):,} (seed {split_seed})")
    if len(train_idx) != baseline_meta["training_samples"] or len(test_idx) != baseline_meta["testing_samples"]:
        print("STOP: split sizes do not match the Stage 4 baseline.")
        return 1

    test_hosts = set(groups[test_idx])
    train_hosts = set(groups[train_idx])
    if train_hosts & test_hosts:
        print("STOP: original hostname leakage.")
        return 1

    X_all = np.asarray(X_lists, dtype=np.float32)
    y_all = np.asarray(y_lists, dtype=np.int8)
    train_urls = [urls[i] for i in train_idx]
    test_urls = [urls[i] for i in test_idx]
    test_url_set = set(test_urls)
    styles = np.asarray([style_of(u) for u in test_urls], dtype=object)
    del X_lists, y_lists, urls, groups

    X_train = X_all[train_idx]
    X_test = X_all[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    del X_all, y_all

    print("\n[2] Model A: evaluating the untouched Stage 4 baseline...")
    baseline = joblib.load(BASELINE_MODEL)
    if baseline.n_features_in_ != len(FEATURE_NAMES):
        print("STOP: baseline feature count mismatch.")
        return 1
    metrics_a = evaluate(baseline, X_test, y_test, styles, "A: baseline")
    print(
        f"    acc={metrics_a['accuracy']:.4f} prec={metrics_a['precision']:.4f} "
        f"rec={metrics_a['recall']:.4f} f1={metrics_a['f1']:.4f} auc={metrics_a['roc_auc']:.4f}"
    )
    if abs(metrics_a["accuracy"] - baseline_meta["accuracy"]) > 1e-6:
        print("STOP: baseline metrics do not reproduce; split changed.")
        return 1

    # --- Augmentation (training-only) ----------------------------------------
    print("\n[3] Generating training-only benign scheme augmentation...")
    train_url_set = set(train_urls)
    augmented_urls: list[str] = []
    augmented_vectors: list[list[int]] = []
    augmented_source_labels: list[int] = []
    stats = {
        "bare_benign_train_rows": 0,
        "https_variants": 0,
        "www_variants": 0,
        "skipped_existing_url": 0,
        "skipped_test_hostname_collision": 0,
        "skipped_unparseable": 0,
    }

    for position, row_index in enumerate(train_idx):
        if y_train[position] != 0:
            continue
        original_url = train_urls[position]
        if not is_bare(original_url):
            continue
        stats["bare_benign_train_rows"] += 1
        original_host = train.hostname_of(original_url)
        if not original_host:
            stats["skipped_unparseable"] += 1
            continue

        candidates: list[tuple[str, str]] = []
        https_url = "https://" + original_url
        candidates.append(("https", https_url))
        if can_add_www(original_host):
            candidates.append(("www", "https://www." + original_url))

        for kind, candidate in candidates:
            if candidate in train_url_set:
                stats["skipped_existing_url"] += 1
                continue
            candidate_host = train.hostname_of(candidate)
            if not candidate_host:
                stats["skipped_unparseable"] += 1
                continue
            # Derived variants must never introduce a test hostname.
            if candidate_host in test_hosts:
                stats["skipped_test_hostname_collision"] += 1
                continue
            train_url_set.add(candidate)
            augmented_urls.append(candidate)
            augmented_vectors.append(vector_of(candidate))
            augmented_source_labels.append(int(y_train[position]))
            stats[f"{kind}_variants"] += 1

    stats["augmented_rows"] = len(augmented_urls)
    print(f"    Bare benign training rows considered : {stats['bare_benign_train_rows']:,}")
    print(f"    https:// variants generated          : {stats['https_variants']:,}")
    print(f"    https://www. variants generated      : {stats['www_variants']:,}")
    print(f"    Skipped (URL already in training)    : {stats['skipped_existing_url']:,}")
    print(f"    Skipped (derived host in test set)   : {stats['skipped_test_hostname_collision']:,}")
    print(f"    Total augmented rows                 : {stats['augmented_rows']:,}")

    # --- Leakage verification -------------------------------------------------
    augmented_hosts = {train.hostname_of(url) for url in augmented_urls}
    host_overlap = (train_hosts | augmented_hosts) & test_hosts
    exact_overlap = set(augmented_urls) & test_url_set
    all_benign = all(label == 0 for label in augmented_source_labels)
    print("\n[4] Leakage verification")
    print(f"    Augmented+train hostnames overlapping test : {len(host_overlap)}")
    print(f"    Augmented URLs present in test set         : {len(exact_overlap)}")
    print(f"    Augmented rows derived from benign rows    : {all_benign}")
    if host_overlap or exact_overlap or not all_benign:
        print("STOP: leakage check failed.")
        return 1

    # Disclosure: derived rows could coincide with sanity URLs when the dataset
    # contains the bare form. Report it so the sanity suite is judged honestly.
    augmented_set = set(augmented_urls)
    sanity_generated = [url for url in LEGIT_SANITY if url in augmented_set]
    print(f"    Legit sanity URLs generated by augmentation: {len(sanity_generated)}")
    for url in sanity_generated:
        print(f"      {url}")

    X_train_aug = np.vstack(
        [X_train, np.asarray(augmented_vectors, dtype=np.float32)]
    )
    y_train_aug = np.concatenate(
        [y_train, np.asarray(augmented_source_labels, dtype=np.int8)]
    )
    del augmented_vectors
    print(
        f"    Training rows: {len(y_train):,} -> {len(y_train_aug):,} "
        f"(+{len(y_train_aug) - len(y_train):,})"
    )

    # --- Model B --------------------------------------------------------------
    print("\n[5] Training Model B (same hyperparameters, seed 42)...")
    model_b = RandomForestClassifier(
        n_estimators=200,
        random_state=train.RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    train_start = time.perf_counter()
    model_b.fit(X_train_aug, y_train_aug)
    seconds_b = time.perf_counter() - train_start
    print(f"    Training time: {seconds_b:.1f}s")
    nodes_b = sum(tree.tree_.node_count for tree in model_b.estimators_)
    print(f"    Trees: {len(model_b.estimators_):,}  total nodes: {nodes_b:,}")

    metrics_b = evaluate(model_b, X_test, y_test, styles, "B: augmented")
    print(
        f"    acc={metrics_b['accuracy']:.4f} prec={metrics_b['precision']:.4f} "
        f"rec={metrics_b['recall']:.4f} f1={metrics_b['f1']:.4f} auc={metrics_b['roc_auc']:.4f}"
    )

    # --- Comparison -----------------------------------------------------------
    print("\n[6] Held-out comparison (identical test set)")
    print(f"    {'Metric':<12}{'A baseline':>14}{'B augmented':>14}{'Delta':>12}")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(
            f"    {key:<12}{metrics_a[key]:>14.4f}{metrics_b[key]:>14.4f}"
            f"{metrics_b[key] - metrics_a[key]:>+12.4f}"
        )
    print(
        f"    {'false_pos':<12}{metrics_a['fp']:>14,}{metrics_b['fp']:>14,}"
        f"{metrics_b['fp'] - metrics_a['fp']:>+12,}"
    )
    print(
        f"    {'false_neg':<12}{metrics_a['fn']:>14,}{metrics_b['fn']:>14,}"
        f"{metrics_b['fn'] - metrics_a['fn']:>+12,}"
    )
    print(
        f"    {'TP':<12}{metrics_a['tp']:>14,}{metrics_b['tp']:>14,}"
        f"{metrics_b['tp'] - metrics_a['tp']:>+12,}"
    )
    print(
        f"    {'TN':<12}{metrics_a['tn']:>14,}{metrics_b['tn']:>14,}"
        f"{metrics_b['tn'] - metrics_a['tn']:>+12,}"
    )

    print("\n[7] Performance by original URL style")
    print(f"    {'style':<7}{'group':<11}{'n':>9}{'A flagged':>12}{'B flagged':>12}")
    for scheme in ("bare", "http", "https"):
        a_style = metrics_a["by_style"][scheme]
        b_style = metrics_b["by_style"][scheme]
        print(
            f"    {scheme:<7}{'benign':<11}{a_style['benign_count']:>9,}"
            f"{a_style['benign_fp_rate'] * 100:>11.2f}%{b_style['benign_fp_rate'] * 100:>11.2f}%"
        )
        print(
            f"    {scheme:<7}{'malicious':<11}{a_style['malicious_count']:>9,}"
            f"{a_style['malicious_recall'] * 100:>11.2f}%{b_style['malicious_recall'] * 100:>11.2f}%"
        )

    # --- Feature importance comparison ---------------------------------------
    importance_a = dict(zip(FEATURE_NAMES, baseline.feature_importances_.tolist()))
    importance_b = dict(zip(FEATURE_NAMES, model_b.feature_importances_.tolist()))
    print("\n[8] Feature importance comparison")
    print(f"    {'feature':<26}{'A':>10}{'B':>10}{'Delta':>10}")
    for name in sorted(FEATURE_NAMES, key=lambda n: importance_b[n], reverse=True):
        print(
            f"    {name:<26}{importance_a[name]:>10.4f}{importance_b[name]:>10.4f}"
            f"{importance_b[name] - importance_a[name]:>+10.4f}"
        )

    # --- Focus-feature diagnostic (summary statistics only) -------------------
    print("\n[9] Focus features: sanity vectors vs training class means (original rows)")
    benign_mask = y_train == 0
    malicious_mask = y_train == 1
    benign_mean_vector = X_train[benign_mask].mean(axis=0)
    malicious_mean_vector = X_train[malicious_mask].mean(axis=0)
    print(
        f"    {'feature':<20}{'benign mean':>13}{'malicious mean':>16}"
        f"{'google':>9}{'wikipedia':>11}{'github':>9}"
    )
    focus_urls = LEGIT_SANITY[:3]
    focus_vectors = [extract_features(u) for u in focus_urls]
    for name in FOCUS_FEATURES:
        index = FEATURE_NAMES.index(name)
        values = [features[name] for features in focus_vectors]
        print(
            f"    {name:<20}{benign_mean_vector[index]:>13.3f}"
            f"{malicious_mean_vector[index]:>16.3f}"
            f"{values[0]:>9}{values[1]:>11}{values[2]:>9}"
        )

    # --- Sanity suite ---------------------------------------------------------
    print("\n[10] Sanity suite (not used for any metric)")
    print(
        f"    {'kind':<11}{'A pred':>8}{'A score':>9}{'B pred':>8}{'B score':>9}"
        f"{'host in train?':>16}  URL"
    )
    sanity_rows = []
    for kind, suite in (("legit", LEGIT_SANITY), ("suspicious", SUSPICIOUS_SANITY)):
        for url in suite:
            vector = np.asarray([vector_of(url)], dtype=np.float32)
            a_pred = int(baseline.predict(vector)[0])
            a_score = float(baseline.predict_proba(vector)[0, 1])
            b_pred = int(model_b.predict(vector)[0])
            b_score = float(model_b.predict_proba(vector)[0, 1])
            host_seen = "yes" if train.hostname_of(url) in train_hosts else "no"
            print(
                f"    {kind:<11}{a_pred:>8}{a_score:>9.4f}{b_pred:>8}{b_score:>9.4f}"
                f"{host_seen:>16}  {url}"
            )
            sanity_rows.append(
                {
                    "kind": kind,
                    "url": url,
                    "model_a_prediction": a_pred,
                    "model_a_score": round(a_score, 6),
                    "model_b_prediction": b_pred,
                    "model_b_score": round(b_score, 6),
                    "hostname_seen_in_training": host_seen == "yes",
                }
            )

    legit_a = sum(1 for row in sanity_rows if row["kind"] == "legit" and row["model_a_prediction"] == 0)
    legit_b = sum(1 for row in sanity_rows if row["kind"] == "legit" and row["model_b_prediction"] == 0)
    susp_a = sum(1 for row in sanity_rows if row["kind"] == "suspicious" and row["model_a_prediction"] == 1)
    susp_b = sum(1 for row in sanity_rows if row["kind"] == "suspicious" and row["model_b_prediction"] == 1)
    print(f"\n    Legit URLs classified safe     : A {legit_a}/10   B {legit_b}/10")
    print(f"    Suspicious URLs flagged risky  : A {susp_a}/4   B {susp_b}/4")

    # --- Save experimental artifacts only -------------------------------------
    print("\n[11] Saving experimental Model B (baseline untouched)...")
    joblib.dump(model_b, EXPERIMENT_MODEL, compress=3)
    reloaded = joblib.load(EXPERIMENT_MODEL)
    if reloaded.n_features_in_ != len(FEATURE_NAMES):
        print("STOP: saved experimental model failed feature check.")
        return 1

    metadata = {
        "model_name": "FraudShield URL Risk Model - Experiment B (benign scheme augmentation)",
        "model_type": "RandomForestClassifier",
        "status": "experimental - not approved for deployment",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_filename": DATASET_PATH.name,
        "dataset_sha256": dataset_hash_now,
        "dataset_unchanged_vs_baseline": dataset_unchanged,
        "dataset_original_rows": report.total_rows,
        "dataset_usable_rows": report.usable_rows,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "random_state": train.RANDOM_STATE,
        "split_strategy": baseline_meta["split_strategy"],
        "original_training_samples": int(len(y_train)),
        "augmented_training_samples": int(len(y_train_aug)),
        "augmentation": {
            "applied_to": "training-only bare benign URLs",
            "rules": [
                "https://<original>",
                "https://www.<original> only for two-label alphabetic hosts "
                "that do not already start with www.",
            ],
            "stats": stats,
            "derived_hostnames_overlapping_test": len(host_overlap),
            "derived_urls_present_in_test": len(exact_overlap),
            "sanity_urls_generated_by_augmentation": sanity_generated,
        },
        "testing_samples": int(len(y_test)),
        "hostname_overlap": 0,
        "training_seconds": round(seconds_b, 2),
        "metrics_model_a": {
            key: round(metrics_a[key], 6)
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        } | {"false_positives": metrics_a["fp"], "false_negatives": metrics_a["fn"]},
        "metrics_model_b": {
            key: round(metrics_b[key], 6)
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        } | {"false_positives": metrics_b["fp"], "false_negatives": metrics_b["fn"]},
        "confusion_matrix_model_b": [[metrics_b["tn"], metrics_b["fp"]], [metrics_b["fn"], metrics_b["tp"]]],
        "by_style_model_a": metrics_a["by_style"],
        "by_style_model_b": metrics_b["by_style"],
        "feature_importances_model_a": {name: round(value, 6) for name, value in importance_a.items()},
        "feature_importances_model_b": {name: round(value, 6) for name, value in importance_b.items()},
        "sanity_suite": sanity_rows,
    }
    EXPERIMENT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    size_mb = EXPERIMENT_MODEL.stat().st_size / (1024 * 1024)
    print(f"    Experimental model   : {EXPERIMENT_MODEL} ({size_mb:.1f} MB)")
    print(f"    Experimental metadata: {EXPERIMENT_METADATA}")
    print("    Baseline files       : untouched")

    print("\n" + "=" * 76)
    print("Experiment complete. Review the comparison before any deployment decision.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
