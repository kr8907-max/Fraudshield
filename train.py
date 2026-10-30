"""
FraudShield - Stage 4: train and evaluate the URL risk model
============================================================

Pipeline:
    data/malicious_phish.csv
        -> backend.dataset_loader (validate, de-duplicate, extract features)
        -> hostname-grouped train/test split
        -> RandomForestClassifier
        -> held-out evaluation
        -> model/fraudshield_model.pkl
        -> model/model_metadata.json

Rules honoured here:
    - No pandas.
    - No network requests; every URL is treated as plain text.
    - Feature order comes only from FEATURE_NAMES.
    - data/test_urls.csv is refused: it is a pipeline fixture, not training data.

Run:
    python train.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
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
from sklearn.model_selection import StratifiedGroupKFold

# Allow "python train.py" from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.dataset_loader import load_dataset
from backend.feature_extractor import FEATURE_NAMES, extract_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "data" / "malicious_phish.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "fraudshield_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_FRACTION = 0.20          # StratifiedGroupKFold uses 5 folds -> 20% held out
N_SPLITS = 5

# Kaggle source classes -> FraudShield binary labels.
SOURCE_TO_BINARY: dict[str, int] = {
    "benign": 0,
    "phishing": 1,
    "defacement": 1,
    "malware": 1,
}
SOURCE_CLASS_ORDER = ("benign", "phishing", "defacement", "malware")

# Reserved/non-live sanity-check strings. Never visited, never counted in metrics.
SANITY_URLS = (
    "https://www.google.com",
    "https://www.wikipedia.org",
    "http://secure-login.example/verify/account",
    "http://192.0.2.123/login/verify/password",
)

_SCHEME_AT_START = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Fingerprint the dataset so metadata can prove which bytes were used."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hostname_of(url: str) -> str:
    """
    Extract a grouping hostname using the same scheme-tolerance as the
    feature extractor.

    Limitation: this is the HOSTNAME, not the registrable domain. Without a
    Public Suffix List dependency, "a.example.co.uk" and "b.example.co.uk"
    are different groups even though they belong to the same site. Grouping
    by hostname is still far safer than a purely random row split, because a
    hostname can never appear in both sets.
    """
    text = url.strip()
    candidate = text if _SCHEME_AT_START.match(text) else "https://" + text
    try:
        return (urlsplit(candidate).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def binary_counts(labels) -> tuple[int, int]:
    """Return (benign_count, malicious_count) for an iterable of 0/1 labels."""
    counts = Counter(int(value) for value in labels)
    return counts.get(0, 0), counts.get(1, 0)


def pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.2f}%" if whole else "0.00%"


def split_by_hostname(urls: list[str], y: list[int]):
    """
    Deterministic, group-aware 80/20 split.

    StratifiedGroupKFold keeps every hostname entirely in one fold and tries
    to preserve the benign/malicious ratio. We hold out the first of 5 folds.
    """
    groups = np.asarray([hostname_of(url) for url in urls], dtype=object)
    labels = np.asarray(y, dtype=np.int8)
    placeholder = np.zeros((len(urls), 1), dtype=np.uint8)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(placeholder, labels, groups))
    return train_idx, test_idx, groups


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 68)
    print("FraudShield - Random Forest training (Stage 4)")
    print("=" * 68)

    # --- Guard: never train on the development fixture ----------------------
    if not DATASET_PATH.is_file():
        print(f"Dataset not found: {DATASET_PATH}")
        return 1
    if DATASET_PATH.resolve() == (PROJECT_ROOT / "data" / "test_urls.csv").resolve():
        print("Refusing to train: data/test_urls.csv is a test fixture, not a dataset.")
        return 1

    size_mb = DATASET_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDataset : {DATASET_PATH}")
    print(f"Size    : {size_mb:.1f} MB")
    print("Safety  : URLs are processed as text only (no requests, no DNS).")

    print("\n[1/6] Reading, validating, cleaning, extracting features...")
    print("      This may take a minute on the full dataset.")
    dataset_start = time.perf_counter()
    bundle = load_dataset(
        DATASET_PATH,
        label_map=SOURCE_TO_BINARY,
        url_column="url",
        label_column="type",
        progress_every=100_000,
    )
    prep_seconds = time.perf_counter() - dataset_start

    report = bundle.report
    print()
    print("Dataset cleaning summary")
    print("-" * 62)
    print(f"  Original rows      : {report.total_rows:,}")
    print(f"  Blank rows skipped : {report.blank_rows:,}")
    print(f"  Invalid/rejected   : {report.invalid_rows:,}")
    print(f"  Duplicates removed : {report.duplicate_rows_removed:,}")
    print(f"  Conflicting URLs   : {len(report.conflicts):,} "
          f"({report.conflict_rows_removed:,} rows excluded)")
    print(f"  Usable rows        : {report.usable_rows:,}")
    print("  (Individual dataset URLs are intentionally not printed.)")

    if report.usable_rows < 1_000:
        print("\nRefusing to train: fewer than 1,000 usable rows.")
        return 1
    if len(set(bundle.y)) < 2:
        print("\nRefusing to train: only one class present after cleaning.")
        return 1

    # Original four-class counts straight from the CSV rows we just read.
    print("\nOriginal source-class distribution (all non-blank rows):")
    for source_class in SOURCE_CLASS_ORDER:
        count = report.raw_label_counts.get(source_class, 0)
        print(f"  {source_class:<12}{count:>9,}  ({pct(count, report.total_rows)})")
    other = {
        key: value
        for key, value in report.raw_label_counts.items()
        if key not in SOURCE_CLASS_ORDER
    }
    if other:
        print(f"  unexpected  {other}")

    print("\nUsable source-class distribution (after cleaning):")
    for source_class in SOURCE_CLASS_ORDER:
        count = report.usable_source_counts.get(source_class, 0)
        print(f"  {source_class:<12}{count:>9,}  ({pct(count, report.usable_rows)})")

    print("\nBinary distribution used for training:")
    print(f"  benign    (0) {report.safe_count:>9,}  ({pct(report.safe_count, report.usable_rows)})")
    print(f"  malicious (1) {report.malicious_count:>9,}  ({pct(report.malicious_count, report.usable_rows)})")

    X_lists, y_lists, urls = bundle.X, bundle.y, bundle.urls
    if any(len(row) != len(FEATURE_NAMES) for row in X_lists):
        print("\nFeature length mismatch: refusing to train.")
        return 1
    print(f"\nFeature matrix: X=({len(X_lists):,}, {len(FEATURE_NAMES)})  y=({len(y_lists):,})")
    print(f"Data preparation time: {prep_seconds:.1f}s")
    del bundle

    # --- Split -------------------------------------------------------------
    print("\n[2/6] Hostname-grouped split (StratifiedGroupKFold)...")
    train_idx, test_idx, groups = split_by_hostname(urls, y_lists)

    train_hosts = set(groups[train_idx])
    test_hosts = set(groups[test_idx])
    overlap = train_hosts & test_hosts
    if overlap:
        print(f"STOP: hostname leakage detected in {len(overlap)} hostnames.")
        return 1

    X_all = np.asarray(X_lists, dtype=np.float32)
    y_all = np.asarray(y_lists, dtype=np.int8)
    del X_lists, y_lists

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    del X_all, y_all, groups, urls

    train_benign, train_malicious = binary_counts(y_train)
    test_benign, test_malicious = binary_counts(y_test)

    print(f"  Training URLs : {len(y_train):,}")
    print(f"    benign      : {train_benign:,} ({pct(train_benign, len(y_train))})")
    print(f"    malicious   : {train_malicious:,} ({pct(train_malicious, len(y_train))})")
    print(f"  Testing URLs  : {len(y_test):,}")
    print(f"    benign      : {test_benign:,} ({pct(test_benign, len(y_test))})")
    print(f"    malicious   : {test_malicious:,} ({pct(test_malicious, len(y_test))})")
    print(f"  Unique train hostnames: {len(train_hosts):,}")
    print(f"  Unique test hostnames : {len(test_hosts):,}")
    print("  Hostname overlap      : 0")

    # --- Train -------------------------------------------------------------
    print("\n[3/6] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - train_start
    print(f"  Training time: {train_seconds:.1f}s")
    total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
    print(f"  Trees: {len(model.estimators_):,}  total nodes: {total_nodes:,}")

    if model.n_features_in_ != len(FEATURE_NAMES):
        print(f"STOP: model expects {model.n_features_in_} features, contract is {len(FEATURE_NAMES)}.")
        return 1

    # --- Evaluate ----------------------------------------------------------
    print("\n[4/6] Evaluating on the held-out test set (positive class = malicious)...")
    y_pred = model.predict(X_test)
    malicious_scores = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, malicious_scores)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print("\n  Confusion matrix (rows = actual, columns = predicted):")
    print("                 benign    malicious")
    print(f"    benign    {tn:>9,} {fp:>12,}")
    print(f"    malicious {fn:>9,} {tp:>12,}")
    print(f"\n  True negatives : {tn:,}")
    print(f"  False positives: {fp:,}  (legitimate URL flagged as malicious)")
    print(f"  False negatives: {fn:,}  (malicious URL classified as safe)")
    print(f"  True positives : {tp:,}")
    print("\n  Interpretation: in a security product, false negatives are the costly")
    print("  error here, so malicious recall matters more than raw accuracy.")

    # --- Feature importance ------------------------------------------------
    print("\n[5/6] Feature importance (Random Forest):")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for rank, (name, value) in enumerate(importances, start=1):
        print(f"  {rank:>2}. {name:<26}{value:.4f}")
    print("  Note: importance shows how useful a feature was to this model,")
    print("        not a causal relationship with fraud.")

    # --- Save --------------------------------------------------------------
    print("\n[6/6] Saving model and metadata...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    reloaded = joblib.load(MODEL_PATH)
    if reloaded.n_features_in_ != len(FEATURE_NAMES):
        print("STOP: saved model failed the feature-count check.")
        return 1
    if int(reloaded.predict(X_test[:1])[0]) != int(y_pred[0]):
        print("STOP: saved model produced a different prediction than the trained model.")
        return 1

    metadata = {
        "model_name": "FraudShield URL Risk Model",
        "model_type": "RandomForestClassifier",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_filename": DATASET_PATH.name,
        "dataset_size_bytes": DATASET_PATH.stat().st_size,
        "dataset_sha256": file_sha256(DATASET_PATH),
        "dataset_source": (
            "Kaggle Malicious URLs Dataset (sid321axn/malicious-urls-dataset), "
            "file malicious_phish.csv"
        ),
        "dataset_original_rows": report.total_rows,
        "dataset_usable_rows": report.usable_rows,
        "invalid_rows": report.invalid_rows,
        "duplicates_removed": report.duplicate_rows_removed,
        "conflicting_urls": len(report.conflicts),
        "conflict_rows_removed": report.conflict_rows_removed,
        "source_class_counts_original": {
            name: report.raw_label_counts.get(name, 0) for name in SOURCE_CLASS_ORDER
        },
        "source_class_counts_usable": {
            name: report.usable_source_counts.get(name, 0) for name in SOURCE_CLASS_ORDER
        },
        "class_mapping": {
            "source_to_binary": SOURCE_TO_BINARY,
            "binary": {"0": "benign/safe", "1": "malicious/suspicious"},
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "training_samples": int(len(y_train)),
        "testing_samples": int(len(y_test)),
        "train_class_counts": {"benign": train_benign, "malicious": train_malicious},
        "test_class_counts": {"benign": test_benign, "malicious": test_malicious},
        "random_state": RANDOM_STATE,
        "split_strategy": (
            "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); "
            "first fold held out (~20%). Groups are hostnames parsed with "
            "urllib.parse; hostname is not a registrable domain because no "
            "Public Suffix List dependency is used."
        ),
        "hostname_overlap": 0,
        "unique_train_hostnames": len(train_hosts),
        "unique_test_hostnames": len(test_hosts),
        "training_seconds": round(train_seconds, 2),
        "data_preparation_seconds": round(prep_seconds, 2),
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "feature_importances": [
            {"feature": name, "importance": round(float(value), 6)}
            for name, value in importances
        ],
    }
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    model_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  Model   : {MODEL_PATH}  ({model_mb:.1f} MB)")
    print(f"  Metadata: {METADATA_PATH}")

    # --- Sanity checks (never used for metrics) ----------------------------
    print("\nSanity checks on reserved test strings (NOT part of accuracy):")
    for url in SANITY_URLS:
        vector = np.asarray(
            [[extract_features(url)[name] for name in FEATURE_NAMES]], dtype=np.float32
        )
        prediction = int(model.predict(vector)[0])
        score = float(model.predict_proba(vector)[0, 1])
        label = "malicious" if prediction == 1 else "benign"
        print(f"  {url}")
        print(f"    predicted: {prediction} ({label})   malicious score: {score:.4f}")

    print("\nScore note: predict_proba() is a MODEL RISK SCORE, not a calibrated")
    print("real-world probability of danger. Calibration can be added later.")
    print("\n" + "=" * 68)
    print("Done. Stage 4 complete - no API or frontend changes were made.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
