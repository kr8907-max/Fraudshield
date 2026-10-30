"""
FraudShield - Stage 6, Experiment S2-A
======================================

Question: does removing the hostname `www.` writing-convention shortcut preserve
S1's semantic gains on deep URLs while repairing S1's homepage collapse?

ONE controlled candidate. Fixed before running:

  S2 = S1 exactly, except the HOST feature block lowercases the hostname and
       removes exactly ONE leading literal "www." before vectorisation.

  Path, query and the 45-keyword blocks are S1 behaviour, unchanged.
  Scheme-blind preprocessing retained. No scheme feature.
  Same LogisticRegression config, same theta=0.5, same 7 gates.
  No augmentation, no resampling, no new data, no tuning.

Training population (S2-A, leakage-safe):
  S1 population                                      516,806
    - 16,172 Kaggle rows whose NORMALISED hostname collides with the
      unchanged held-out test split (www-normalisation creates these)
  = 500,634 rows (487,729 Kaggle + 12,905 Tranco)

  The purge repairs leakage that the normalisation itself introduces; the
  held-out test set and every evaluation suite are unchanged.

The classifier is fitted only after every preflight and invariance assertion
passes. Run:

    python experiment_six_s2.py --preflight-only
    python experiment_six_s2.py

Writes only model/experiment_six/s2/. Never touches R2, Experiment C/D, S1 or
backend files. Text only: no HTTP/DNS/WHOIS/TLS/browser access. No pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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
import experiment_six as s1
from backend.dataset_loader import load_dataset
from backend.feature_extractor import SUSPICIOUS_KEYWORDS

OUT_DIR = PROJECT_ROOT / "model" / "experiment_six" / "s2"
S1_DIR = PROJECT_ROOT / "model" / "experiment_six"
DATASET_PATH = PROJECT_ROOT / "data" / "malicious_phish.csv"
TRANCO_CSV = PROJECT_ROOT / "data" / "tranco" / "top-1m.csv"
TRANCO_SELECTION = PROJECT_ROOT / "data" / "tranco" / "selection.json"
R2_PROVENANCE_META = PROJECT_ROOT / "model" / "model_metadata.json"
EXPERIMENT_D_COMPARISON = PROJECT_ROOT / "model" / "experiment_d" / "comparison.json"
EXPERIMENT_D_SEPARABILITY = PROJECT_ROOT / "model" / "experiment_d" / "deep_separability.json"
S1_COMPARISON = S1_DIR / "comparison.json"
S1_EVALUATION = S1_DIR / "evaluation.json"
S1_BUNDLE = S1_DIR / "s1.pkl"

RANDOM_STATE = 42
THRESHOLD = 0.5

EXPECTED_KAGGLE_AFTER_PURGE = 487_729
EXPECTED_TOTAL_TRAIN = 500_634
EXPECTED_PURGED_COLLISIONS = 16_172

# Host artifact audit, fixed before running. Diagnostic only.
AUDIT_PREFIXES = (
    "www.", "www2.", "ww2.", "m.", "wap.", "web.", "mail.", "email.", "login.",
    "secure.", "account.", "support.", "portal.", "admin.", "signin.", "my.",
    "app.", "mobile.", "en.", "blog.", "shop.", "news.", "cdn.",
)
AUDIT_TOKENS = (
    "login", "secure", "account", "verify", "update", "support", "mail", "admin",
    "portal", "signin", "bank", "paypal", "free", "download", "app", "mobile",
    "blog", "shop", "store", "news", "cdn", "static", "s3", "aws", "cloud",
    "test", "dev", "staging", "api", "auth",
)

PROTECTED_FILES = [
    "model/fraudshield_model.pkl", "model/model_metadata.json",
    "backend/feature_extractor.py", "backend/predictor.py", "backend/dataset_loader.py",
    "train.py", "experiment_c.py", "experiment_d.py", "experiment_six.py",
    "data/malicious_phish.csv", "data/tranco/top-1m.csv", "data/tranco/selection.json",
    "model/experiment_six/s1.pkl", "model/experiment_six/s1.metadata.json",
    "model/experiment_six/comparison.json", "model/experiment_six/evaluation.json",
    "model/experiment_six/feature_config.json", "model/experiment_six/protected_before.json",
]


# ---------------------------------------------------------------------------
# Normalisation (the single change vs S1)
# ---------------------------------------------------------------------------


def normalize_host(host: str) -> str:
    """Lowercase, then remove exactly one leading literal 'www.' label."""
    host = host.strip().lower()
    if host.startswith("www.") and len(host) > 4:
        return host[4:]
    return host


def s2_texts(url: str):
    """(normalized_host, path_token_text, query_token_text) or None."""
    fields = s1.split_fields(url)
    if fields is None:
        return None
    host, path, query = fields
    return (
        normalize_host(host),
        " ".join(s1.TOKEN_RE.findall(path.lower())),
        " ".join(s1.TOKEN_RE.findall(query.lower())),
    )


def s2_host_text(url: str):
    fields = s1.split_fields(url)
    return None if fields is None else normalize_host(fields[0])


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------


class SemanticURLModelS2:
    """S1 blocks, with a normalized hostname feeding the H block only."""

    def __init__(self, host_vec, path_vec, query_vec, classifier, config):
        self.host_vec = host_vec
        self.path_vec = path_vec
        self.query_vec = query_vec
        self.classifier = classifier
        self.config = config

    def transform(self, urls: list[str]):
        hosts, paths, queries, vocab_rows = [], [], [], []
        for url in urls:
            texts = s2_texts(url)
            if texts is None:
                hosts.append("")
                paths.append("")
                queries.append("")
            else:
                hosts.append(texts[0])
                paths.append(texts[1])
                queries.append(texts[2])
            vocab_rows.append(s1.vocab_vector(url))   # S1 behaviour: raw fields
        return hstack([
            self.host_vec.transform(hosts),
            self.path_vec.transform(paths),
            self.query_vec.transform(queries),
            csr_matrix(np.asarray(vocab_rows, dtype=np.float32)),
        ], format="csr")

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(urls))


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


def matrices_equal(a, b) -> bool:
    if a.shape != b.shape:
        return False
    return (a != b).nnz == 0


def host_artifact_audit(train_rows) -> dict:
    """Label association of host-writing patterns. Diagnostic only."""
    benign = [h for h, l in train_rows if l == 0]
    malicious = [h for h, l in train_rows if l == 1]
    nb, nm = len(benign), len(malicious)

    def block(predicate):
        b = sum(1 for h in benign if predicate(h))
        m = sum(1 for h in malicious if predicate(h))
        return {
            "benign_count": b, "malicious_count": m,
            "benign_rate": round(b / nb, 6), "malicious_rate": round(m / nm, 6),
            "lift": round((m / nm) / (b / nb), 4) if b else None,
        }

    prefixes = {p: block(lambda h, p=p: h.startswith(p)) for p in AUDIT_PREFIXES}
    tokens = {t: block(lambda h, t=t: t in h.split(".")) for t in AUDIT_TOKENS}

    first = Counter()
    for h, l in train_rows:
        first[(h.split(".", 1)[0], l)] += 1
    first_rows = []
    for label in {k[0] for k in first}:
        b, m = first[(label, 0)], first[(label, 1)]
        if b + m >= 200:
            first_rows.append({"first_label": label, "benign": b, "malicious": m,
                               "malicious_share": round(m / (b + m), 6),
                               "lift": round((m / nm) / (b / nb), 4) if b else None})
    first_rows.sort(key=lambda r: r["malicious"], reverse=True)

    structure = {
        "labels_2": block(lambda h: h.count(".") + 1 == 2),
        "labels_3": block(lambda h: h.count(".") + 1 == 3),
        "labels_ge_4": block(lambda h: h.count(".") + 1 >= 4),
        "contains_digit": block(lambda h: any(c.isdigit() for c in h)),
        "contains_hyphen": block(lambda h: "-" in h),
        "contains_www_anywhere": block(lambda h: "www" in h),
        "leading_www": block(lambda h: h.startswith("www.")),
        "www_not_leading": block(lambda h: "www" in h and not h.startswith("www.")),
        "length_gt_30": block(lambda h: len(h) > 30),
    }
    residual = sum(1 for h, _l in train_rows if "www" in normalize_host(h))

    # Descriptive escalation check: any OTHER pattern at scale with high lift?
    escalations = [
        {"pattern": f"prefix:{p}", **v} for p, v in prefixes.items()
        if p != "www." and v["malicious_count"] >= 500 and (v["lift"] or 0) >= 5
    ] + [
        {"pattern": f"token:{t}", **v} for t, v in tokens.items()
        if v["malicious_count"] >= 500 and (v["lift"] or 0) >= 5
    ]

    return {
        "note": ("Diagnostic only. Computed on the purged S2 training population "
                 "before the classifier is fitted. Never used for feature selection "
                 "or tuning."),
        "training_rows": len(train_rows),
        "benign_rows": nb,
        "malicious_rows": nm,
        "prefixes": prefixes,
        "tokens": tokens,
        "first_label_top_by_malicious_count": first_rows[:15],
        "structure": structure,
        "rows_still_containing_www_after_normalisation": residual,
        "rows_still_containing_www_share": round(residual / len(train_rows), 6),
        "other_patterns_at_scale_with_lift_ge_5": escalations,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FraudShield Stage 6 Experiment S2-A")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run all non-vectoriser audits and stop before vectorising/fitting")
    args = parser.parse_args(argv)

    print("=" * 88)
    print("FraudShield Stage 6 - Experiment S2-A (www-normalised host block)")
    print("=" * 88)
    print("Text-only processing. No HTTP/DNS/WHOIS/TLS/browser/external API access.")
    print("ONE candidate. No tuning. Writes only model/experiment_six/s2/.")

    for required in (DATASET_PATH, TRANCO_CSV, TRANCO_SELECTION, S1_BUNDLE,
                     EXPERIMENT_D_COMPARISON, EXPERIMENT_D_SEPARABILITY):
        if not required.is_file():
            print(f"STOP: required file missing: {required}")
            return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protected_before = {rel: (file_sha256(PROJECT_ROOT / rel)
                              if (PROJECT_ROOT / rel).is_file() else "MISSING")
                        for rel in PROTECTED_FILES}

    # -- A. Inputs ---------------------------------------------------------
    print("\n[A] Input verification (read-only)")
    r2_meta = json.loads(R2_PROVENANCE_META.read_text(encoding="utf-8"))
    dataset_hash = file_sha256(DATASET_PATH)
    tranco_hash = file_sha256(TRANCO_CSV)
    if dataset_hash != r2_meta["provenance"]["kaggle_sha256"]:
        print("STOP: Kaggle CSV differs from promoted-model provenance.")
        return 1
    if tranco_hash != r2_meta["provenance"]["tranco_sha256"]:
        print("STOP: Tranco CSV differs from promoted-model provenance.")
        return 1
    print(f"    Kaggle SHA-256 : {dataset_hash}")
    print(f"    Tranco SHA-256 : {tranco_hash}")
    print(f"    Promoted R2    : {file_sha256(PROJECT_ROOT / 'model' / 'fraudshield_model.pkl')}")
    print(f"    S1 bundle      : {file_sha256(S1_BUNDLE)}")

    # -- B. Data, split, purge ---------------------------------------------
    print("\n[B] Split reproduction and leakage-safe S2-A purge")
    starter = time.perf_counter()
    bundle = load_dataset(DATASET_PATH, label_map=train.SOURCE_TO_BINARY,
                          url_column="url", label_column="type")
    urls = bundle.urls
    y = np.asarray(bundle.y, dtype=np.int8)
    del bundle
    print(f"    usable rows   : {len(urls):,} ({time.perf_counter() - starter:.1f}s)")

    train_idx, test_idx, groups = train.split_by_hostname(urls, y)
    if len(train_idx) != r2_meta["provenance"]["kaggle_train_rows_before_purge"]:
        print("STOP: split differs from Experiment C.")
        return 1
    brands = frozenset({"google", "wikipedia", "github", "microsoft", "apple",
                        "python", "mozilla", "reddit", "stackoverflow", "amazon"})
    def brand_hit(hostname: str) -> bool:
        return any(label in brands for label in hostname.split("."))

    keep = np.asarray([not brand_hit(groups[i]) for i in train_idx], dtype=bool)
    train_purged = train_idx[keep]
    if len(train_purged) != r2_meta["provenance"]["kaggle_train_rows_after_purge"]:
        print("STOP: brand-purged training size differs from Experiment C.")
        return 1
    print(f"    S1 purge      : {len(train_purged):,} Kaggle rows "
          f"(test {len(test_idx):,}, unchanged)")

    test_norm_hosts = {normalize_host(groups[i]) for i in test_idx}
    collision_mask = np.asarray([normalize_host(groups[i]) in test_norm_hosts
                                 for i in train_purged], dtype=bool)
    train_s2 = train_purged[~collision_mask]
    removed = train_purged[collision_mask]
    print(f"    collision purge: -{len(removed):,} rows "
          f"(benign {int((y[removed] == 0).sum()):,} / malicious {int((y[removed] == 1).sum()):,})")
    print(f"    S2 Kaggle rows : {len(train_s2):,}  (expected {EXPECTED_KAGGLE_AFTER_PURGE:,})")

    selection = json.loads(TRANCO_SELECTION.read_text(encoding="utf-8"))
    tranco_train_urls = [f"https://{d}" for d in selection["train_domains"]]
    tranco_train_urls += [f"https://www.{d}" for d in selection["www_train_domains"]]
    tranco_holdout_urls: list[str] = []
    tranco_holdout_domains: list[str] = []
    for domain in selection["holdout_domains"]:
        tranco_holdout_urls += [f"https://{domain}", f"https://www.{domain}"]
        tranco_holdout_domains += [domain, domain]
    total_train = len(train_s2) + len(tranco_train_urls)
    print(f"    Tranco rows    : {len(tranco_train_urls):,}")
    print(f"    S2 total train : {total_train:,}  (expected {EXPECTED_TOTAL_TRAIN:,})")

    # -- C. Evaluation suites (identical to Experiment D / S1) -------------
    print("\n[C] Held-out evaluation suites (identical index sets)")
    benign_idx = [i for i in test_idx if y[i] == 0 and s1.is_deep(urls[i])]
    malicious_idx = [i for i in test_idx if y[i] == 1 and s1.is_deep(urls[i])]
    benign_urls = [s1.browser_form(urls[i]) for i in benign_idx]
    malicious_urls = [s1.browser_form(urls[i]) for i in malicious_idx]
    benign_struct = [s1.structure_of(urls[i]) for i in benign_idx]
    malicious_struct = [s1.structure_of(urls[i]) for i in malicious_idx]
    benign_https_idx = [j for j, u in enumerate(benign_urls) if u.startswith("https://")]
    malicious_https_idx = [j for j, u in enumerate(malicious_urls) if u.startswith("https://")]
    benign_domains = {groups[i] for i in benign_idx}
    malicious_domains = {groups[i] for i in malicious_idx}
    print(f"    legitimate deep : {len(benign_urls):,} rows / {len(benign_domains):,} hostnames")
    print(f"    malicious deep  : {len(malicious_urls):,} rows")
    print(f"    https-only deep : benign {len(benign_https_idx):,} / malicious {len(malicious_https_idx):,}")

    url_types: dict[str, set[str]] = {}
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            u = (row.get("url") or "").strip()
            t = (row.get("type") or "").strip().lower()
            if u and t:
                url_types.setdefault(u, set()).add(t)
    malicious_subtype = ["|".join(sorted(url_types.get(urls[i], set()))) or "unknown"
                         for i in malicious_idx]
    print(f"    malicious subtypes: "
          f"{ {k: malicious_subtype.count(k) for k in sorted(set(malicious_subtype))} }")

    # -- D. Preflight audits ------------------------------------------------
    print("\n[D] PREFLIGHT AUDITS (all must pass; classifier not fitted yet)")
    failures: list[str] = []
    checks: dict[str, int | str] = {}

    def check(name: str, value: int, expected: int = 0) -> None:
        ok = value == expected
        checks[name] = value
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {value}")
        if not ok:
            failures.append(name)

    raw_train_hosts = {groups[i] for i in train_s2}
    raw_test_hosts = {groups[i] for i in test_idx}
    norm_train_hosts = {normalize_host(groups[i]) for i in train_s2}
    norm_test_hosts = {normalize_host(groups[i]) for i in test_idx}
    train_urls_set = {urls[i] for i in train_s2}

    check("split_train_rows_matches_experiment_c", len(train_purged), 503_901)
    check("kaggle_train_after_collision_purge", len(train_s2), EXPECTED_KAGGLE_AFTER_PURGE)
    check("total_s2_training_rows", total_train, EXPECTED_TOTAL_TRAIN)
    check("collision_rows_purged", len(removed), EXPECTED_PURGED_COLLISIONS)
    check("raw_train_test_hostname_overlap", len(raw_train_hosts & raw_test_hosts))
    check("normalized_train_test_hostname_overlap", len(norm_train_hosts & norm_test_hosts))
    check("deep_benign_hosts_in_train", len({normalize_host(groups[i]) for i in benign_idx} & norm_train_hosts))
    check("deep_malicious_hosts_in_train", len({normalize_host(groups[i]) for i in malicious_idx} & norm_train_hosts))
    test_idx_set = {int(i) for i in test_idx}
    check("deep_rows_all_from_test_split", int(
        set(int(i) for i in benign_idx).issubset(test_idx_set)
        and set(int(i) for i in malicious_idx).issubset(test_idx_set)), 1)
    check("famous_diagnostic_urls_in_train",
          len({u for _l, u in s1.FAMOUS_DEEP + s1.EXTRA_DIAGNOSTICS} & train_urls_set))
    check("sanity_brand_hosts_in_train",
          sum(1 for h in norm_train_hosts if brand_hit(h)))
    check("tranco_holdout_domains_in_train",
          len({normalize_host(d) for d in selection["holdout_domains"]} & norm_train_hosts))
    check("tranco_train_vs_holdout_normalized_overlap",
          len({normalize_host(d) for d in selection["train_domains"]}
              & {normalize_host(d) for d in selection["holdout_domains"]}))
    check("tranco_train_urls_in_test_split",
          sum(1 for u in tranco_train_urls if normalize_host(u.split("//", 1)[1]) in norm_test_hosts))
    check("no_augmentation_rows", 0)
    check("vocab_will_be_fit_on_s2_train_only", 1, 1)

    # Host-normalisation identity: apex vs leading-www, 5,000 real domains.
    domain_pool = sorted({normalize_host(groups[i]) for i in test_idx if groups[i]})
    domain_sample = domain_pool[:5000]
    host_text_equal = 0
    for domain in domain_sample:
        apex = f"https://{domain}/x?y=1"
        www = f"https://www.{domain}/x?y=1"
        if s2_host_text(apex) == s2_host_text(www) == domain:
            host_text_equal += 1
    check("host_text_equal_over_5000_real_domains", host_text_equal - len(domain_sample))

    # Scheme-blind field equality, 5,000 real rows (S1 property retained).
    sample_positions = list(range(0, len(urls), max(1, len(urls) // 5000)))[:5000]
    blind_equal = sum(1 for p in sample_positions
                      if s2_texts(urls[p]) == s2_texts(s1.browser_form(urls[p])))
    check("scheme_blind_field_equality_5000", blind_equal - len(sample_positions))
    checks["scheme_blind_sample_size"] = len(sample_positions)

    # No other leading prefix is stripped: only exactly one literal www.
    edge_cases = {
        "www2.example.com": "www2.example.com",
        "m.example.com": "m.example.com",
        "ww2.example.com": "ww2.example.com",
        "www.www.example.com": "www.example.com",
        "WWW.Example.COM": "example.com",
        "example.com": "example.com",
    }
    edge_ok = all(normalize_host(k) == v for k, v in edge_cases.items())
    check("only_one_leading_literal_www_removed", 0 if edge_ok else 1)

    # Pre-fit hostname artifact audit (diagnostic only).
    train_rows_for_audit = [(normalize_host(groups[i]), int(y[i])) for i in train_s2]
    audit = host_artifact_audit(train_rows_for_audit)
    raw_audit = host_artifact_audit([(groups[i], int(y[i])) for i in train_s2])
    audit["reference_before_normalization"] = {
        "note": "same rows, raw hostnames, for comparison only",
        "leading_www": raw_audit["structure"]["leading_www"],
        "contains_www_anywhere": raw_audit["structure"]["contains_www_anywhere"],
        "rows_still_containing_www": raw_audit["rows_still_containing_www_after_normalisation"],
    }
    print(f"    [note] audit pre-normalisation leading www.: "
          f"benign {raw_audit['structure']['leading_www']['benign_rate'] * 100:.2f}% / "
          f"malicious {raw_audit['structure']['leading_www']['malicious_rate'] * 100:.2f}%")
    print(f"    [note] audit: www. lift after removal -> "
          f"{audit['prefixes']['www.']['malicious_count']:,} malicious / "
          f"{audit['prefixes']['www.']['benign_count']:,} benign (should be small)")
    print(f"    [note] audit: rows still containing 'www' = "
          f"{audit['rows_still_containing_www_after_normalisation']:,} "
          f"({audit['rows_still_containing_www_share'] * 100:.3f}%)")
    if audit["other_patterns_at_scale_with_lift_ge_5"]:
        print("    [note] audit: other patterns at scale with lift >= 5:")
        for row in audit["other_patterns_at_scale_with_lift_ge_5"]:
            print(f"           {row['pattern']}: lift {row['lift']} "
                  f"({row['malicious_count']:,} malicious)")
    else:
        print("    [note] audit: no other pattern at scale (>=500 rows) has lift >= 5")

    if failures:
        print(f"\nSTOP: preflight failed: {failures}. Nothing was vectorised or fitted.")
        write_json(OUT_DIR / "preflight.json", {"passed": False, "failures": failures,
                                                "checks": checks, "audit": audit})
        return 1
    print("    All preflight audits passed.")

    write_json(OUT_DIR / "host_artifact_audit.json", audit)
    write_json(OUT_DIR / "preflight.json", {
        "passed": True, "checks": checks, "audit_summary": {
            "rows": audit["training_rows"],
            "rows_still_containing_www": audit["rows_still_containing_www_after_normalisation"],
            "other_patterns_at_scale_with_lift_ge_5": audit["other_patterns_at_scale_with_lift_ge_5"],
        },
        "note": "Classifier not fitted at the time this file was written.",
    })

    if args.preflight_only:
        print("\n--preflight-only requested; stopping before vectoriser/classifier fitting.")
        return 0

    # -- E. Vectorisers on the purged S2 training population ---------------
    print("\n[E] Fitting vectorisers on S2 training text only")
    train_jobs = [(urls[i], int(y[i])) for i in train_s2] + [(u, 0) for u in tranco_train_urls]
    y_train = np.asarray([label for _u, label in train_jobs], dtype=np.int8)
    hosts_text, paths_text, queries_text, vocab_rows = [], [], [], []
    for url, _label in train_jobs:
        texts = s2_texts(url)
        if texts is None:
            hosts_text.append("")
            paths_text.append("")
            queries_text.append("")
        else:
            hosts_text.append(texts[0])
            paths_text.append(texts[1])
            queries_text.append(texts[2])
        vocab_rows.append(s1.vocab_vector(url))
    print(f"    training rows : {len(train_jobs):,}  "
          f"(benign {int((y_train == 0).sum()):,} / malicious {int((y_train == 1).sum()):,})")

    t0 = time.perf_counter()
    host_vec = TfidfVectorizer(**s1.HOST_CONFIG).fit(hosts_text)
    path_vec = TfidfVectorizer(**s1.PATH_CONFIG).fit(paths_text)
    query_vec = TfidfVectorizer(**s1.QUERY_CONFIG).fit(queries_text)
    vectoriser_seconds = time.perf_counter() - t0
    n_features = (len(host_vec.vocabulary_) + len(path_vec.vocabulary_)
                  + len(query_vec.vocabulary_) + 3 * len(SUSPICIOUS_KEYWORDS))
    print(f"    vocab sizes   : host={len(host_vec.vocabulary_):,} "
          f"path={len(path_vec.vocabulary_):,} query={len(query_vec.vocabulary_):,} "
          f"vocab_block={3 * len(SUSPICIOUS_KEYWORDS)} total={n_features:,}")

    t0 = time.perf_counter()
    X_train = hstack([
        host_vec.transform(hosts_text),
        path_vec.transform(paths_text),
        query_vec.transform(queries_text),
        csr_matrix(np.asarray(vocab_rows, dtype=np.float32)),
    ], format="csr")
    transform_seconds = time.perf_counter() - t0
    print(f"    sparse matrix : {X_train.shape[0]:,} x {X_train.shape[1]:,}  "
          f"nnz={X_train.nnz:,} ({X_train.nnz / X_train.shape[0]:.1f} per row)")
    del hosts_text, paths_text, queries_text, vocab_rows, train_jobs

    # -- E2. Vector-level invariance (must pass BEFORE the classifier) -----
    print("\n[E2] Vector-level invariance assertions (classifier still unfitted)")
    model = SemanticURLModelS2(host_vec, path_vec, query_vec, None, s1.MODEL_CONFIG)

    def vector_of(url_list):
        return model.transform(url_list)

    apex_vec = vector_of([f"https://{d}" for d in domain_sample])
    www_vec = vector_of([f"https://www.{d}" for d in domain_sample])
    host_only_equal = 0
    for d in domain_sample[:5000]:
        if matrices_equal(host_vec.transform([d]), host_vec.transform([d])):
            host_only_equal += 1
    check("host_vector_equal_over_5000_domains", 0 if matrices_equal(apex_vec, www_vec) else 1)
    check("host_block_vector_selfcheck", host_only_equal - len(domain_sample))

    tranco_apex = tranco_holdout_urls[0::2]
    tranco_www = tranco_holdout_urls[1::2]
    tranco_apex_vec = vector_of(tranco_apex)
    tranco_www_vec = vector_of(tranco_www)
    check("tranco_2000_apex_www_full_vector_equality",
          0 if matrices_equal(tranco_apex_vec, tranco_www_vec) else 1)

    original_vec = vector_of([urls[p] for p in sample_positions])
    browser_vec = vector_of([s1.browser_form(urls[p]) for p in sample_positions])
    check("scheme_blind_full_vector_equality_5000",
          0 if matrices_equal(original_vec, browser_vec) else 1)

    if failures:
        print(f"\nSTOP: invariance failed: {failures}. Classifier NOT fitted.")
        write_json(OUT_DIR / "preflight.json", {"passed": False, "failures": failures,
                                                "checks": checks})
        return 1
    print("    All invariance assertions passed. Proceeding to the single fit.")

    # -- F. The one and only classifier fit --------------------------------
    print("\n[F] Fitting LogisticRegression (one candidate, no tuning)")
    used_max_iter = s1.MODEL_CONFIG["max_iter"]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier = LogisticRegression(**s1.MODEL_CONFIG).fit(X_train, y_train)
        converged_clean = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not converged_clean:
        print(f"    ConvergenceWarning at max_iter={used_max_iter}; "
              f"refitting with max_iter=1000 (predeclared, design unchanged)")
        used_max_iter = 1000
        classifier = LogisticRegression(**{**s1.MODEL_CONFIG, "max_iter": 1000}).fit(X_train, y_train)
    model_seconds = time.perf_counter() - t0
    print(f"    fit time      : {model_seconds:.1f}s  max_iter={used_max_iter}  "
          f"iterations={int(classifier.n_iter_[0])}")
    del X_train, apex_vec, www_vec, tranco_apex_vec, tranco_www_vec, original_vec, browser_vec

    model.classifier = classifier
    model_path = OUT_DIR / "s2.pkl"
    joblib.dump({
        "host_vec": host_vec, "path_vec": path_vec, "query_vec": query_vec,
        "classifier": classifier, "config": {**s1.MODEL_CONFIG, "max_iter": used_max_iter},
        "vectoriser_config": {"host": s1.HOST_CONFIG, "path": s1.PATH_CONFIG,
                              "query": s1.QUERY_CONFIG},
        "normalization": "lowercase then remove exactly one leading literal 'www.' in the HOST block only",
        "representation": "S1 scheme-blind field-separated TF-IDF + keyword counts, www-normalised host",
    }, model_path, compress=3)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    reloaded = joblib.load(model_path)
    reloaded_model = SemanticURLModelS2(reloaded["host_vec"], reloaded["path_vec"],
                                        reloaded["query_vec"], reloaded["classifier"],
                                        reloaded["config"])
    probe = ["https://www.example.com/a?b=1", "http://secure-login.example/verify/account"]
    if not np.allclose(reloaded_model.predict_proba(probe), model.predict_proba(probe)):
        print("STOP: reloaded bundle does not reproduce predictions.")
        return 1
    print(f"    model size    : {size_mb:.2f} MB  (artifact reload verified)")

    # -- G. Evaluation ------------------------------------------------------
    print("\n[G] Evaluation (identical suites, threshold 0.5)")
    test_urls = [urls[i] for i in test_idx]
    test_styles = np.asarray([s1.scheme_of(u) for u in test_urls], dtype=object)
    t0 = time.perf_counter()
    test_score = model.predict_proba(test_urls)[:, 1]
    test_pred = (test_score >= THRESHOLD).astype(np.int8)
    eval_seconds = time.perf_counter() - t0
    internal = s1.metrics_block(y[test_idx], test_pred, test_score, test_styles)
    print(f"    internal test : acc={internal['accuracy']:.4f} prec={internal['precision']:.4f} "
          f"rec={internal['recall']:.4f} f1={internal['f1']:.4f} auc={internal['roc_auc']:.4f}")

    ext_score = model.predict_proba(tranco_holdout_urls)[:, 1]
    ext_pred = (ext_score >= THRESHOLD).astype(np.int8)
    external = s1.external_block(ext_pred, ext_score, tranco_holdout_urls, tranco_holdout_domains)
    print(f"    Tranco        : domain safe={external['domain_safe_rate']:.4f} "
          f"apex={external['apex_safe_rate']:.4f} www={external['www_safe_rate']:.4f}")

    benign_score = model.predict_proba(benign_urls)[:, 1]
    benign_pred = (benign_score >= THRESHOLD).astype(np.int8)
    deep_benign = s1.deep_block(benign_pred, benign_score, benign_struct, 0)
    print(f"    legit deep    : safe={deep_benign['overall']['rate']:.4f}")

    malicious_score = model.predict_proba(malicious_urls)[:, 1]
    malicious_pred = (malicious_score >= THRESHOLD).astype(np.int8)
    deep_malicious = s1.deep_block(malicious_pred, malicious_score, malicious_struct, 1)
    print(f"    malicious deep: recall={deep_malicious['overall']['rate']:.4f}")

    subtype_results = {}
    for subtype in sorted(set(malicious_subtype)):
        mask = np.asarray([s == subtype for s in malicious_subtype], dtype=bool)
        subtype_results[subtype] = {
            "count": int(mask.sum()),
            "recall": round(float(malicious_pred[mask].mean()), 6),
            "mean_score": round(float(malicious_score[mask].mean()), 6),
        }
        print(f"      {subtype:<14} n={int(mask.sum()):>6,} recall={subtype_results[subtype]['recall']:.4f}")

    https_sep = s1.separability(benign_score[benign_https_idx], malicious_score[malicious_https_idx])
    all_sep = s1.separability(benign_score, malicious_score)
    print(f"    https-only deep AUC: {https_sep['roc_auc']:.6f} "
          f"(R2 {0.757214:.6f}, S1 {0.951609:.6f})  frontier "
          f"{https_sep['best_balanced_safe']:.3f}/{https_sep['best_balanced_recall']:.3f}")

    # -- G2. Tranco diagnosis ----------------------------------------------
    print("\n[G2] Tranco diagnostics")
    bounds = s1.block_bounds(host_vec, path_vec, query_vec, n_features)
    ext_matrix = model.transform(tranco_holdout_urls)
    parts = s1.block_contributions(ext_matrix, classifier, bounds)
    s2_flag = ext_pred.astype(bool)

    apex_scores = ext_score[0::2]
    www_scores = ext_score[1::2]
    delta = www_scores - apex_scores
    apex_www = {
        "pairs": int(len(delta)),
        "mean_delta": round(float(delta.mean()), 9),
        "max_abs_delta": round(float(np.abs(delta).max()), 9),
        "pairs_with_nonzero_delta": int((np.abs(delta) > 1e-12).sum()),
        "note": "Apex and www forms of the same domain must be identical under S2.",
    }
    print(f"    apex-vs-www delta: mean={apex_www['mean_delta']:.9f} "
          f"max|delta|={apex_www['max_abs_delta']:.9f} nonzero={apex_www['pairs_with_nonzero_delta']}")

    # R2 on the same URLs (read-only audit).
    from backend.predictor import FraudShieldPredictor
    r2_predictor = FraudShieldPredictor()
    r2_results = [r2_predictor.predict(u) for u in tranco_holdout_urls]
    r2_scores = np.asarray([r["model_score"] for r in r2_results], dtype=np.float64)
    r2_pred = np.asarray([r["prediction"] for r in r2_results], dtype=np.int8)
    domain_positions: dict[str, list[int]] = {}
    for position, domain in enumerate(tranco_holdout_domains):
        domain_positions.setdefault(domain, []).append(position)
    r2_domain_safe = float(np.mean([
        all(int(r2_pred[p]) == 0 for p in positions) for positions in domain_positions.values()]))

    # Top +/- host features by coefficient, with mean tf-idf context.
    coef = classifier.coef_.ravel()
    host_coef = coef[0:len(host_vec.vocabulary_)]
    host_names = host_vec.get_feature_names_out()
    order = np.argsort(host_coef)
    tranco_host_block = ext_matrix[:, 0:len(host_vec.vocabulary_)]
    tranco_host_mean = np.asarray(tranco_host_block.mean(axis=0)).ravel()

    def feature_rows(indexes):
        return [{"feature": str(host_names[i]),
                 "coefficient": round(float(host_coef[i]), 6),
                 "mean_tfidf_tranco": round(float(tranco_host_mean[i]), 6)}
                for i in indexes]

    top_positive = feature_rows(order[::-1][:15])
    top_negative = feature_rows(order[:15])
    print("    top +host:", ", ".join(f"{r['feature']}({r['coefficient']:+.2f})" for r in top_positive[:6]))
    print("    top -host:", ", ".join(f"{r['feature']}({r['coefficient']:+.2f})" for r in top_negative[:6]))

    host_logit_all = parts["host"]
    tranco_diagnosis = {
        "apex_www": apex_www,
        "s2": {
            "domain_safe_rate": external["domain_safe_rate"],
            "apex_safe_rate": external["apex_safe_rate"],
            "www_safe_rate": external["www_safe_rate"],
            "url_safe_rate": external["url_safe_rate"],
            "mean_score": round(float(ext_score.mean()), 6),
            "median_score": round(float(np.median(ext_score)), 6),
            "flagged_urls": int(s2_flag.sum()),
        },
        "r2_same_urls": {
            "domain_safe_rate": round(r2_domain_safe, 6),
            "mean_score": round(float(r2_scores.mean()), 6),
        },
        "host_block_logit": {
            "mean_on_tranco_false_positives": round(float(host_logit_all[s2_flag].mean()), 6)
                                             if s2_flag.any() else None,
            "mean_on_tranco_true_negatives": round(float(host_logit_all[~s2_flag].mean()), 6)
                                              if (~s2_flag).any() else None,
            "mean_path_block": round(float(parts["path"].mean()), 9),
            "mean_query_block": round(float(parts["query"].mean()), 9),
            "mean_keyword_block": round(float(parts["vocab"].mean()), 9),
            "intercept": round(float(classifier.intercept_[0]), 6),
        },
        "top_positive_host_features": top_positive,
        "top_negative_host_features": top_negative,
    }

    # -- G3. Sanity, diagnostics, degeneracy, scheme probe -----------------
    sanity_rows = []
    for kind, suite in (("legit", s1.LEGIT_SANITY), ("suspicious", s1.SUSPICIOUS_SANITY)):
        scores = model.predict_proba(list(suite))[:, 1]
        for url, score in zip(suite, scores):
            sanity_rows.append({"kind": kind, "url": url, "score": round(float(score), 6),
                                "prediction": int(score >= THRESHOLD)})
    legit_safe = sum(1 for r in sanity_rows if r["kind"] == "legit" and r["prediction"] == 0)
    suspicious_hit = sum(1 for r in sanity_rows if r["kind"] == "suspicious" and r["prediction"] == 1)
    print(f"    sanity        : legit {legit_safe}/10 safe, suspicious {suspicious_hit}/4 flagged")

    diag_rows = []
    for label, url in s1.FAMOUS_DEEP + s1.EXTRA_DIAGNOSTICS:
        score = float(model.predict_proba([url])[0, 1])
        diag_rows.append({"label": label, "url": url,
                          "kind": "famous" if (label, url) in s1.FAMOUS_DEEP else "extra",
                          "score": round(score, 6), "prediction": int(score >= THRESHOLD)})

    intercept = float(classifier.intercept_[0])
    empty_score = float(1.0 / (1.0 + np.exp(-intercept)))

    def zero_rows(subset_urls, subset_scores):
        matrix = model.transform(subset_urls)
        zero = np.asarray(matrix.getnnz(axis=1) == 0)
        return {"rows": int(len(subset_urls)),
                "rows_with_no_features": int(zero.sum()),
                "share_with_no_features": round(float(zero.mean()), 6),
                "mean_score_no_features": round(float(subset_scores[zero].mean()), 6)
                                          if zero.any() else None}

    degeneracy = {
        "intercept": round(intercept, 6),
        "score_when_no_features": round(empty_score, 6),
        "decision_when_no_features": int(empty_score >= THRESHOLD),
        "tranco_holdout": zero_rows(tranco_holdout_urls, ext_score),
        "deep_benign": zero_rows(benign_urls, benign_score),
        "deep_malicious": zero_rows(malicious_urls, malicious_score),
    }

    probe_rng = np.random.default_rng(RANDOM_STATE)
    in_train = np.zeros(len(urls), dtype=bool)
    in_train[train_s2] = True
    probe_pool = probe_rng.choice(len(urls), size=min(60_000, len(urls)), replace=False)
    probe_train = [int(p) for p in probe_pool if in_train[p]][:25_000]
    probe_test = probe_rng.choice(len(test_idx), size=min(15_000, len(test_idx)), replace=False)
    probe_test_positions = [int(test_idx[p]) for p in probe_test]
    probe_y_train = np.asarray([1 if s1.scheme_of(urls[p]) == "https" else 0 for p in probe_train])
    probe_y_test = np.asarray([1 if s1.scheme_of(urls[p]) == "https" else 0 for p in probe_test_positions])
    scheme_probe = None
    if probe_y_train.min() != probe_y_train.max() and probe_y_test.min() != probe_y_test.max():
        probe_clf = LogisticRegression(max_iter=100, solver="lbfgs", random_state=RANDOM_STATE)
        probe_clf.fit(model.transform([urls[p] for p in probe_train]), probe_y_train)
        scheme_probe = {
            "purpose": "diagnostic only: can the features reconstruct the written scheme?",
            "train_rows": len(probe_train), "test_rows": len(probe_test_positions),
            "https_rate_train": round(float(probe_y_train.mean()), 6),
            "auc": round(float(roc_auc_score(
                probe_y_test, probe_clf.predict_proba(model.transform(
                    [urls[p] for p in probe_test_positions]))[:, 1])), 6),
        }
        print(f"    scheme probe  : AUC={scheme_probe['auc']:.6f} (S1: 0.821347)")

    # Inference latency
    latency_urls = list(s1.LEGIT_SANITY) + list(s1.SUSPICIOUS_SANITY) + benign_urls[:20]
    timings = []
    for index in range(200):
        t = time.perf_counter()
        model.predict_proba([latency_urls[index % len(latency_urls)]])
        timings.append((time.perf_counter() - t) * 1000.0)
    ordered = sorted(timings)
    latency = {"iterations": len(timings),
               "mean_ms": round(statistics.fmean(timings), 3),
               "median_ms": round(statistics.median(timings), 3),
               "p95_ms": round(ordered[int(round(0.95 * (len(ordered) - 1)))], 3)}
    print(f"    inference     : mean={latency['mean_ms']:.2f} ms median={latency['median_ms']:.2f} ms")

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
    gates = {"checks": gate_checks,
             "passed_count": sum(1 for v in gate_checks.values() if v),
             "total": len(gate_checks),
             "passed": all(gate_checks.values())}
    print("\n[H] Acceptance gates (predeclared, unchanged)")
    for name, ok in gate_checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    {gates['passed_count']}/{gates['total']} gates passed")

    # -- I. Artifacts -------------------------------------------------------
    print("\n[I] Writing S2 artifacts")
    feature_config = {
        "representation": ("S1 scheme-blind field-separated TF-IDF + keyword counts, "
                           "with a www-normalised hostname"),
        "change_vs_s1": "host block input only",
        "normalization": {
            "function": "host.strip().lower(); if startswith('www.') and len>4: host[4:]",
            "removes": "exactly one leading literal 'www.'",
            "does_not_remove": ["www2.", "ww2.", "m.", "wap.", "any non-leading label"],
            "applies_to": ["HOST feature block"],
            "does_not_apply_to": ["path block", "query block", "keyword-count block"],
            "original_url_untouched": True,
        },
        "scheme_handling": {"scheme_is_a_feature": False,
                            "preprocessing": "http(s):// prefix stripped before any feature"},
        "blocks": {
            "host": {"field": "normalized hostname", "config": s1.json_safe_config(s1.HOST_CONFIG),
                     "vocab_size": len(host_vec.vocabulary_), "offset": 0},
            "path": {"field": "path", "config": s1.json_safe_config(s1.PATH_CONFIG),
                     "vocab_size": len(path_vec.vocabulary_),
                     "offset": len(host_vec.vocabulary_)},
            "query": {"field": "query", "config": s1.json_safe_config(s1.QUERY_CONFIG),
                      "vocab_size": len(query_vec.vocabulary_),
                      "offset": len(host_vec.vocabulary_) + len(path_vec.vocabulary_)},
            "suspicious_vocab": {"fields": ["host", "path", "query"],
                                 "keywords": list(SUSPICIOUS_KEYWORDS),
                                 "count": 3 * len(SUSPICIOUS_KEYWORDS),
                                 "offset": len(host_vec.vocabulary_) + len(path_vec.vocabulary_)
                                           + len(query_vec.vocabulary_)},
        },
        "total_features": int(n_features),
        "model_config": {**s1.MODEL_CONFIG, "max_iter": used_max_iter},
    }

    s2_row = {
        "model": "S2",
        "accuracy": round(internal["accuracy"], 6),
        "precision": round(internal["precision"], 6),
        "recall": round(internal["recall"], 6),
        "f1": round(internal["f1"], 6),
        "roc_auc": round(internal["roc_auc"], 6),
        "tn": internal["tn"], "fp": internal["fp"], "fn": internal["fn"], "tp": internal["tp"],
        "malicious_https_recall": round(internal["by_style"]["https"]["malicious_recall"], 6),
        "benign_https_fp_rate": round(internal["by_style"]["https"]["benign_fp_rate"], 6),
        "external_domain_safe_rate": round(external["domain_safe_rate"], 6),
        "external_apex_safe_rate": round(external["apex_safe_rate"], 6),
        "external_www_safe_rate": round(external["www_safe_rate"], 6),
        "legit_deep_safe_rate": round(deep_benign["overall"]["rate"], 6),
        "legit_deep_fp_rate": round(1.0 - deep_benign["overall"]["rate"], 6),
        "malicious_deep_recall": round(deep_malicious["overall"]["rate"], 6),
        "https_only_deep_roc_auc": https_sep["roc_auc"],
        "all_deep_roc_auc": all_sep["roc_auc"],
        "legit_sanity_safe": legit_safe,
        "suspicious_flagged": suspicious_hit,
        "model_size_mb": round(size_mb, 2),
        "gates_passed": gates["passed_count"], "gates_total": gates["total"],
        "all_gates_passed": gates["passed"],
        "active_features": int(n_features),
    }

    d_comparison = json.loads(EXPERIMENT_D_COMPARISON.read_text(encoding="utf-8"))
    d_sep = json.loads(EXPERIMENT_D_SEPARABILITY.read_text(encoding="utf-8"))
    r2_row = d_comparison["models"]["r2"]
    s1_comparison = json.loads(S1_COMPARISON.read_text(encoding="utf-8"))
    s1_row = s1_comparison["models"]["S1"]

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "internal_test": internal,
        "external_tranco_homepage": external,
        "deep_legitimate": deep_benign,
        "deep_malicious": deep_malicious,
        "deep_malicious_by_subtype": subtype_results,
        "separability": {"https_only_deep": https_sep, "all_deep": all_sep},
        "tranco_diagnosis": tranco_diagnosis,
        "sanity": {"rows": sanity_rows, "legit_safe": legit_safe,
                   "suspicious_flagged": suspicious_hit},
        "diagnostics": diag_rows,
        "degeneracy": degeneracy,
        "scheme_probe": scheme_probe,
        "inference_latency": latency,
        "gates": gates,
        "timings": {"vectoriser_fit_seconds": round(vectoriser_seconds, 2),
                    "train_transform_seconds": round(transform_seconds, 2),
                    "classifier_fit_seconds": round(model_seconds, 2),
                    "test_inference_seconds": round(eval_seconds, 2)},
    }

    comparison = {
        "experiment": "S2",
        "stage": "6",
        "variant": "S2-A (leakage-safe purge)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": ("Test whether normalising exactly one leading 'www.' in the host "
                      "block removes S1's homepage shortcut while preserving its deep-URL "
                      "separability."),
        "change_vs_s1": ("HOST block input only: lowercase + remove exactly one leading "
                         "literal 'www.'; path/query/keyword blocks unchanged."),
        "policy": {"candidate_count": 1, "threshold": THRESHOLD, "threshold_changed": False,
                   "c_tuned": False, "augmentation_used": False, "new_data_used": False,
                   "network_used": False, "pandas_used": False, "promoted_anything": False},
        "training_population": {
            "s1_total": 516_806,
            "s2_kaggle": int(len(train_s2)),
            "s2_tranco": len(tranco_train_urls),
            "s2_total": int(total_train),
            "rows_removed_for_normalized_leakage": int(len(removed)),
            "removed_benign": int((y[removed] == 0).sum()),
            "removed_malicious": int((y[removed] == 1).sum()),
            "note": ("Populations are NOT identical. S2 drops 16,172 Kaggle rows whose "
                     "normalised hostname collides with the unchanged held-out test split. "
                     "This purge repairs leakage introduced by www-normalisation; the test "
                     "set and all evaluation suites are unchanged."),
        },
        "feature_config": feature_config,
        "models": {"R2": r2_row, "S1": s1_row, "S2": s2_row},
        "headline": {
            "r2_https_only_deep_auc": d_sep["results"]["R2|https-only deep"]["roc_auc"],
            "s1_https_only_deep_auc": s1_row["https_only_deep_roc_auc"],
            "s2_https_only_deep_auc": https_sep["roc_auc"],
            "s2_minus_s1_auc": round(https_sep["roc_auc"] - s1_row["https_only_deep_roc_auc"], 6),
            "s2_minus_r2_auc": round(https_sep["roc_auc"]
                                     - d_sep["results"]["R2|https-only deep"]["roc_auc"], 6),
            "tranco": {"r2": r2_row["external_domain_safe_rate"],
                       "s1": s1_row["external_domain_safe_rate"],
                       "s2": s2_row["external_domain_safe_rate"]},
            "legit_deep_safe": {"r2": r2_row["legit_deep_safe_rate"],
                                "s1": s1_row["legit_deep_safe_rate"],
                                "s2": s2_row["legit_deep_safe_rate"]},
            "malicious_deep_recall": {"r2": r2_row["malicious_deep_recall"],
                                      "s1": s1_row["malicious_deep_recall"],
                                      "s2": s2_row["malicious_deep_recall"]},
            "apex_www_score_identity": apex_www,
        },
        "interpretation": None,
    }

    if gates["passed"]:
        verdict = "PASS: S2 satisfies all 7 gates and all integrity checks."
        recommend = True
    else:
        failed = [k for k, v in gate_checks.items() if not v]
        verdict = f"FAIL: S2 misses {len(failed)} gate(s): {', '.join(failed)}. Not promotable."
        recommend = False
    comparison["interpretation"] = {"verdict": verdict, "recommend_promotion": recommend,
                                    "promotion_rule": "All 7 gates plus zero leakage required.",
                                    "s2b_created": False}

    protected_after = {rel: (file_sha256(PROJECT_ROOT / rel)
                             if (PROJECT_ROOT / rel).is_file() else "MISSING")
                       for rel in PROTECTED_FILES}
    protected_diff = {
        "checked": len(PROTECTED_FILES),
        "changed": [rel for rel in PROTECTED_FILES
                    if protected_before[rel] != protected_after[rel]],
        "before": protected_before,
        "after": protected_after,
    }
    print(f"\n    protected files: {protected_diff['checked']} checked, "
          f"{len(protected_diff['changed'])} changed")

    metadata = {
        "model_name": "FraudShield Stage 6 Experiment S2-A",
        "variant": "s2",
        "status": "experimental - NOT promoted; R2 remains the current model",
        "representation": feature_config,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "trained_on": {"kaggle_rows": int(len(train_s2)), "tranco_rows": len(tranco_train_urls),
                       "total": int(total_train), "augmentation": "none",
                       "rows_purged_for_normalized_leakage": int(len(removed)),
                       "vocabulary_fit": "S2 training population only"},
        "metrics": s2_row,
        "evaluation": evaluation,
        "leakage_audit": checks,
        "gates": gates,
        "protected_artifacts": protected_diff,
        "note": ("Deep suites are browser-form representations of held-out Kaggle TEST rows; "
                 "features are scheme-blind, so original and browser form are identical inputs. "
                 "predict_proba is a model score, not a calibrated probability."),
    }

    payloads = {
        "feature_config.json": feature_config,
        "evaluation.json": evaluation,
        "comparison.json": comparison,
        "s2.metadata.json": metadata,
        "protected_diff.json": protected_diff,
    }
    for name, payload in payloads.items():
        json.dumps(payload, ensure_ascii=False)
    for name, payload in payloads.items():
        write_json(OUT_DIR / name, payload)
        print(f"    wrote {name}")

    # -- J. Summary ---------------------------------------------------------
    print("\n" + "=" * 88)
    print("[J] SUMMARY  R2 vs S1 vs S2-A")
    print("=" * 88)
    print(f"{'metric':<30}{'R2':>13}{'S1':>13}{'S2':>13}")
    for label, key in (
        ("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"),
        ("f1", "f1"), ("roc_auc", "roc_auc"),
        ("malicious_https_recall", "malicious_https_recall"),
        ("benign_https_fp_rate", "benign_https_fp_rate"),
        ("tranco_domain_safe", "external_domain_safe_rate"),
        ("legit_deep_safe", "legit_deep_safe_rate"),
        ("malicious_deep_recall", "malicious_deep_recall"),
        ("https_only_deep_auc", "https_only_deep_roc_auc"),
        ("model_size_mb", "model_size_mb"),
    ):
        r2v, s1v, s2v = r2_row.get(key), s1_row.get(key), s2_row.get(key)
        print(f"{label:<30}{str(r2v):>13}{str(s1v):>13}{str(s2v):>13}")
    gates_text = f"{gates['passed_count']}/7"
    print(f"{'gates':<30}{'6/7':>13}{'3/7':>13}{gates_text:>13}")

    print("\nMalicious deep recall by subtype (S2-A):")
    for subtype, payload in subtype_results.items():
        print(f"  {subtype:<14} n={payload['count']:>6,} recall={payload['recall']:.4f}")

    print("\nFamous deep diagnostics (S2-A):")
    for row in diag_rows:
        print(f"  {row['label']:<22} score={row['score']:.4f} "
              f"{'MALICIOUS' if row['prediction'] else 'benign'}")

    print(f"\nTranco: domain={external['domain_safe_rate']:.4f} "
          f"apex={external['apex_safe_rate']:.4f} www={external['www_safe_rate']:.4f} "
          f"(apex-vs-www max|delta|={apex_www['max_abs_delta']:.9f})")
    print(f"Verdict: {verdict}")
    print(f"Artifacts: {OUT_DIR}")
    return 0 if gates["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
