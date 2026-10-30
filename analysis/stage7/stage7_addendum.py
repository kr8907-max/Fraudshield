"""
Stage 7 addendum - READ-ONLY.

Adds:
  * the legitimate-deep suite measured in its EVALUATED browser form
    (https:// added to bare rows), which is how S1/S2/R2 scored it;
  * within-subtype scheme profiles (https vs http vs bare) with read-only
    model scores, to separate subtype-driven from formatting-driven failure;
  * train-host membership of each scheme profile (label ambiguity check).

No fitting, no network, no pandas, no artifact writes outside analysis/stage7/.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
import experiment_six as s1  # noqa: E402
import experiment_six_s2 as s2  # noqa: E402

sys.path.insert(0, str(ROOT / "analysis" / "stage7"))
import stage7_audit as a  # noqa: E402

OUT = ROOT / "analysis" / "stage7" / "stage7_subgroups.json"


def load_models():
    import joblib
    r2 = joblib.load(a.R2_MODEL)
    s1b = joblib.load(a.S1_BUNDLE)
    s2b = joblib.load(a.S2_BUNDLE)
    m1 = s1.SemanticURLModel(s1b["host_vec"], s1b["path_vec"], s1b["query_vec"],
                             s1b["classifier"], s1b["config"])
    m2 = s2.SemanticURLModelS2(s2b["host_vec"], s2b["path_vec"], s2b["query_vec"],
                               s2b["classifier"], s2b["config"])
    return r2, m1, m2, s1b


def profile(positions, urls, subtypes, labels, train_hosts, benign_train_hosts,
            malicious_train_hosts, r2, m1, m2, host_vocab, host_analyzer):
    subset = [urls[i] for i in positions]
    y = np.asarray([labels[i] for i in positions])
    if not subset:
        return {"rows": 0}
    X = np.asarray([a.extract_feature_vector(u) for u in subset], dtype=np.int16)
    r2s = r2.predict_proba(X)[:, 1]
    s1s = m1.predict_proba(subset)[:, 1]
    s2s = m2.predict_proba(subset)[:, 1]

    tlds = Counter()
    host_lengths = []
    depths = []
    www = 0
    hosts = set()
    in_train = 0
    in_benign_train = 0
    in_malicious_train = 0
    cov = []
    susp = 0
    for u in subset:
        parsed = a.parse_url(u)
        if parsed is None:
            continue
        host, path, query = parsed
        hosts.add(host)
        tlds[host.rsplit(".", 1)[-1] if "." in host else host] += 1
        host_lengths.append(len(host))
        depths.append(a.path_depth(path))
        www += int(host.startswith("www."))
        in_train += int(host in train_hosts)
        in_benign_train += int(host in benign_train_hosts)
        in_malicious_train += int(host in malicious_train_hosts)
        grams = host_analyzer(host)
        if grams:
            cov.append(sum(1 for g in grams if g in host_vocab) / len(grams))
        susp += int(any(k in host.lower() or k in path.lower() or k in query.lower()
                        for k in __import__("backend.feature_extractor",
                                            fromlist=["SUSPICIOUS_KEYWORDS"]).SUSPICIOUS_KEYWORDS))
    return {
        "rows": len(subset),
        "distinct_hosts": len(hosts),
        "share_www": a.share(www, len(subset)),
        "host_length_median": a.pctl(sorted(host_lengths), 0.50),
        "host_length_p90": a.pctl(sorted(host_lengths), 0.90),
        "path_depth_median": a.pctl(sorted(depths), 0.50),
        "path_depth_p90": a.pctl(sorted(depths), 0.90),
        "host_ngram_coverage_mean": round(float(np.mean(cov)), 6) if cov else None,
        "share_suspicious_keyword": a.share(susp, len(subset)),
        "top_tlds": [{"value": k, "count": v} for k, v in tlds.most_common(8)],
        "share_host_in_train": a.share(in_train, len(subset)),
        "share_host_in_benign_train": a.share(in_benign_train, len(subset)),
        "share_host_in_malicious_train": a.share(in_malicious_train, len(subset)),
        "models": {
            name: {
                "recall_at_0_5": round(float((score >= 0.5).mean()), 6),
                "mean_score": round(float(score.mean()), 6),
                "p10": round(float(np.quantile(score, 0.10)), 6),
                "median": round(float(np.median(score)), 6),
                "p90": round(float(np.quantile(score, 0.90)), 6),
            }
            for name, score in (("R2", r2s), ("S1", s1s), ("S2-A", s2s))
        },
    }


def browser_form(url):
    text = url.strip()
    return text if s1.SCHEME_RE.match(text) else "https://" + text


def main():
    t0 = time.perf_counter()
    rows, _ = a.load_corpus()
    urls = [r["url"] for r in rows]
    subtypes = [r["subtype"] for r in rows]
    labels = [r["label"] for r in rows]
    train_idx, test_idx, groups = train.split_by_hostname(urls, labels)
    keep = np.asarray([not s1.brand_hit(groups[i]) for i in train_idx], dtype=bool)
    train_purged = train_idx[keep]
    test_idx = np.asarray(test_idx)

    train_hosts = {a.parse_url(urls[i])[0] for i in train_purged
                   if a.parse_url(urls[i])}
    benign_train_hosts = {a.parse_url(urls[i])[0] for i in train_purged
                          if labels[i] == 0 and a.parse_url(urls[i])}
    malicious_train_hosts = {a.parse_url(urls[i])[0] for i in train_purged
                             if labels[i] == 1 and a.parse_url(urls[i])}

    r2, m1, m2, s1b = load_models()
    host_vocab = s1b["host_vec"].vocabulary_
    host_analyzer = s1b["host_vec"].build_analyzer()

    # --- Evaluated deep suite (browser form) ------------------------------
    deep_positions = [i for i in test_idx if labels[i] == 0 and s1.is_deep(urls[i])]
    deep_urls = [browser_form(urls[i]) for i in deep_positions]
    # Reuse coverage machinery with synthetic positions.
    base = len(urls)
    all_urls = urls + deep_urls
    all_sub = subtypes + ["benign"] * len(deep_urls)
    all_lab = labels + [0] * len(deep_urls)
    deep_idx = list(range(base, base + len(deep_urls)))
    evaluated = a.benign_coverage(deep_idx, all_urls, all_sub, all_lab,
                                  "legit deep suite in evaluated browser form", [])

    # Counterfactual: the same Kaggle benign deep rows if written with https.
    kaggle_benign_deep = [i for i in train_purged
                          if labels[i] == 0 and s1.is_deep(urls[i])]
    cf_urls = urls + [browser_form(urls[i]) for i in kaggle_benign_deep]
    cf_sub = subtypes + ["benign"] * len(kaggle_benign_deep)
    cf_lab = labels + [0] * len(kaggle_benign_deep)
    cf_idx = list(range(len(urls), len(cf_urls)))
    counterfactual = a.benign_coverage(
        cf_idx, cf_urls, cf_sub, cf_lab,
        "Kaggle benign deep TRAIN rows if every bare row were written https://", [])

    # --- Within-subtype scheme profiles on the fixed test split -----------
    profiles = {}
    for subtype in ("phishing", "malware", "defacement"):
        profiles[subtype] = {}
        for scheme in a.SCHEMES:
            positions = [i for i in test_idx
                         if subtypes[i] == subtype and s1.scheme_of(urls[i]) == scheme]
            profiles[subtype][scheme] = profile(
                positions, urls, subtypes, labels, train_hosts,
                benign_train_hosts, malicious_train_hosts, r2, m1, m2,
                host_vocab, host_analyzer)
            print(f"      {subtype}/{scheme}: {profiles[subtype][scheme].get('rows')}")

    payload = {
        "meta": {
            "stage": 7, "addendum": True, "mode": "READ-ONLY",
            "no_fit": True, "no_network": True, "no_pandas": True,
            "seconds": round(time.perf_counter() - t0, 2),
        },
        "legit_deep_evaluated_browser_form": evaluated,
        "counterfactual_https_written_benign_train": counterfactual,
        "within_subtype_scheme_profiles": profiles,
        "note": (
            "The deep suites are scored after browser_form(), which prepends https:// "
            "to bare rows. The raw Kaggle suite contains only 509 https rows; the "
            "evaluated suite contains 70,723, because 70,214 bare rows are rewritten."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"seconds={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
