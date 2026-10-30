"""
FraudShield Stage 7 - READ-ONLY data-redesign study (sections A-H support).

HARD CONSTRAINTS enforced here:
  * No classifier, vectoriser or model is created, fitted or refitted.
  * No .fit() / .partial_fit() call of any kind.
  * No network, DNS, WHOIS, TLS, browser or external API access.
  * No dataset download.
  * No pandas.
  * Python 3.14 compatible.
  * Existing artifacts are opened READ-ONLY. Nothing is written outside
    model/stage7_study/.
  * Inference on ALREADY-FITTED artifacts (R2 / S1 / S2-A) is read-only and
    permitted; no threshold is changed and no score is used to refit anything.

Reuses the already-verified audit module analysis/stage7/stage7_audit.py
(import only, never modified) so that URL parsing, scheme detection, structure
classification and the corpus loader are byte-identical to the prior session's
A/B/C computation.

Outputs (all under model/stage7_study/):
  protected_before.json
  protected_after.json
  protected_diff.json
  stage7_addendum.json     (subtype x scheme profiles, evaluated deep suite,
                            counterfactual, defacement lexical study)
  stage7_derived.json      (D / G derived numbers)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDY = ROOT / "model" / "stage7_study"
STUDY.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "analysis" / "stage7"))

import stage7_audit as a  # noqa: E402
import train  # noqa: E402
import experiment_six as s1  # noqa: E402
import experiment_six_s2 as s2  # noqa: E402

# ---------------------------------------------------------------------------
# Protected artifact set: the project's own manifest plus S1/S2 artifacts.
# ---------------------------------------------------------------------------

PROTECTED = [
    "model/fraudshield_model.pkl",
    "model/model_metadata.json",
    "backend/feature_extractor.py",
    "backend/predictor.py",
    "backend/dataset_loader.py",
    "train.py",
    "experiment_c.py",
    "experiment_d.py",
    "experiment_six.py",
    "experiment_six_s2.py",
    "data/malicious_phish.csv",
    "data/tranco/top-1m.csv",
    "data/tranco/selection.json",
    "model/experiment_c/c1_r1.pkl",
    "model/experiment_c/c1_r1.metadata.json",
    "model/experiment_c/c2_r1.pkl",
    "model/experiment_c/c2_r1.metadata.json",
    "model/experiment_c/comparison.json",
    "model/experiment_c/winner_r2.pkl",
    "model/experiment_c/winner_r2.metadata.json",
    "model/experiment_d/comparison.json",
    "model/experiment_d/deep_separability.json",
    "model/experiment_d/d1.pkl",
    "model/experiment_d/d1.metadata.json",
    "model/experiment_d/d2.pkl",
    "model/experiment_d/d2.metadata.json",
    "model/experiment_six/s1.pkl",
    "model/experiment_six/s1.metadata.json",
    "model/experiment_six/comparison.json",
    "model/experiment_six/evaluation.json",
    "model/experiment_six/feature_config.json",
    "model/experiment_six/protected_before.json",
    "model/experiment_six/s2/s2.pkl",
    "model/experiment_six/s2/s2.metadata.json",
    "model/experiment_six/s2/comparison.json",
    "model/experiment_six/s2/evaluation.json",
    "model/experiment_six/s2/feature_config.json",
    "model/experiment_six/s2/host_artifact_audit.json",
    "model/experiment_six/s2/preflight.json",
    "model/experiment_six/s2/protected_diff.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_protected() -> dict:
    out = {}
    for rel in PROTECTED:
        p = ROOT / rel
        out[rel] = sha256(p) if p.exists() else None
    return out


def write_json(name: str, payload) -> Path:
    p = STUDY / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    print(f"  wrote {p.name} ({p.stat().st_size:,} bytes)")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def browser_form(url: str) -> str:
    text = url.strip()
    return text if s1.SCHEME_RE.match(text) else "https://" + text


def token_set(text: str) -> set[str]:
    return set(a.TOKEN_RE.findall(text.lower()))


def jaccard(x: set, y: set) -> float:
    if not x and not y:
        return 1.0
    union = x | y
    return len(x & y) / len(union) if union else 0.0


def score_block(scores: np.ndarray, positive: bool = True) -> dict:
    if scores.size == 0:
        return {"n": 0}
    if positive:
        rate = float((scores >= 0.5).mean())
    else:
        rate = float((scores < 0.5).mean())
    return {
        "n": int(scores.size),
        "rate_at_0_5": round(rate, 6),
        "mean": round(float(scores.mean()), 6),
        "p10": round(float(np.quantile(scores, 0.10)), 6),
        "median": round(float(np.median(scores)), 6),
        "p90": round(float(np.quantile(scores, 0.90)), 6),
    }


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print("Stage 7 READ-ONLY addendum + derived analysis")
    print("=" * 74)

    print("\n[0] Hashing protected artifacts (before)")
    before = hash_protected()
    write_json("protected_before.json", before)
    missing_before = [k for k, v in before.items() if v is None]
    print(f"    {len(before)} artifacts, {len(missing_before)} missing")

    # ---------------------------------------------------------------
    print("\n[1] Loading corpus (identical loader to A/B/C)")
    rows, load_report = a.load_corpus()
    urls = [r["url"] for r in rows]
    subtypes = [r["subtype"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"    usable rows: {len(urls):,}")

    print("\n[2] Reproducing the fixed split + brand purge")
    train_idx, test_idx, groups = train.split_by_hostname(urls, labels)
    keep = np.asarray([not s1.brand_hit(groups[i]) for i in train_idx], dtype=bool)
    train_purged = train_idx[keep]
    test_idx = np.asarray(test_idx)
    print(f"    train after purge: {len(train_purged):,}  test: {len(test_idx):,}")

    train_hosts, benign_train_hosts, malicious_train_hosts = set(), set(), set()
    for i in train_purged:
        parsed = a.parse_url(urls[i])
        if parsed is None:
            continue
        host = parsed[0]
        train_hosts.add(host)
        if labels[i] == 0:
            benign_train_hosts.add(host)
        else:
            malicious_train_hosts.add(host)

    # ---------------------------------------------------------------
    print("\n[3] Loading already-fitted models (read-only, no fit)")
    import joblib
    r2 = joblib.load(a.R2_MODEL)
    s1b = joblib.load(a.S1_BUNDLE)
    s2b = joblib.load(a.S2_BUNDLE)
    m1 = s1.SemanticURLModel(s1b["host_vec"], s1b["path_vec"], s1b["query_vec"],
                             s1b["classifier"], s1b["config"])
    m2 = s2.SemanticURLModelS2(s2b["host_vec"], s2b["path_vec"], s2b["query_vec"],
                               s2b["classifier"], s2b["config"])
    host_vocab = s1b["host_vec"].vocabulary_
    host_analyzer = s1b["host_vec"].build_analyzer()
    print("    R2 / S1 / S2-A loaded")

    def score_urls(subset):
        if not subset:
            return None, None, None
        X = np.asarray([a.extract_feature_vector(u) for u in subset], dtype=np.int16)
        return (r2.predict_proba(X)[:, 1],
                m1.predict_proba(subset)[:, 1],
                m2.predict_proba(subset)[:, 1])

    # ---------------------------------------------------------------
    # [4] Within-subtype x scheme profiles on the fixed test split
    # ---------------------------------------------------------------
    print("\n[4] Within-subtype x scheme profiles (test split)")
    profiles = {}
    for subtype in ("phishing", "malware", "defacement"):
        profiles[subtype] = {}
        for scheme in a.SCHEMES:
            positions = [i for i in test_idx
                         if subtypes[i] == subtype and s1.scheme_of(urls[i]) == scheme]
            if not positions:
                profiles[subtype][scheme] = {"rows": 0}
                print(f"    {subtype:11s}/{scheme:5s}: 0")
                continue
            subset = [urls[i] for i in positions]
            r2s, s1s, s2s = score_urls(subset)

            tlds, host_lengths, depths, cov = Counter(), [], [], []
            www = in_train = in_ben = in_mal = susp = 0
            hosts = set()
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
                in_ben += int(host in benign_train_hosts)
                in_mal += int(host in malicious_train_hosts)
                grams = host_analyzer(host)
                if grams:
                    cov.append(sum(1 for g in grams if g in host_vocab) / len(grams))
                low_h, low_p, low_q = host.lower(), path.lower(), query.lower()
                susp += int(any(k in low_h or k in low_p or k in low_q
                                for k in a.SUSPICIOUS_KEYWORDS))

            profiles[subtype][scheme] = {
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
                "share_host_in_benign_train": a.share(in_ben, len(subset)),
                "share_host_in_malicious_train": a.share(in_mal, len(subset)),
                "models": {
                    "R2": score_block(r2s),
                    "S1": score_block(s1s),
                    "S2-A": score_block(s2s),
                },
            }
            print(f"    {subtype:11s}/{scheme:5s}: {len(subset):6,d} rows  "
                  f"R2={profiles[subtype][scheme]['models']['R2']['rate_at_0_5']:.4f} "
                  f"S1={profiles[subtype][scheme]['models']['S1']['rate_at_0_5']:.4f} "
                  f"S2={profiles[subtype][scheme]['models']['S2-A']['rate_at_0_5']:.4f}")

    # ---------------------------------------------------------------
    # [5] Legitimate-deep suite in EVALUATED browser form + counterfactual
    # ---------------------------------------------------------------
    print("\n[5] Legitimate-deep suite: raw vs evaluated browser form")
    deep_positions = [i for i in test_idx if labels[i] == 0 and s1.is_deep(urls[i])]
    deep_raw = [urls[i] for i in deep_positions]
    deep_eval = [browser_form(u) for u in deep_raw]
    rewritten = sum(1 for r, e in zip(deep_raw, deep_eval) if r != e)
    print(f"    deep benign rows: {len(deep_raw):,}  rewritten to https://: {rewritten:,}")

    r2s, s1s, s2s = score_urls(deep_eval)
    deep_eval_block = {
        "rows": len(deep_eval),
        "rows_rewritten_to_https": rewritten,
        "share_rewritten": a.share(rewritten, len(deep_eval)),
        "models": {
            "R2": score_block(r2s, positive=False),
            "S1": score_block(s1s, positive=False),
            "S2-A": score_block(s2s, positive=False),
        },
        "note": ("rate_at_0_5 here is the SAFE rate (score < 0.5), matching how "
                 "the project reports legitimate-deep safe rate."),
    }
    for k, v in deep_eval_block["models"].items():
        print(f"    {k:5s} safe rate (evaluated form): {v['rate_at_0_5']:.6f}")

    # Same rows, raw (unrewritten) form, to isolate the rewrite effect.
    r2r, s1r, s2r = score_urls(deep_raw)
    deep_raw_block = {
        "rows": len(deep_raw),
        "models": {
            "R2": score_block(r2r, positive=False),
            "S1": score_block(s1r, positive=False),
            "S2-A": score_block(s2r, positive=False),
        },
        "note": "Same rows scored WITHOUT the https:// rewrite, i.e. as stored.",
    }
    for k, v in deep_raw_block["models"].items():
        print(f"    {k:5s} safe rate (raw form):       {v['rate_at_0_5']:.6f}")

    # Counterfactual: Kaggle benign deep TRAIN rows written as https.
    kaggle_benign_deep = [i for i in train_purged
                          if labels[i] == 0 and s1.is_deep(urls[i])]
    cf = [browser_form(urls[i]) for i in kaggle_benign_deep]
    cf_https = sum(1 for i in kaggle_benign_deep
                   if s1.scheme_of(urls[i]) == "https")
    counterfactual = {
        "kaggle_benign_deep_train_rows": len(kaggle_benign_deep),
        "already_https": cf_https,
        "would_become_https": len(kaggle_benign_deep) - cf_https,
        "share_already_https": a.share(cf_https, len(kaggle_benign_deep)),
        "note": ("Kaggle benign deep training rows as stored are overwhelmingly "
                 "bare; only this small share is genuinely https."),
    }
    print(f"    Kaggle benign deep TRAIN rows: {len(kaggle_benign_deep):,}  "
          f"already https: {cf_https:,} ({counterfactual['share_already_https']:.4%})")

    # ---------------------------------------------------------------
    # [6] Defacement lexical study - scheme confound held constant
    # ---------------------------------------------------------------
    print("\n[6] Defacement study: scheme confound held constant (http + deep only)")
    rng = np.random.default_rng(42)

    def sample_positions(pred, cap):
        pos = [i for i in test_idx if pred(i)]
        if len(pos) > cap:
            pos = list(rng.choice(pos, cap, replace=False))
        return pos

    def_http = sample_positions(
        lambda i: subtypes[i] == "defacement" and s1.scheme_of(urls[i]) == "http", 6000)
    ben_http = sample_positions(
        lambda i: subtypes[i] == "benign" and s1.scheme_of(urls[i]) == "http"
        and s1.is_deep(urls[i]), 6000)
    print(f"    defacement http+deep: {len(def_http):,}   benign http+deep: {len(ben_http):,}")

    def token_profile(positions):
        toks, host_toks = Counter(), Counter()
        depths, lens, params = [], [], []
        susp = 0
        all_tokens = set()
        for i in positions:
            parsed = a.parse_url(urls[i])
            if parsed is None:
                continue
            host, path, query = parsed
            pt = a.TOKEN_RE.findall((path + " " + query).lower())
            toks.update(pt)
            all_tokens.update(pt)
            host_toks.update(a.TOKEN_RE.findall(host.lower()))
            depths.append(a.path_depth(path))
            lens.append(len(path) + len(query))
            params.append(a.query_params(query))
            low = (host + path + query).lower()
            susp += int(any(k in low for k in a.SUSPICIOUS_KEYWORDS))
        return {
            "rows": len(positions),
            "distinct_path_tokens": len(toks),
            "distinct_host_tokens": len(host_toks),
            "top_path_tokens": [{"value": k, "count": v} for k, v in toks.most_common(20)],
            "top_host_tokens": [{"value": k, "count": v} for k, v in host_toks.most_common(15)],
            "path_depth_median": a.pctl(sorted(depths), 0.50),
            "path_depth_p90": a.pctl(sorted(depths), 0.90),
            "path_query_length_median": a.pctl(sorted(lens), 0.50),
            "query_params_median": a.pctl(sorted(params), 0.50),
            "share_suspicious_keyword": a.share(susp, len(positions)),
            "_tokens": all_tokens,
        }

    dp = token_profile(def_http)
    bp = token_profile(ben_http)
    d_tokens, b_tokens = dp.pop("_tokens"), bp.pop("_tokens")
    overlap = jaccard(d_tokens, b_tokens)
    print(f"    path-token Jaccard (defacement vs benign, http+deep): {overlap:.4f}")

    # Non-fitted rule diagnostics, scheme held constant.
    def rule_eval(positions, rule):
        hit = 0
        for i in positions:
            parsed = a.parse_url(urls[i])
            if parsed is None:
                continue
            host, path, query = parsed
            hit += int(rule(host, path, query))
        return a.share(hit, len(positions))

    rules = {
        "has_suspicious_keyword": lambda h, p, q: any(
            k in (h + p + q).lower() for k in a.SUSPICIOUS_KEYWORDS),
        "has_query": lambda h, p, q: q != "",
        "path_depth_ge_3": lambda h, p, q: a.path_depth(p) >= 3,
        "host_has_digit": lambda h, p, q: any(c.isdigit() for c in h),
        "path_has_extension": lambda h, p, q: any(
            p.lower().endswith(e) for e in (".php", ".html", ".htm", ".asp", ".jsp")),
    }
    rule_table = {
        name: {
            "defacement_http_deep_rate": rule_eval(def_http, fn),
            "benign_http_deep_rate": rule_eval(ben_http, fn),
        }
        for name, fn in rules.items()
    }
    for name, v in rule_table.items():
        gap = (v["defacement_http_deep_rate"] or 0) - (v["benign_http_deep_rate"] or 0)
        print(f"    rule {name:26s} def={v['defacement_http_deep_rate']:.4f} "
              f"ben={v['benign_http_deep_rate']:.4f} gap={gap:+.4f}")

    # Model separability with scheme held constant.
    d_scores = score_urls([urls[i] for i in def_http])
    b_scores = score_urls([urls[i] for i in ben_http])
    sep = {}
    for name, (ds, bs) in zip(("R2", "S1", "S2-A"), zip(d_scores, b_scores)):
        if ds is None or bs is None:
            sep[name] = {"note": "insufficient rows"}
            continue
        # AUC via rank statistic (no fitting).
        allv = np.concatenate([ds, bs])
        order = allv.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, allv.size + 1)
        n1, n0 = ds.size, bs.size
        auc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        sep[name] = {
            "auc_defacement_vs_benign_http_deep": round(float(auc), 6),
            "defacement_mean_score": round(float(ds.mean()), 6),
            "benign_mean_score": round(float(bs.mean()), 6),
            "defacement_recall_at_0_5": round(float((ds >= 0.5).mean()), 6),
            "benign_fpr_at_0_5": round(float((bs >= 0.5).mean()), 6),
        }
        print(f"    {name:5s} AUC(def vs ben | http+deep) = {auc:.6f}  "
              f"def_recall={sep[name]['defacement_recall_at_0_5']:.4f} "
              f"ben_fpr={sep[name]['benign_fpr_at_0_5']:.4f}")

    # Scheme-only confound: how well does "http" alone identify defacement?
    def_share_http = a.share(
        sum(1 for i in range(len(urls)) if subtypes[i] == "defacement"
            and s1.scheme_of(urls[i]) == "http"),
        sum(1 for i in range(len(urls)) if subtypes[i] == "defacement"))
    ben_share_http = a.share(
        sum(1 for i in range(len(urls)) if subtypes[i] == "benign"
            and s1.scheme_of(urls[i]) == "http"),
        sum(1 for i in range(len(urls)) if subtypes[i] == "benign"))
    mal_share_http = a.share(
        sum(1 for i in range(len(urls)) if labels[i] == 1
            and s1.scheme_of(urls[i]) == "http"),
        sum(1 for i in range(len(urls)) if labels[i] == 1))

    defacement_block = {
        "scheme_confound": {
            "defacement_share_http": def_share_http,
            "benign_share_http": ben_share_http,
            "all_malicious_share_http": mal_share_http,
            "note": ("Defacement is 100% http and 0% bare/https/homepage in this "
                     "corpus. Scheme alone therefore carries a large share of any "
                     "apparent defacement predictability."),
        },
        "http_deep_token_profile": {
            "defacement": dp,
            "benign": bp,
            "path_token_jaccard": round(overlap, 6),
        },
        "non_fitted_rules_http_deep": rule_table,
        "model_separability_http_deep": sep,
    }

    # ---------------------------------------------------------------
    payload = {
        "meta": {
            "stage": 7,
            "mode": "READ-ONLY",
            "no_fit": True,
            "no_network": True,
            "no_pandas": True,
            "no_download": True,
            "threshold": 0.5,
            "seconds": round(time.perf_counter() - t0, 2),
            "reuses": "analysis/stage7/stage7_audit.py (import only, unmodified)",
        },
        "split_reproduction": {
            "train_after_purge": int(len(train_purged)),
            "test_rows": int(len(test_idx)),
            "matches_r2_metadata": {
                "train_after_purge": len(train_purged) == 503901,
                "test_rows": len(test_idx) == 128214,
            },
        },
        "within_subtype_scheme_profiles": profiles,
        "legit_deep_evaluated_form": deep_eval_block,
        "legit_deep_raw_form": deep_raw_block,
        "kaggle_benign_deep_counterfactual": counterfactual,
        "defacement_study": defacement_block,
    }
    write_json("stage7_addendum.json", payload)

    # ---------------------------------------------------------------
    print("\n[7] Hashing protected artifacts (after)")
    after = hash_protected()
    write_json("protected_after.json", after)

    changed = [k for k in before if before[k] != after.get(k)]
    missing = [k for k, v in after.items() if v is None]
    diff = {
        "checked": len(before),
        "changed": changed,
        "missing": missing,
        "changed_count": len(changed),
        "missing_count": len(missing),
        "before": before,
        "after": after,
    }
    write_json("protected_diff.json", diff)
    print(f"    changed: {len(changed)}  missing: {len(missing)}")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
