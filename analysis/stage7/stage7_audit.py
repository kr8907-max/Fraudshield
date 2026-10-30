"""
FraudShield Stage 7 - READ-ONLY data-redesign audit.

Constraints enforced by this script:
  * No classifier or vectoriser is created or fitted.
  * No .fit() call of any kind.
  * No network, DNS, WHOIS, TLS, browser or external API access.
  * No pandas.
  * Existing artifacts are opened read-only; nothing is written outside
    analysis/stage7/.
  * Inference on already-fitted artifacts (R2 / S1 / S2) is permitted and is
    read-only; no threshold is changed.

Outputs:
  analysis/stage7/stage7_stats.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.feature_extractor import (  # noqa: E402
    SUSPICIOUS_KEYWORDS,
    extract_feature_vector,
    extract_features,
)
import train  # noqa: E402
import experiment_six as s1  # noqa: E402
import experiment_six_s2 as s2  # noqa: E402

DATASET = ROOT / "data" / "malicious_phish.csv"
TRANCO_SELECTION = ROOT / "data" / "tranco" / "selection.json"
R2_MODEL = ROOT / "model" / "fraudshield_model.pkl"
S1_BUNDLE = ROOT / "model" / "experiment_six" / "s1.pkl"
S2_BUNDLE = ROOT / "model" / "experiment_six" / "s2" / "s2.pkl"
OUT = ROOT / "analysis" / "stage7" / "stage7_stats.json"

S1_DIR = ROOT / "model" / "experiment_six"
S2_DIR = S1_DIR / "s2"

SUBTYPES = ("benign", "phishing", "malware", "defacement")
MALICIOUS = ("phishing", "malware", "defacement")
SCHEMES = ("bare", "http", "https")
STRUCTS = ("homepage", "path", "query", "path+query")
LABEL_MAP = {"benign": 0, "phishing": 1, "malware": 1, "defacement": 1}

CC2_SECOND_LEVEL = {
    "co", "com", "org", "net", "ac", "gov", "edu", "ne", "or", "go", "gob",
    "mil", "sch", "nic", "gen", "web", "nom", "firm", "store", "arts", "rec",
    "info", "biz", "name", "pro", "int", "ltd", "plc", "me", "tv", "cc", "ws",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


def pctl(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def dist(values):
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p10": round(pctl(ordered, 0.10), 4),
        "p25": round(pctl(ordered, 0.25), 4),
        "median": round(pctl(ordered, 0.50), 4),
        "mean": round(float(np.mean(ordered)), 4),
        "p75": round(pctl(ordered, 0.75), 4),
        "p90": round(pctl(ordered, 0.90), 4),
        "p95": round(pctl(ordered, 0.95), 4),
        "p99": round(pctl(ordered, 0.99), 4),
        "max": round(ordered[-1], 4),
    }


def counter_payload(counter: Counter, top: int = 15, total: int | None = None):
    total = total if total is not None else sum(counter.values())
    rows = [
        {"value": str(k), "count": int(v),
         "share": round(v / total, 6) if total else 0.0}
        for k, v in counter.most_common(top)
    ]
    return {"distinct": len(counter), "total": int(total), "top": rows}


def share(part, whole, nd=6):
    return round(part / whole, nd) if whole else None


def approx_suffix(host: str) -> str:
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in CC2_SECOND_LEVEL:
        return ".".join(labels[-2:])
    return labels[-1] if labels else ""


def approx_domain(host: str) -> str:
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in CC2_SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def norm_host(host: str) -> str:
    h = host.strip().lower().rstrip(".")
    if h.startswith("www.") and len(h) > 4:
        return h[4:]
    return h


def parse_url(url: str):
    fields = s1.split_fields(url)
    if fields is None:
        return None
    host, path, query = fields
    return host, path, query


def path_depth(path: str) -> int:
    return sum(1 for part in path.split("/") if part)


def query_params(query: str) -> int:
    if not query:
        return 0
    return query.count("&") + 1


def struct_of(path: str, query: str) -> str:
    has_path = path not in ("", "/")
    has_query = query != ""
    if has_path and has_query:
        return "path+query"
    if has_path:
        return "path"
    if has_query:
        return "query"
    return "homepage"


# ---------------------------------------------------------------------------
# Load the corpus exactly as the project loader does, but keep the subtype
# ---------------------------------------------------------------------------


def load_corpus():
    """Return (rows, load_report).

    rows = list of dict(url, subtype, label) in first-occurrence order after
    exact-URL dedup and cross-label conflict removal.
    """
    started = time.perf_counter()
    raw_counts = Counter()
    invalid_counts = Counter()
    grouped: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    total_lines = 0

    with DATASET.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [h.strip().lower() for h in next(reader)]
        uidx = header.index("url")
        tidx = header.index("type")
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            total_lines += 1
            url = (row[uidx] if uidx < len(row) else "").strip()
            subtype = (row[tidx] if tidx < len(row) else "").strip().lower()
            raw_counts[subtype] += 1
            feats = extract_features(url)
            if feats["url_length"] == 0 or feats["hostname_length"] == 0:
                invalid_counts[subtype] += 1
                continue
            if subtype not in LABEL_MAP:
                invalid_counts[subtype] += 1
                continue
            if url not in grouped:
                grouped[url] = []
                order.append(url)
            grouped[url].append((LABEL_MAP[subtype], subtype))

    conflicting = {
        url for url, entries in grouped.items()
        if len({label for label, _ in entries}) > 1
    }

    rows = []
    duplicate_counts = Counter()
    conflict_counts = Counter()
    for url in order:
        entries = grouped[url]
        subtype = entries[0][1]
        if url in conflicting:
            conflict_counts[subtype] += len(entries)
            continue
        if len(entries) > 1:
            duplicate_counts[subtype] += len(entries) - 1
        rows.append({"url": url, "subtype": subtype,
                     "label": LABEL_MAP[subtype]})

    report = {
        "source_file": str(DATASET.relative_to(ROOT)),
        "data_rows_read": total_lines,
        "raw_type_counts": dict(raw_counts),
        "invalid_or_unknown_counts": dict(invalid_counts),
        "exact_duplicate_rows_removed_by_subtype": dict(duplicate_counts),
        "conflicting_url_rows_removed_by_subtype": dict(conflict_counts),
        "conflicting_urls": len(conflicting),
        "usable_rows": len(rows),
        "usable_by_subtype": dict(Counter(r["subtype"] for r in rows)),
        "load_seconds": round(time.perf_counter() - started, 2),
    }
    return rows, report


# ---------------------------------------------------------------------------
# Theme A - corpus audit by subtype
# ---------------------------------------------------------------------------


def subtype_audit(urls, subtypes):
    n = len(urls)
    scheme_counts = Counter()
    struct_counts = Counter()
    scheme_struct = defaultdict(Counter)
    www = Counter()
    labels = Counter()
    tlds = Counter()
    suffixes = Counter()
    domains = Counter()
    host_lengths = []
    depths = []
    param_counts = []
    query_rows = 0
    host_tokens = []
    path_tokens = []
    query_tokens = []
    distinct_hosts = set()
    distinct_domains = set()
    distinct_host_path_query = set()
    suspicious_any = Counter()
    suspicious_field = {"host": 0, "path": 0, "query": 0}
    nonempty = {"host": 0, "path": 0, "query": 0}
    ip_hosts = 0
    leading_www = 0
    long_host = 0
    hosts_with_many_labels = Counter()

    for url, subtype in zip(urls, subtypes):
        parsed = parse_url(url)
        if parsed is None:
            continue
        host, path, query = parsed
        scheme = s1.scheme_of(url)
        struct = struct_of(path, query)
        scheme_counts[scheme] += 1
        struct_counts[struct] += 1
        scheme_struct[scheme][struct] += 1
        distinct_hosts.add(host)
        distinct_domains.add(approx_domain(host))
        distinct_host_path_query.add((host, path, query))
        if host.startswith("www."):
            www["www"] += 1
            leading_www += 1
        else:
            www["non_www"] += 1
        nlabels = host.count(".") + 1
        labels[nlabels] += 1
        hosts_with_many_labels["3+"] += int(nlabels >= 3)
        hosts_with_many_labels["4+"] += int(nlabels >= 4)
        hosts_with_many_labels["5+"] += int(nlabels >= 5)
        tlds[host.rsplit(".", 1)[-1] if "." in host else host] += 1
        suffixes[approx_suffix(host)] += 1
        domains[approx_domain(host)] += 1
        host_lengths.append(len(host))
        long_host += int(len(host) > 30)
        depths.append(path_depth(path))
        qp = query_params(query)
        if query:
            query_rows += 1
            param_counts.append(qp)
        nonempty["host"] += int(bool(host))
        nonempty["path"] += int(bool(path) and path != "/")
        nonempty["query"] += int(bool(query))
        host_tokens.append(len(TOKEN_RE.findall(host.lower())))
        path_tokens.append(len(TOKEN_RE.findall(path.lower())))
        query_tokens.append(len(TOKEN_RE.findall(query.lower())))
        ip_hosts += int(extract_features(url)["has_ip_host"] == 1)
        lowered = {"host": host.lower(), "path": path.lower(), "query": query.lower()}
        hits = 0
        for field, text in lowered.items():
            field_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text)
            suspicious_field[field] += int(field_hits > 0)
            hits += field_hits
        if hits:
            suspicious_any["any"] += 1
            suspicious_any["total_hits"] += hits

    return {
        "rows": n,
        "scheme": {k: {"count": scheme_counts.get(k, 0),
                       "share": share(scheme_counts.get(k, 0), n)}
                   for k in SCHEMES},
        "structure": {k: {"count": struct_counts.get(k, 0),
                          "share": share(struct_counts.get(k, 0), n)}
                      for k in STRUCTS},
        "scheme_x_structure": {
            scheme: {struct: int(scheme_struct[scheme].get(struct, 0))
                     for struct in STRUCTS}
            for scheme in SCHEMES
        },
        "leading_www": {
            "count": leading_www, "share": share(leading_www, n),
            "non_www": n - leading_www,
        },
        "hostname_label_counts": {
            "distribution": {str(k): int(v) for k, v in sorted(labels.items())},
            "share_3_plus_labels": share(hosts_with_many_labels["3+"], n),
            "share_4_plus_labels": share(hosts_with_many_labels["4+"], n),
            "share_5_plus_labels": share(hosts_with_many_labels["5+"], n),
        },
        "tld_last_label": counter_payload(tlds, top=20, total=n),
        "suffix_heuristic_last2": counter_payload(suffixes, top=20, total=n),
        "approx_registrable_domains": {
            "distinct": len(domains), "rows": n,
            "rows_per_domain_mean": round(n / len(domains), 3) if domains else None,
            "top": counter_payload(domains, top=10, total=n)["top"],
        },
        "distinct_hostnames": len(distinct_hosts),
        "rows_per_hostname_mean": round(n / len(distinct_hosts), 3) if distinct_hosts else None,
        "distinct_normalized_host_path_query": len(distinct_host_path_query),
        "hostname_length": dist(host_lengths),
        "share_hostname_over_30": share(long_host, n),
        "path_depth": {
            "distribution": {str(k): int(v) for k, v in
                             sorted(Counter(depths).items())},
            "stats": dist(depths),
        },
        "query": {
            "rows_with_query": query_rows,
            "share_with_query": share(query_rows, n),
            "param_count_stats": dist(param_counts),
            "param_count_distribution": {str(k): int(v) for k, v in
                                         sorted(Counter(param_counts).items())},
        },
        "field_nonempty": {k: {"count": v, "share": share(v, n)}
                           for k, v in nonempty.items()},
        "token_counts": {
            "host": dist(host_tokens),
            "path": dist(path_tokens),
            "query": dist(query_tokens),
        },
        "ip_literal_hosts": {"count": ip_hosts, "share": share(ip_hosts, n)},
        "suspicious_keyword_fixed_vocab": {
            "definition": "fixed 15-word vocabulary; no fitting",
            "rows_with_any_hit": suspicious_any.get("any", 0),
            "share_with_any_hit": share(suspicious_any.get("any", 0), n),
            "mean_distinct_hits_per_hit_row": (
                round(suspicious_any.get("total_hits", 0) /
                      suspicious_any["any"], 4)
                if suspicious_any.get("any") else None
            ),
            "rows_with_hit_by_field": {
                field: {"count": v, "share": share(v, n)}
                for field, v in suspicious_field.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Cross-form / duplicate relationships
# ---------------------------------------------------------------------------


def cross_form_audit(urls, subtypes):
    groups = defaultdict(set)
    group_subtypes = defaultdict(set)
    raw_host_by_norm = defaultdict(set)
    host_forms = defaultdict(set)
    host_set = set()

    for url, subtype in zip(urls, subtypes):
        parsed = parse_url(url)
        if parsed is None:
            continue
        host, path, query = parsed
        scheme = s1.scheme_of(url)
        key = (norm_host(host), path, query)
        groups[key].add(scheme)
        group_subtypes[key].add(subtype)
        raw_host_by_norm[norm_host(host)].add(host)
        host_forms[host].add(scheme)
        host_set.add(host)

    scheme_sets = Counter(tuple(sorted(v)) for v in groups.values())
    multi_scheme = {k: v for k, v in scheme_sets.items() if len(k) > 1}
    multi_scheme_groups = sum(multi_scheme.values())
    total_groups = len(groups)

    # Apex / www presence.
    apex_present = {h[4:] for h in host_set if h.startswith("www.") and h[4:] in host_set}
    www_present = {h for h in host_set if h.startswith("www.") and h[4:] in host_set}
    bare_www_only = {h for h in host_set if h.startswith("www.") and h[4:] not in host_set}

    # www-normalisation collisions.
    collisions = {k: sorted(v) for k, v in raw_host_by_norm.items() if len(v) > 1}
    collision_rows = sum(len(v) for v in collisions.values())

    # Scheme multiplicity of the same hostname.
    host_multi_scheme = sum(1 for v in host_forms.values() if len(v) > 1)

    # Same host, different path/query variants.
    paths_per_host = Counter()
    for _key in groups:
        paths_per_host[_key[0]] += 1

    examples = []
    for key, schemes in list(groups.items()):
        if len(schemes) > 1:
            examples.append({"normalized_host": key[0], "path": key[1],
                             "query": key[2][:80], "schemes": sorted(schemes)})
            if len(examples) >= 8:
                break

    return {
        "equivalent_url_groups": total_groups,
        "groups_with_multiple_schemes": multi_scheme_groups,
        "share_groups_multiple_schemes": share(multi_scheme_groups, total_groups),
        "scheme_set_distribution": {
            "+".join(k) if k else "none": int(v) for k, v in scheme_sets.most_common(10)
        },
        "multi_scheme_examples": examples,
        "hosts_with_multiple_schemes": host_multi_scheme,
        "share_hosts_with_multiple_schemes": share(host_multi_scheme, len(host_set)),
        "apex_present_for_www_hosts": len(apex_present),
        "www_present_for_apex_hosts": len(www_present),
        "www_hosts_without_apex_row": len(bare_www_only),
        "www_normalization_collisions": len(collisions),
        "rows_in_collision_groups": collision_rows,
        "collision_examples": [{"normalized": k, "raw_hosts": v}
                               for k, v in list(collisions.items())[:8]],
        "host_variant_stats": {
            "hosts": len(host_set),
            "mean_host_path_query_variants": (
                round(total_groups / len(paths_per_host), 3) if paths_per_host else None
            ),
            "hosts_with_over_10_variants": sum(1 for v in paths_per_host.values() if v > 10),
            "max_variants_for_one_host": max(paths_per_host.values()) if paths_per_host else 0,
        },
    }


# ---------------------------------------------------------------------------
# Lexical coverage against the already-fitted S1 vocabulary (no fitting)
# ---------------------------------------------------------------------------


def vocab_coverage(urls, subtypes, host_vec, path_vec, query_vec, seed=42,
                   cap_per_subtype=6000):
    host_analyzer = host_vec.build_analyzer()
    host_vocab = host_vec.vocabulary_
    path_vocab = path_vec.vocabulary_
    query_vocab = query_vec.vocabulary_

    by_subtype = defaultdict(list)
    for i, subtype in enumerate(subtypes):
        by_subtype[subtype].append(i)

    rng = np.random.default_rng(seed)
    out = {}
    for subtype in SUBTYPES:
        positions = by_subtype.get(subtype, [])
        if len(positions) > cap_per_subtype:
            positions = list(rng.choice(positions, cap_per_subtype, replace=False))
        host_shares, path_shares, query_shares = [], [], []
        zero_host = 0
        for i in positions:
            parsed = parse_url(urls[i])
            if parsed is None:
                continue
            host, path, query = parsed
            grams = host_analyzer(host)
            if grams:
                known = sum(1 for g in grams if g in host_vocab)
                host_shares.append(known / len(grams))
                zero_host += int(known == 0)
            ptokens = TOKEN_RE.findall(path.lower())
            if ptokens:
                known = sum(1 for t in ptokens if t in path_vocab)
                path_shares.append(known / len(ptokens))
            qtokens = TOKEN_RE.findall(query.lower())
            if qtokens:
                known = sum(1 for t in qtokens if t in query_vocab)
                query_shares.append(known / len(qtokens))
        out[subtype] = {
            "sampled_rows": len(positions),
            "host_ngram_coverage_mean": round(float(np.mean(host_shares)), 6)
            if host_shares else None,
            "host_rows_with_zero_known_grams": zero_host,
            "path_token_coverage_mean": round(float(np.mean(path_shares)), 6)
            if path_shares else None,
            "path_rows_measured": len(path_shares),
            "query_token_coverage_mean": round(float(np.mean(query_shares)), 6)
            if query_shares else None,
            "query_rows_measured": len(query_shares),
        }
    return out


# ---------------------------------------------------------------------------
# Theme C - benign representation coverage
# ---------------------------------------------------------------------------


def benign_coverage(indices, urls, subtypes, label_of, name, tranco_train_urls):
    rows = []
    cats = {
        "https_homepage": Counter(),
        "https_deep_path": Counter(),
        "https_deep_query": Counter(),
        "https_deep_path_query": Counter(),
        "https_any_deep": Counter(),
        "https_www": Counter(),
        "https_non_www": Counter(),
        "http_any": Counter(),
        "bare_any": Counter(),
        "deep_any": Counter(),
        "homepage_any": Counter(),
    }
    host_set = set()
    domain_set = set()
    suffix_counter = Counter()
    labels_counter = Counter()
    depths = []
    params = []
    www_counter = Counter()
    for i in indices:
        url = urls[i]
        if label_of[i] != 0:
            continue
        parsed = parse_url(url)
        if parsed is None:
            continue
        host, path, query = parsed
        scheme = s1.scheme_of(url)
        struct = struct_of(path, query)
        is_deep = struct != "homepage"
        is_www = host.startswith("www.")
        host_set.add(host)
        domain_set.add(approx_domain(host))
        suffix_counter[approx_suffix(host)] += 1
        labels_counter[host.count(".") + 1] += 1
        depths.append(path_depth(path))
        params.append(query_params(query))
        www_counter["www" if is_www else "non_www"] += 1
        rows.append((scheme, struct, is_www, is_deep))

    n = len(rows)
    def add(cat, cond):
        if cond:
            cats[cat]["yes"] += 1
        else:
            cats[cat]["no"] += 1

    for scheme, struct, is_www, is_deep in rows:
        add("https_homepage", scheme == "https" and struct == "homepage")
        add("https_deep_path", scheme == "https" and struct == "path")
        add("https_deep_query", scheme == "https" and struct == "query")
        add("https_deep_path_query", scheme == "https" and struct == "path+query")
        add("https_any_deep", scheme == "https" and is_deep)
        add("https_www", scheme == "https" and is_www)
        add("https_non_www", scheme == "https" and not is_www)
        add("http_any", scheme == "http")
        add("bare_any", scheme == "bare")
        add("deep_any", is_deep)
        add("homepage_any", not is_deep)

    return {
        "name": name,
        "benign_rows": n,
        "categories": {
            k: {"count": v.get("yes", 0), "share": share(v.get("yes", 0), n)}
            for k, v in cats.items()
        },
        "distinct_hostnames": len(host_set),
        "distinct_approx_domains": len(domain_set),
        "suffix_heuristic": counter_payload(suffix_counter, top=20, total=n),
        "hostname_label_counts": {
            str(k): int(v) for k, v in sorted(labels_counter.items())
        },
        "share_3_plus_labels": share(
            sum(v for k, v in labels_counter.items() if k >= 3), n),
        "share_4_plus_labels": share(
            sum(v for k, v in labels_counter.items() if k >= 4), n),
        "leading_www": {"www": www_counter.get("www", 0),
                        "non_www": www_counter.get("non_www", 0),
                        "www_share": share(www_counter.get("www", 0), n)},
        "path_depth_stats": dist(depths),
        "query_param_stats": dist(params),
        "rows_with_query": sum(1 for _s, struct, _w, _d in rows
                               if struct in ("query", "path+query")),
        "share_with_query": share(sum(1 for _s, struct, _w, _d in rows
                                      if struct in ("query", "path+query")), n),
    }


# ---------------------------------------------------------------------------
# Model inference (read-only) for Theme B
# ---------------------------------------------------------------------------


def scores_r2(urls):
    import joblib
    started = time.perf_counter()
    model = joblib.load(R2_MODEL)
    X = np.asarray([extract_feature_vector(u) for u in urls], dtype=np.int16)
    score = model.predict_proba(X)[:, 1]
    return score, round(time.perf_counter() - started, 2)


def scores_s1(urls):
    import joblib
    started = time.perf_counter()
    bundle = joblib.load(S1_BUNDLE)
    model = s1.SemanticURLModel(bundle["host_vec"], bundle["path_vec"],
                                bundle["query_vec"], bundle["classifier"],
                                bundle["config"])
    score = model.predict_proba(urls)[:, 1]
    return score, round(time.perf_counter() - started, 2)


def scores_s2(urls):
    import joblib
    started = time.perf_counter()
    bundle = joblib.load(S2_BUNDLE)
    model = s2.SemanticURLModelS2(bundle["host_vec"], bundle["path_vec"],
                                  bundle["query_vec"], bundle["classifier"],
                                  bundle["config"])
    score = model.predict_proba(urls)[:, 1]
    return score, round(time.perf_counter() - started, 2)


def scheme_subtype_table(y, subtypes, schemes, pred, score):
    out = {}
    for subtype in MALICIOUS:
        out[subtype] = {}
        for scheme in SCHEMES:
            mask = (np.asarray(subtypes) == subtype) & (np.asarray(schemes) == scheme)
            n = int(mask.sum())
            if not n:
                out[subtype][scheme] = {"count": 0, "detected": 0,
                                        "recall": None, "mean_score": None}
                continue
            out[subtype][scheme] = {
                "count": n,
                "detected": int(pred[mask].sum()),
                "recall": round(float(pred[mask].mean()), 6),
                "mean_score": round(float(score[mask].mean()), 6),
            }
    return out


def https_gate_attribution(y, subtypes, schemes, pred):
    y = np.asarray(y)
    subtypes = np.asarray(subtypes)
    schemes = np.asarray(schemes)
    https = (schemes == "https") & (y == 1)
    total = int(https.sum())
    detected = int(pred[https].sum())
    missed = total - detected
    per = {}
    for subtype in MALICIOUS:
        mask = https & (subtypes == subtype)
        n = int(mask.sum())
        d = int(pred[mask].sum()) if n else 0
        per[subtype] = {
            "https_rows": n,
            "share_of_malicious_https": share(n, total),
            "detected": d,
            "missed": n - d,
            "recall": round(d / n, 6) if n else None,
            "share_of_https_detections": share(d, detected),
            "share_of_https_misses": share(n - d, missed),
        }
    gate_needed = int(np.ceil(0.90 * total))
    shortfall = max(gate_needed - detected, 0)
    # Leave-one-subtype-out counterfactual recall.
    loo = {}
    for subtype in MALICIOUS:
        n_wo = total - per[subtype]["https_rows"]
        d_wo = detected - per[subtype]["detected"]
        loo[subtype] = {
            "https_rows_without_subtype": n_wo,
            "detected_without_subtype": d_wo,
            "recall_without_subtype": round(d_wo / n_wo, 6) if n_wo else None,
        }
    return {
        "malicious_https_rows": total,
        "malicious_https_detected": detected,
        "malicious_https_missed": missed,
        "malicious_https_recall": round(detected / total, 6) if total else None,
        "rows_needed_for_0_90": gate_needed,
        "shortfall_rows_to_0_90": shortfall,
        "per_subtype": per,
        "leave_one_subtype_out": loo,
        "defacement_https_rows": per["defacement"]["https_rows"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.perf_counter()
    print("[1/7] Loading corpus (validity + exact-URL dedup, same rules as loader)")
    rows, load_report = load_corpus()
    print("     ", load_report["usable_rows"], load_report["usable_by_subtype"])

    urls = [r["url"] for r in rows]
    subtypes = [r["subtype"] for r in rows]
    labels = [r["label"] for r in rows]

    print("[2/7] Reproducing hostname-grouped split and brand purge")
    train_idx, test_idx, groups = train.split_by_hostname(urls, labels)
    keep = np.asarray([not s1.brand_hit(groups[i]) for i in train_idx], dtype=bool)
    train_purged = train_idx[keep]
    test_idx = np.asarray(test_idx)
    split_report = {
        "train_before_purge": int(len(train_idx)),
        "train_after_brand_purge": int(len(train_purged)),
        "purged_as_sanity_brand": int((~keep).sum()),
        "test_rows": int(len(test_idx)),
        "test_subtypes": dict(Counter(subtypes[i] for i in test_idx)),
        "train_subtypes_after_purge": dict(Counter(subtypes[i] for i in train_purged)),
        "matches_r2_metadata": {
            "train_before_purge": int(len(train_idx)) == 512870,
            "train_after_purge": int(len(train_purged)) == 503901,
            "test_rows": int(len(test_idx)) == 128214,
            "purged": int((~keep).sum()) == 8969,
        },
    }
    print("     ", split_report["matches_r2_metadata"])

    print("[3/7] Theme A - subtype audit")
    audit = {}
    for subtype in SUBTYPES:
        positions = [i for i, st in enumerate(subtypes) if st == subtype]
        audit[subtype] = subtype_audit([urls[i] for i in positions],
                                       [subtypes[i] for i in positions])
        print(f"      {subtype}: {audit[subtype]['rows']:,} rows")

    test_positions = list(test_idx)
    test_audit = {}
    for subtype in SUBTYPES:
        positions = [i for i in test_positions if subtypes[i] == subtype]
        test_audit[subtype] = subtype_audit([urls[i] for i in positions],
                                            [subtypes[i] for i in positions])
    print("[4/7] Cross-form / duplicate audit")
    cross_form = {}
    for subtype in SUBTYPES:
        positions = [i for i, st in enumerate(subtypes) if st == subtype]
        cross_form[subtype] = cross_form_audit(
            [urls[i] for i in positions], [subtypes[i] for i in positions]
        )
    cross_form["all"] = cross_form_audit(urls, subtypes)

    print("[5/7] Lexical coverage against fitted S1 vocabulary (no fitting)")
    import joblib
    s1_bundle = joblib.load(S1_BUNDLE)
    coverage = vocab_coverage(urls, subtypes, s1_bundle["host_vec"],
                              s1_bundle["path_vec"], s1_bundle["query_vec"])
    coverage_test = vocab_coverage([urls[i] for i in test_idx],
                                   [subtypes[i] for i in test_idx],
                                   s1_bundle["host_vec"], s1_bundle["path_vec"],
                                   s1_bundle["query_vec"])

    print("[6/7] Theme B - read-only inference on the fixed test split")
    test_urls = [urls[i] for i in test_idx]
    test_subtypes = [subtypes[i] for i in test_idx]
    test_y = np.asarray([labels[i] for i in test_idx], dtype=np.int8)
    test_schemes = [s1.scheme_of(u) for u in test_urls]

    r2_score, r2_seconds = scores_r2(test_urls)
    s1_score, s1_seconds = scores_s1(test_urls)
    s2_score, s2_seconds = scores_s2(test_urls)
    r2_pred = (r2_score >= 0.5).astype(np.int8)
    s1_pred = (s1_score >= 0.5).astype(np.int8)
    s2_pred = (s2_score >= 0.5).astype(np.int8)

    models = {}
    for name, pred, score, seconds in (
        ("R2", r2_pred, r2_score, r2_seconds),
        ("S1", s1_pred, s1_score, s1_seconds),
        ("S2-A", s2_pred, s2_score, s2_seconds),
    ):
        models[name] = {
            "inference_seconds": seconds,
            "https_gate": https_gate_attribution(test_y, test_subtypes,
                                                 test_schemes, pred),
            "by_subtype_and_scheme": scheme_subtype_table(
                test_y, test_subtypes, test_schemes, pred, score),
            "overall_malicious_recall": round(float(pred[test_y == 1].mean()), 6),
            "https_malicious_recall_check": round(
                float(pred[(test_y == 1) & (np.asarray(test_schemes) == "https")].mean()), 6),
        }
        print(f"      {name}: https recall={models[name]['https_gate']['malicious_https_recall']} "
              f"defacement https rows={models[name]['https_gate']['defacement_https_rows']}")

    print("[7/7] Theme C - benign representation coverage")
    # Tranco generated training and holdout rows (deterministic reconstruction).
    selection = json.loads(TRANCO_SELECTION.read_text(encoding="utf-8"))
    tranco_train_urls = []
    for domain in selection["train_domains"]:
        tranco_train_urls.append(f"https://{domain}")
    for domain in selection["www_train_domains"]:
        tranco_train_urls.append(f"https://www.{domain}")
    tranco_holdout_urls = []
    for domain in selection["holdout_domains"]:
        tranco_holdout_urls += [f"https://{domain}", f"https://www.{domain}"]

    # Build index arrays that include synthetic Tranco rows by appending them.
    kaggle_train_positions = list(train_purged)
    legit_deep_positions = [i for i in test_idx
                            if labels[i] == 0 and s1.is_deep(urls[i])]

    all_urls = urls + tranco_train_urls + tranco_holdout_urls
    all_subtypes = subtypes + ["benign"] * (len(tranco_train_urls) + len(tranco_holdout_urls))
    all_label = labels + [0] * (len(tranco_train_urls) + len(tranco_holdout_urls))
    kaggle_train_idx = kaggle_train_positions
    tranco_train_idx = list(range(len(urls), len(urls) + len(tranco_train_urls)))
    tranco_holdout_idx = list(range(len(urls) + len(tranco_train_urls), len(all_urls)))

    coverage_blocks = {
        "kaggle_benign_train": benign_coverage(
            kaggle_train_idx, all_urls, all_subtypes, all_label,
            "Kaggle benign train rows after brand purge", tranco_train_urls),
        "tranco_generated_train": benign_coverage(
            tranco_train_idx, all_urls, all_subtypes, all_label,
            "Tranco generated training representations (https:// apex + https://www.)",
            tranco_train_urls),
        "tranco_holdout": benign_coverage(
            tranco_holdout_idx, all_urls, all_subtypes, all_label,
            "Tranco holdout generated representations (https:// apex + https://www.)",
            tranco_train_urls),
        "legit_deep_suite": benign_coverage(
            legit_deep_positions, all_urls, all_subtypes, all_label,
            "Legitimate deep suite (Kaggle TEST split, benign, path/query)", []),
        "r2_s1_training_total": benign_coverage(
            kaggle_train_idx + tranco_train_idx, all_urls, all_subtypes, all_label,
            "R2/S1 total benign training population (Kaggle purged + Tranco generated)",
            tranco_train_urls),
    }

    # TLD overlap between training benign and suites.
    def suffix_set(block):
        return {row["value"] for row in block["suffix_heuristic"]["top"]}
    coverage_blocks["comparison"] = {
        "note": "Shares are within each block's benign rows. Gaps are the deep-suite "
                "share minus the total-training share.",
        "legit_deep_minus_train": {
            cat: round(
                (coverage_blocks["legit_deep_suite"]["categories"][cat]["share"] or 0)
                - (coverage_blocks["r2_s1_training_total"]["categories"][cat]["share"] or 0), 6)
            for cat in coverage_blocks["legit_deep_suite"]["categories"]
        },
        "tranco_holdout_minus_train": {
            cat: round(
                (coverage_blocks["tranco_holdout"]["categories"][cat]["share"] or 0)
                - (coverage_blocks["r2_s1_training_total"]["categories"][cat]["share"] or 0), 6)
            for cat in coverage_blocks["tranco_holdout"]["categories"]
        },
    }

    # Confirm defacement HTTPS from source across whole corpus.
    deface_https_full = audit["defacement"]["scheme"]["https"]["count"]
    deface_https_test = test_audit["defacement"]["scheme"]["https"]["count"]

    payload = {
        "meta": {
            "stage": 7,
            "study": "data-redesign audit A-H",
            "mode": "READ-ONLY",
            "no_fit": True,
            "no_network": True,
            "no_pandas": True,
            "threshold": 0.5,
            "generated_seconds": round(time.perf_counter() - t0, 2),
            "sources": {
                "corpus": str(DATASET.relative_to(ROOT)),
                "tranco_selection": str(TRANCO_SELECTION.relative_to(ROOT)),
                "r2_model": str(R2_MODEL.relative_to(ROOT)),
                "s1_bundle": str(S1_BUNDLE.relative_to(ROOT)),
                "s2_bundle": str(S2_BUNDLE.relative_to(ROOT)),
            },
        },
        "integrity": {
            "load_report": load_report,
            "split_report": split_report,
            "defacement_https_zero_confirmed": {
                "full_corpus_rows": deface_https_full,
                "test_split_rows": deface_https_test,
                "is_zero": deface_https_full == 0,
            },
        },
        "A_corpus_by_subtype": {"full_corpus": audit, "test_split": test_audit},
        "A_cross_form": cross_form,
        "A_lexical_coverage": {
            "full_corpus": coverage,
            "test_split": coverage_test,
            "definition": "share of a field's tokens/n-grams that exist in the already-fitted "
                          "S1 vocabulary. No vectoriser was fitted or transformed here.",
        },
        "B_malicious_https_attribution": models,
        "C_benign_coverage": coverage_blocks,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"Total seconds: {time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
