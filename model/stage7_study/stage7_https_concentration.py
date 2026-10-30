"""
Stage 7 - effective sample size of the malicious-HTTPS gate. READ-ONLY.

The gate 'malicious HTTPS recall >= 0.90' is computed over 2,321 ROWS. Rows are
not independent: they cluster in a small number of hostnames. This script
measures how much of the gate is really decided by how few hosts.

No fitting, no network, no pandas, no artifact writes outside model/stage7_study/.
Inference only on already-fitted artefacts, and only where already done.

Outputs: model/stage7_study/stage7_https_concentration.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "stage7"))

import stage7_audit as a  # noqa: E402
import train  # noqa: E402
import experiment_six as s1  # noqa: E402

OUT = ROOT / "model" / "stage7_study" / "stage7_https_concentration.json"


def main() -> int:
    rows, _ = a.load_corpus()
    urls = [r["url"] for r in rows]
    subtypes = [r["subtype"] for r in rows]
    labels = [r["label"] for r in rows]

    train_idx, test_idx, groups = train.split_by_hostname(urls, labels)
    keep = [not s1.brand_hit(groups[i]) for i in train_idx]
    test_idx = [i for i, k in zip(train_idx, keep) if k] and list(test_idx)
    test_idx = list(test_idx)

    https_mal = [i for i in test_idx
                 if labels[i] == 1 and s1.scheme_of(urls[i]) == "https"]

    per_sub = defaultdict(list)
    hosts = defaultdict(list)
    domains = defaultdict(list)
    for i in https_mal:
        parsed = a.parse_url(urls[i])
        if parsed is None:
            continue
        host = parsed[0]
        per_sub[subtypes[i]].append(host)
        hosts[host].append(i)
        domains[a.approx_domain(host)].append(i)

    def summarise(name, host_list, index_list):
        hc = Counter(host_list)
        dc = Counter(a.approx_domain(h) for h in host_list)
        total = len(index_list)
        top_hosts = [[k, v, round(v / total, 6)] for k, v in hc.most_common(10)]
        ordered = sorted(hc.values(), reverse=True)
        cum = 0
        hosts_for_80 = 0
        for v in ordered:
            cum += v
            hosts_for_80 += 1
            if cum >= 0.80 * total:
                break
        return {
            "name": name,
            "rows": total,
            "distinct_hosts": len(hc),
            "distinct_approx_domains": len(dc),
            "rows_per_host_mean": round(total / len(hc), 2) if hc else None,
            "rows_per_domain_mean": round(total / len(dc), 2) if dc else None,
            "top_hosts": top_hosts,
            "hosts_needed_for_80pct_of_rows": hosts_for_80,
            "share_of_rows_from_single_largest_host": round(
                ordered[0] / total, 6) if ordered else None,
        }

    out = {
        "meta": {
            "stage": 7,
            "mode": "READ-ONLY",
            "purpose": ("Effective sample size of the malicious-HTTPS gate. Rows "
                        "are clustered in hosts; a gate over rows overstates the "
                        "independent evidence behind it."),
            "gate_rows": len(https_mal),
        },
        "all_malicious_https": summarise(
            "all malicious https (test split)", [h for v in per_sub.values() for h in v],
            https_mal),
        "by_subtype": {
            st: summarise(st, per_sub[st],
                          [i for i in https_mal if subtypes[i] == st])
            for st in ("phishing", "malware", "defacement")
        },
    }

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")
    for k in ("all_malicious_https", "phishing", "malware"):
        v = out["all_malicious_https"] if k == "all_malicious_https" else out["by_subtype"][k]
        print(f"{k:22s} rows={v['rows']:5d} hosts={v['distinct_hosts']:4d} "
              f"rows/host={v['rows_per_host_mean']} "
              f"hosts_for_80pct={v['hosts_needed_for_80pct_of_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())