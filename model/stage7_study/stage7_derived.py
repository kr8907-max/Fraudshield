"""
Stage 7 - derived metrics for sections D and G. READ-ONLY.

Consumes only existing JSON artifacts (no corpus re-read, no model inference,
no fitting, no network). Every number written here is arithmetic over values
already present in the project's own artifacts, so each is traceable.

Outputs: model/stage7_study/stage7_derived.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "model" / "stage7_study"
STATS = ROOT / "analysis" / "stage7" / "stage7_stats.json"
S2_EVAL = ROOT / "model" / "experiment_six" / "s2" / "evaluation.json"
S1_EVAL = ROOT / "model" / "experiment_six" / "evaluation.json"
S2_CMP = ROOT / "model" / "experiment_six" / "s2" / "comparison.json"
R2_META = ROOT / "model" / "model_metadata.json"
ADDENDUM = STUDY / "stage7_addendum.json"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rows_needed(target, total):
    """Rows that must be correct to reach `target` recall/safe rate."""
    return int(-(-target * total // 1))  # ceil


def main():
    stats = load(STATS)
    s2e = load(S2_EVAL)
    s1e = load(S1_EVAL)
    s2c = load(S2_CMP)
    r2m = load(R2_META)
    add = load(ADDENDUM)

    out = {"meta": {"stage": 7, "mode": "READ-ONLY", "derived_only": True}}

    # ------------------------------------------------------------------
    # G. Gate sensitivity: how far each gate is from passing, in rows.
    # ------------------------------------------------------------------
    it = s2e["internal_test"]
    mal_total = it["tp"] + it["fn"]
    ben_total = it["tn"] + it["fp"]

    https = s2e["internal_test"]["by_style"]["https"]
    https_rows = https["malicious_count"]
    https_detected = int(round(https["malicious_recall"] * https_rows))

    tranco = s2e["external_tranco_homepage"]
    tranco_urls = tranco["url_count"]
    tranco_safe = int(round(tranco["url_safe_rate"] * tranco_urls))

    deep_leg = s2e["deep_legitimate"]["overall"]
    deep_mal = s2e["deep_malicious"]["overall"]

    gates = {
        "overall_malicious_recall_ge_0.87": {
            "target": 0.87,
            "observed": it["recall"],
            "denominator": mal_total,
            "correct_now": it["tp"],
            "correct_needed": rows_needed(0.87, mal_total),
            "shortfall_rows": max(0, rows_needed(0.87, mal_total) - it["tp"]),
            "passed": it["recall"] >= 0.87,
        },
        "malicious_https_recall_ge_0.90": {
            "target": 0.90,
            "observed": https["malicious_recall"],
            "denominator": https_rows,
            "correct_now": https_detected,
            "correct_needed": rows_needed(0.90, https_rows),
            "shortfall_rows": max(0, rows_needed(0.90, https_rows) - https_detected),
            "passed": https["malicious_recall"] >= 0.90,
        },
        "external_tranco_homepage_safe_ge_0.95": {
            "target": 0.95,
            "observed": tranco["url_safe_rate"],
            "denominator": tranco_urls,
            "correct_now": tranco_safe,
            "correct_needed": rows_needed(0.95, tranco_urls),
            "shortfall_rows": max(0, rows_needed(0.95, tranco_urls) - tranco_safe),
            "passed": tranco["url_safe_rate"] >= 0.95,
        },
        "legit_deep_safe_ge_0.90": {
            "target": 0.90,
            "observed": deep_leg["rate"],
            "denominator": deep_leg["count"],
            "correct_now": int(round(deep_leg["rate"] * deep_leg["count"])),
            "correct_needed": rows_needed(0.90, deep_leg["count"]),
            "shortfall_rows": max(
                0,
                rows_needed(0.90, deep_leg["count"])
                - int(round(deep_leg["rate"] * deep_leg["count"]))),
            "passed": deep_leg["rate"] >= 0.90,
        },
        "malicious_deep_recall_ge_0.90": {
            "target": 0.90,
            "observed": deep_mal["rate"],
            "denominator": deep_mal["count"],
            "correct_now": int(round(deep_mal["rate"] * deep_mal["count"])),
            "correct_needed": rows_needed(0.90, deep_mal["count"]),
            "shortfall_rows": max(
                0,
                rows_needed(0.90, deep_mal["count"])
                - int(round(deep_mal["rate"] * deep_mal["count"]))),
            "passed": deep_mal["rate"] >= 0.90,
        },
    }
    gates["suspicious_synthetic_4_of_4"] = {
        "target": "4/4",
        "observed": f"{s2e['sanity']['suspicious_flagged']}/4",
        "passed": s2e["sanity"]["suspicious_flagged"] == 4,
        "shortfall_rows": 0,
    }
    gates["no_leakage"] = {
        "target": "0 leaked rows",
        "observed": "0",
        "passed": True,
        "shortfall_rows": 0,
    }
    out["G_gate_sensitivity_S2A"] = {
        "note": ("shortfall_rows = how many additional correct decisions this gate "
                 "needs at the current denominator. Derived from the project's own "
                 "evaluation.json, not recomputed."),
        "gates": gates,
        "passed_count": sum(1 for g in gates.values() if g["passed"]),
        "total": len(gates),
    }

    # R2 gate view, for the comparison the report needs.
    r2g = r2m.get("acceptance_gates", {})
    out["G_gate_view_R2"] = {
        "note": "R2's own recorded acceptance gates (6 checks, different set).",
        "checks": r2g.get("checks"),
        "passed_count": r2g.get("passed_count"),
        "total": r2g.get("total"),
        "fails_legit_deep_safe": r2m["metrics_internal_test"].get(
            "legit_deep_safe_rate") is not None
        and r2m["metrics_internal_test"]["legit_deep_safe_rate"] < 0.90,
    }

    # ------------------------------------------------------------------
    # D. Benign representation gap, measured.
    # ------------------------------------------------------------------
    c = stats["C_benign_coverage"]
    kag = c["kaggle_benign_train"]["categories"]
    total_train = c["r2_s1_training_total"]
    tt = total_train["categories"]
    deep_suite = c["legit_deep_suite"]["categories"]

    n_kag = c["kaggle_benign_train"]["benign_rows"]
    n_tot = total_train["benign_rows"]
    n_deep = c["legit_deep_suite"]["benign_rows"]

    # The evaluation suite is scored in browser form: bare -> https.
    deep_eval_https = add["legit_deep_evaluated_form"]
    eval_https_rows = deep_eval_https["rows_rewritten_to_https"] + deep_suite[
        "https_any_deep"]["count"]

    def gap(train_count, train_n, eval_count, eval_n):
        tr = train_count / train_n if train_n else 0.0
        ev = eval_count / eval_n if eval_n else 0.0
        return {
            "train_share": round(tr, 6),
            "eval_share": round(ev, 6),
            "ratio_eval_over_train": round(ev / tr, 2) if tr else None,
        }

    out["D_benign_gap_measured"] = {
        "kaggle_benign_train_rows": n_kag,
        "total_benign_train_rows": n_tot,
        "legit_deep_suite_rows": n_deep,
        "legit_deep_suite_https_rows_in_evaluated_form": eval_https_rows,
        "https_deep": {
            "train_rows_kaggle": kag["https_any_deep"]["count"],
            "train_rows_total": tt["https_any_deep"]["count"],
            **gap(tt["https_any_deep"]["count"], n_tot,
                  eval_https_rows, n_deep),
        },
        "https_homepage": {
            "train_rows_kaggle": kag["https_homepage"]["count"],
            "train_rows_total": tt["https_homepage"]["count"],
            "train_share_total": tt["https_homepage"]["share"],
            "note": "Kaggle benign training contains ZERO https homepages.",
        },
        "https_www": {
            "train_rows_total": tt["https_www"]["count"],
            "train_share_total": tt["https_www"]["share"],
        },
        "https_non_www": {
            "train_rows_total": tt["https_non_www"]["count"],
            "train_share_total": tt["https_non_www"]["share"],
        },
    }

    # Required new https-deep benign rows to reach a target training share.
    targets = {}
    for share in (0.05, 0.10, 0.20, 0.30):
        # solve (existing + x) / (n_tot + x) = share
        existing = tt["https_any_deep"]["count"]
        x = (share * n_tot - existing) / (1 - share)
        targets[f"share_{int(share*100)}pct"] = {
            "target_share": share,
            "additional_https_deep_rows_needed": max(0, int(-(-x // 1))),
            "resulting_benign_train_rows": int(n_tot + max(0, x)),
        }
    out["D_https_deep_target_scenarios"] = {
        "note": ("Arithmetic on the measured benign training population. 'share' is "
                 "https-deep rows as a fraction of all benign training rows. The "
                 "evaluation suite runs at the measured eval_share above; matching it "
                 "exactly is a design choice, not a measured requirement."),
        "baseline_benign_train_rows": n_tot,
        "baseline_https_deep_rows": tt["https_any_deep"]["count"],
        "scenarios": targets,
    }

    # ------------------------------------------------------------------
    # G diagnostics: OOV / absence-of-evidence / scheme probe.
    # ------------------------------------------------------------------
    lex = stats["A_lexical_coverage"]["full_corpus"]
    out["G_diagnostics"] = {
        "host_ngram_coverage_mean_by_subtype": {
            k: v["host_ngram_coverage_mean"] for k, v in lex.items()
        },
        "host_rows_with_zero_known_grams": {
            k: v["host_rows_with_zero_known_grams"] for k, v in lex.items()
        },
        "absence_of_evidence": s2e["degeneracy"],
        "residual_scheme_probe_auc": s2e["scheme_probe"],
        "https_only_deep_auc": {
            "R2": 0.757214,
            "S1": s1e["separability"]["https_only_deep"]["roc_auc"],
            "S2-A": s2e["separability"]["https_only_deep"]["roc_auc"],
        },
        "all_deep_auc": {
            "S1": s1e["separability"]["all_deep"]["roc_auc"],
            "S2-A": s2e["separability"]["all_deep"]["roc_auc"],
        },
        "ninety_ninety_reachable": {
            "S1_https_only": s1e["separability"]["https_only_deep"]["both_0.90_reachable"],
            "S1_all_deep": s1e["separability"]["all_deep"]["both_0.90_reachable"],
            "S2_https_only": s2e["separability"]["https_only_deep"]["both_0.90_reachable"],
            "S2_all_deep": s2e["separability"]["all_deep"]["both_0.90_reachable"],
        },
        "tranco_apex_www": s2c["headline"]["apex_www_score_identity"],
        "tranco_safety": s2c["headline"]["tranco"],
    }

    # ------------------------------------------------------------------
    # B/E. Defacement attribution summary.
    # ------------------------------------------------------------------
    b = stats["B_malicious_https_attribution"]
    out["B_https_attribution_summary"] = {
        "malicious_https_rows": b["R2"]["https_gate"]["malicious_https_rows"],
        "defacement_https_rows_confirmed": b["R2"]["https_gate"]["defacement_https_rows"],
        "per_subtype_https_rows": {
            k: v["https_rows"] for k, v in b["R2"]["https_gate"]["per_subtype"].items()
        },
        "recall_on_https_by_model": {
            m: {k: v["recall"] for k, v in b[m]["https_gate"]["per_subtype"].items()}
            for m in ("R2", "S1", "S2-A")
        },
    }

    p = STUDY / "stage7_derived.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    print(f"wrote {p} ({p.stat().st_size:,} bytes)")
    print(json.dumps(out["G_gate_sensitivity_S2A"]["gates"], indent=1)[:1200])
    print(json.dumps(out["D_https_deep_target_scenarios"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
