# FraudShield — Testing and Evaluation

*This document consolidates the current state of verification. Its primary
source is the full QA evaluation `reports/qa_evaluation_2026-11-02.md` plus the
promoted model's own record `model/model_metadata.json`. Every figure below is
traceable; synthetic and live measurements are always labelled.*

---

## 1. Test suites and current counts (measured this documentation phase)

| Suite file | Parametrized test count | Character |
|---|---:|---|
| `tests/test_evaluation.py` | 150 | **Synthetic** QA bank: legitimate/malicious/unverified/edge cases + security tests, runs the real engine on crafted inspection dicts — deterministic, no network |
| `tests/test_inspection.py` | 63 | Mixed: inspector internals + engine units (**synthetic**) and marked live-network tests (`KNOWN_LEGITIMATE`, auto-skipped offline) |
| `tests/test_url_policy.py` | 34 | Synthetic structural-policy contract tests |
| `tests/test_api_integration.py` | 7 | API-level: FastAPI TestClient with synthetic backends (detect no live network in CI path except marked suites) |
| **Total** | **254 passing** | `pytest tests\ -q` — latest run: 254 passed, 1 deprecation warning (from `fastapi/testclient`, third-party, not project code) |

Predictor contract self-test: **`python backend\predictor.py --self-test` →
31/31 checks passed** (feature-order contract, HTTPS semantics, parse
fallbacks, sanity URL predictions, invalid-input handling — recorded in
`model_metadata.json` → `stage5_validation.self_test`).

Model artifact integrity: `model/fraudshield_model.pkl`
SHA-256 `d74190913f656a4260578e4dfa1b4b8d17bc4b0681e1215d18974713876a1bf8` —
recomputed from disk after the QA phase and again in this documentation phase:
**unchanged**. (No retraining, no threshold change.)

## 2. Synthetic evaluation results (deterministic bank — logic verification)

Source: `reports/qa_evaluation_2026-11-02.md` §A, produced by
`tests/test_evaluation.py` against the real risk engine.

| Metric | Result | What it verifies |
|---|---|---|
| Legitimate false-positive rate | **0/20 (0%)** — 20 crafted legitimate inspections (social/profiles, GitHub repos/profiles, video/playlists, cloud & dev platforms, e-commerce/banking-form sites, deep paths & query params), *evaluated with the string model FLAGGED* (raw_prediction=1, score 0.95) | the layered verdict never rises to MALICIOUS/high findings on benign evidence even when R2 disagrees |
| Malicious detection rate | **16/16 (100%)** detected (13 MALICIOUS verdicts, 3 conservative SUSPICIOUS-with-signal) — 16 crafted threats incl. brand impersonation/lookalikes, deceptive subdomains (`paypal.com.verify-account.xyz`), fake logins over HTTP, credential-harvesting forms, punycode/Cyrillic homoglyph hosts, executable downloads, deceptive meta-refresh redirects — *evaluated with the model BENIGN (no model help)* | inspection findings alone carry the verdict |
| Unverified discipline | **5/5** failure/shell cases → INSUFFICIENT EVIDENCE for a benign model; never SAFE, never MALICIOUS; a flagged thin shell is SUSPICIOUS (named sole signal) | failed/incomplete fetches never become claims |
| Raw-R2 preservation property | hold on **all 41 bank entries × 2 model views** | `signals.r2_raw_decision` / `r2_model_score` always echo the model verbatim; the verdict never comes from the raw score |
| Evidence-vs-verdict consistency | property tests across banks | prelude text, availability field, finding severities and the decision ladder agree |

**Important reading:** these rates verify the *decision logic*, not field
recall on real phishing kits. No live phishing corpus is bundled, and visiting
live malware is prohibited by project rules — so no live-corpus detection rate
exists anywhere in this project.

## 3. Security test results (all automated, permanent)

Source: `reports/qa_evaluation_2026-11-02.md` §B; implemented across
`tests/test_evaluation.py`, `tests/test_inspection.py`, `tests/test_url_policy.py`.

| Area | Verified behaviour | Status |
|---|---|---|
| SSRF blocklist | 10 IPv4/IPv6 private/reserved ranges + `169.254.169.254` metadata, `100.64/10` CGNAT, `198.18/15` benchmark, broadcast — blocked **before any connection** | PASS |
| IPv4-mapped IPv6 | `::ffff:127.0.0.1`, `::ffff:10.0.0.5`, `::ffff:169.254.169.254` blocked | PASS (runtime Python 3.14; explicit tests guard this) |
| Resolver-level bypass tricks | `0177.0.0.1`, `0x7f.1`, `2130706433` — fail to resolve or resolve to blocked IP; nothing contacted | PASS |
| DNS rebinding shape | per-hop re-validation; private-IP result refused pre-contact; IP pinning with hostname kept for SNI/Host | PASS |
| Resolver OS errors | non-`gaierror` `OSError` handled as `dns_resolution_failed` (no crash) | PASS |
| Multi-address fallback | connect falls through validated IPs when refused; TLS failures stop immediately (host-level); all-fail → `CONNECTION_FAILED` | PASS |
| Redirect validation | every hop revalidated; scheme must stay web; HTTPS→HTTP downgrade recorded; embedded credentials rejected; ≤5 hops | PASS |
| TLS verification | default strict context; TLS failure reports honestly (connection failure → INSUFFICIENT EVIDENCE), never MALICIOUS | PASS |
| Size/timeout limits | 2 MB cap with honest truncation; 6 s connect / 12 s read / 25 s budget; body-timeout → availability TIMEOUT | PASS |
| Malicious HTML handling | HTMLParser reads as text only; JS never executed; forms never submitted; executable/binary content types refused before parse; parser exceptions tolerated | PASS |
| Input validation | non-web schemes, empty/whitespace, non-string, invalid host/port, embedded credentials → clean `valid=False`/4xx, no scanning | PASS |
| Availability taxonomy | 7 values mapped only from transport/status facts; availability independent of verdict | PASS |

## 4. Live inspection examples (real-world measurements, environment-dependent)

From the 2026-11-02 QA session (through the real `/api/scan` / `inspect_website`):

| URL | Verdict | Availability | R2 raw score | Measured facts | Inspection time |
|---|---|---|---|---:|---|
| `https://www.instagram.com/khushiroy.31/` | SAFE | REACHABLE | 0.996271 *flagged* (overridden, explained) | 200 · full profile page (title + 799 chars) · DigiCert TLS `*.www.instagram.com` · DNS validated | 729 ms |
| `https://github.com/microsoft/vscode` | SAFE | REACHABLE | flagged (overridden, explained) | 200 · full page | 1249 ms |
| `https://httpbin.org/status/404` | SUSPICIOUS | PAGE_NOT_FOUND | 0.951676 | 404 without body → URL-feature classification proceeds (not blanket-UNVERIFIED) | 1082 ms |
| `https://httpbin.org/status/200` (empty) | SUSPICIOUS | TITLE_UNAVAILABLE | 0.951676 | thin 200; model flag stands, cannot be refuted | ~1.4 s |
| `https://httpbin.org/delay/3` | SUSPICIOUS | TITLE_UNAVAILABLE | 0.956294 | slow JSON response; body present, non-HTML | ~4.3 s |
| `http://localhost:8000/` | INSUFFICIENT EVIDENCE | UNKNOWN | 0.585957 | SSRF-blocked destination; nothing contacted | — |
| `https://instagram.com.evil.com/khushiroy.31/` | INSUFFICIENT EVIDENCE | DNS_FAILED | 0.98865 | no resolution; policy never approves lookalikes | — |
| `https://www.amazon.com/` | SAFE | REACHABLE | 0.920372 flagged (overridden) | confirms the dot-anchored brand-namespace fix (was a real FP class) | — |

Neighbour hyperparameters measured live: per-hop revalidation on the
`youtu.be → www.youtube.com` chain in `test_inspection.py` (skip-guarded).

Performance observation: 0.7–1.3 s inspection wall time per reachable site vs
a 25 s cap (sample of 3, single Windows machine). String-only prediction is
fast by measurement: mean 4.44 ms/prediction over 500 iterations
(`model_metadata.json` → `performance_local_string_only`).

## 5. Model-level published metrics (training-time, recorded provenance)

Summarised here; full detail in `docs/MODEL_METHODOLOGY.md` and
`model/model_metadata.json`:

- **Internal held-out test (128,214 rows):** accuracy 0.8941 · precision 0.7946 ·
  recall 0.9188 · F1 0.8522 · ROC AUC 0.9652 (TN 75,494 / FP 10,119 / FN 3,458 / TP 39,143).
- **External 4,000-URL Tranco homepage holdout (2,000 domains, never used for
  fitting/tuning):** URL safe rate 99.55%, domain safe rate 99.1%.
- **Known limitation (measured in Stage 5):** homepage-validated model flags
  ordinary deep-path URLs HIGH RISK (e.g. `google.com/search?q=fraud` 0.9761,
  `wikipedia.org/wiki/Phishing` 0.9823, `github.com/microsoft/vscode` 0.9565).
- These are **controlled-suite numbers** (labelling discipline: held-out split
  grouped by hostname, holdout domains untouched by fitting/tuning). Not a
  field-trial result.

## 6. Limitations of the current evaluation

1. **No live-corpus FP/recall measurement exists.** All detection rates
   above are synthetic-bank logic verification; the only live checks are the
   small exemplar set in §4 plus the network-marked tests.
2. Synthetic malicious cases are *ours* — a real adversary with different
   patterns could evade what the bank cannot express.
3. Live tests are environment-dependent (DNS, geo, anti-bot walls); they skip,
   never fail, when infrastructure disagrees.
4. The punycode policy flags every IDN host SUSPICIOUS — internationalized
   legitimate domains get cautious verdicts (deliberate trade-off).
5. Performance numbers come from one Windows workstation sample; no load
   testing was performed.
6. Third-party warning: `StarletteDeprecationWarning` from the installed
   fastapi/testclient — environmental, not project code.

## 7. Reproducing the evaluation

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest tests\ -q                              # 254 tests
python backend\predictor.py --self-test                 # 31/31
python -c "import hashlib;print(hashlib.sha256(open('model/fraudshield_model.pkl','rb').read()).hexdigest())"
# expect: d74190913f656a4260578e4dfa1b4b8d17bc4b0681e1215d18974713876a1bf8
```

Marked live tests run when the network is available; offline they are skipped
by design.
