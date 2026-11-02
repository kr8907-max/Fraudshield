# FraudShield QA Evaluation Report

Date: 2026-11-02 · Build: layered scanner (R2 URL model + live inspector + risk engine + policy layer)
Model: `model/fraudshield_model.pkl` — SHA-256 `d7419091f6...76a1bf8` **verified unchanged** on this date; no retraining, no threshold change.

> **Synthetic vs. real-world.** Section A/B rates are measured against a
> permanent deterministic fixture bank (`tests/test_evaluation.py`) using the
> real risk engine and inspector logic, not live websites. Section C contains
> the small number of live-network measurements and reproductions.

---

## A. Engine behaviour on the synthetic evaluation bank (permanent tests)

| Metric | Result (synthetic bank) |
|---|---|
| Legitimate false-positive rate | **0/20 (0%)** on 20 crafted legitimate inspections, evaluated with the string model FLAGGED (`raw_prediction=1, score=0.95`) |
| Malicious detection rate | **16/16 (100%)** flagged SUSPICIOUS-with-signal or MALICIOUS on 16 crafted malware/phishing/impersonation/homoglyph inspections with the model BENIGN (no model help). 13/16 reached the MALICIOUS verdict; 3/16 surfaced SUSPICIOUS (signal detected, verdict conservative) |
| Unverified discipline | **5/5** failure/shell cases produce INSUFFICIENT EVIDENCE for a benign model; never SAFE, never MALICIOUS; the flagged view of a thin shell is SUSPICIOUS (model flag stands, cannot be refuted) |
| Raw-R2 preservation property | **PASS** on all 41 bank entries: `signals.r2_raw_decision` / `r2_model_score` always echo the model verbatim; the verdict never comes from the raw score |

Limitations of the synthetic bank: it validates decision logic, not recall on
real phishing kits. Real-world recall was NOT measured (would require a curated
live phishing corpus — none is bundled; visiting live malware is prohibited).

## B. Security audit results (all automated, permanent)

| Area | Test | Result |
|---|---|---|
| SSRF blocklist | 10 IPv4/IPv6 private ranges + `169.254.169.254`, `100.64/10`, `198.18/15`, broadcast | PASS (blocked before contact) |
| IPv4-mapped IPv6 bypass | `::ffff:127.0.0.1`, `::ffff:10.0.0.5`, `::ffff:169.254.169.254` | PASS on Python 3.14 (`ipv4_mapped` handled by stdlib; guarded by explicit tests) |
| resolver-level bypass tricks | `0177.0.0.1`, `0x7f.1`, `2130706433` | blocked at resolution stage: value fails or is resolved to a blocked IP, nothing contacted |
| DNS rebinding shape | per-hop revalidation, private IP result | refused before contact |
| Host-level getaddrinfo OSError | monkeypatched resolver crash | reported as `dns_resolution_failed`; no crash |
| Multi-address fallback | first validated IP refuses connect | falls back to remaining validated IPs; TLS errors do NOT hop (host-level), all-fail reports CONNECTION_FAILED |
| Redirect validation | scheme/credentials/percent-encoding rules, every hop revalidated | PASS (existing suite) |
| TLS verification | default verifying context + SNI = hostname validated; TLS failure never yields MALICIOUS (unreachable ⇒ INSUFFICIENT) | PASS |
| Response-size limits | hard 2 MB cap, truncation recorded | PASS |
| Timeouts | 6 s connect / 12 s read / 25 s budget; body-timeout mapped to availability TIMEOUT | PASS |
| Malicious HTML handling | HTMLParser only, never executes JS/fills forms; parser exceptions tolerated | PASS |
| Input validation | non-web schemes, embedded credentials, invalid host/port | PASS |

**Fixes shipped in this phase** (production code):
1. `backend/inspector.py` — `getaddrinfo` non-`gaierror` `OSError` handled (was a crash path).
2. `backend/inspector.py` — multi-IP connect fallback for dual-stack hosts (was: first resolved IP only).
3. `backend/inspector.py` — `_host_is_official_brand()` with dot-anchored subdomain semantics: **fixes real FP class** (e.g. `www.amazon.com` would previously have been flagged MALICIOUS; only `amazon.com` was in the official set).
4. `backend/inspector.py` + `backend/risk_engine.py` — punycode/non-ASCII hostnames recorded (`host_flags`) and surfaced as MEDIUM `deceptive_hostname` finding ⇒ SUSPICIOUS, never silently trusted.
5. `backend/risk_engine.py` — evidence hygiene: the "agrees with inspection findings" R2 line is no longer printed when there are no findings; the pure-model-flag case names the model as the main signal.

## C. Live-network measurements (small, environment-dependent)

| URL | Verdict | Availability | R2 score | Inspection | elapsed (ms) |
|---|---|---|---|---|---|
| instagram.com/khushiroy.31 (profile) | SAFE | REACHABLE | 0.996271 (flagged, overridden + explained) | 200, full page, valid DigiCert TLS | 729 |
| github.com/microsoft/vscode | SAFE | REACHABLE | flagged, overridden | 200, full page | 1249 |
| httpbin.org/status/404 | SUSPICIOUS | PAGE_NOT_FOUND | 0.951676 | 404 without body; URL-feature classification proceeds | 1082 |

Live inspection budget observed 0.7–1.3 s per reachable site (cap 25 s).
Live lookalike domains (`instagram.com.evil.com`, `instagram-login.example.com`)
resolve-fail in this environment ⇒ INSUFFICIENT EVIDENCE + DNS_FAILED, policy
never approves them; engine-level synthetic tests cover the resolved case
(deceptive/punycode ⇒ SUSPICIOUS at minimum, impersonating title ⇒ MALICIOUS).

## D. Test suites (post-change)

| Suite | Count |
|---|---|
| `tests/` total | **254 passed** (includes url_policy, inspection, engine, API integration, QA evaluation bank) |
| of which new this phase | 108 permanently (QA bank + security audit + API verdicts + live platform/deep-path/redirect cases) |
| predictor self-test | 31/31 |

## E. Remaining limitations (no claim of perfect accuracy)

1. Synthetic malicious bank proves verdict logic, not real-world recall; no live phishing corpus is bundled.
2. Punycode hosts are always flagged SUSPICIOUS ⇒ internationalized-but-legitimate domains will be cautious-graded (deliberate, reversible trade-off).
3. `deceptive_hostname`/brand detection rely on page titles; JS-rendered pages (Instagram shells) yield thin content ⇒ model-flag-only SUSPICIOUS.
4. Threat-intelligence feeds remain unconfigured (documented as "not configured", not faked).
5. Brand table covers 17 high-value brands; niche brands are not represented.
6. FraShield does NOT currently measure FP/recall on live datasets — treat all percentages above as logic-verification, not field performance.

## F. Launch commands

```powershell
cd C:\fraudshield
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000         # app + UI at http://127.0.0.1:8000
.\venv\Scripts\python.exe -m pytest tests\ -q                          # full suite
.\venv\Scripts\python.exe backend\predictor.py --self-test             # model self-check
```
