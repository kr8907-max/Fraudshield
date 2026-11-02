# FraudShield — Project Overview

*Beginner-friendly introduction that stays technically accurate. Every claim in
this file points to the code, metadata or reports that back it.*

---

## 1. What problem does FraudShield solve?

When you receive a link, two different questions matter:

1. **Is this *string* suspicious?** (weird domain, many hyphens, `@` sign, words
   like "login-verify-account"…) — a *lexical* question you can answer without
   opening the page.
2. **Is the *website* at the end of the link actually malicious?** — a *live*
   question that needs DNS, TLS, redirects and page content.

A single squint at the string answers neither reliably. The promoted R2 model
demonstrates this itself: it scores ordinary legitimate pages like
`https://www.google.com/search?q=fraud` at **0.9761** ("HIGH RISK") and
`https://en.wikipedia.org/wiki/Phishing` at **0.9823**, because its most
important feature (`num_slashes`, importance 0.2618) learned how the training
dataset *wrote* URLs, not what danger looks like (measured and documented in
`model/model_metadata.json` → `known_limitation`).

But a live scanner alone has the opposite failure: it *reaches* a phishing site
and can be fooled; and when it *cannot* reach a site, it must not invent a
verdict.

**FraudShield therefore runs both questions independently and combines them:
raw lexical score (kept honest, never decisive) + read-only live inspection +
deterministic structural policy notes — joined in one risk engine that shows
its work.**

---

## 2. What does it output?

For `POST /api/scan`:

| Field | Meaning |
|---|---|
| `verdict` | Security assessment: `SAFE`, `SUSPICIOUS`, `MALICIOUS`, `INSUFFICIENT EVIDENCE` |
| `availability` | Website availability: `REACHABLE`, `PAGE_NOT_FOUND`, `TIMEOUT`, `CONNECTION_FAILED`, `DNS_FAILED`, `TITLE_UNAVAILABLE`, `UNKNOWN` — **mapped only from transport/status facts, never a security statement** |
| `findings` | Individual observations with severity (high/medium/low/info) and code, e.g. `brand_impersonation`, `password_over_http`, `deceptive_hostname` |
| `evidence` | Human-readable lines explaining why the verdict was reached |
| `url_model` | The raw R2 output (`model_score`, `raw_prediction`) — **preserved verbatim, never the decider** |
| `inspection` / `security` / `signals` | Full inspection dict, TLS/DNS/redirect facts, per-source signal summaries |

Example (live, 2026-11-02 session): `https://www.instagram.com/khushiroy.31/`
→ `verdict=SAFE`, `availability=REACHABLE`, R2 raw score 0.996271 **flagged** but
overridden by a verified real Instagram profile page, with the override explained
in `evidence`.

`POST /api/predict` keeps the older URL-only contract (no live inspection).

## 3. How is a scan performed? (bird's-eye view)

1. **Input validation** (FastAPI layer and `predictor.py`): only `http`/`https`
   strings are accepted; evil schemes ("javascript:"), embedded
   credentials, empty input → clean 4xx error, no scanning.
2. **Raw R2 score** (`backend/predictor.py`): 14 lexical features → forest
   probability. Reported as-is. The 0.35/0.65 display bands and the raw 0.5
   argmax decision are predictor-layer presentation; the layered verdict is
   decided by the risk engine, not by these bands.
3. **Trusted-URL policy** (`backend/url_policy.py`): deterministic structural
   checks (exact YouTube/Instagram hostnames, validated path/parameter shapes).
   Affects the *URL-only* verdict's presentation; the final layered verdict does
   not get auto-approved by policy.
4. **Live inspection** (`backend/inspector.py`): resolve + validate every
   destination IP (SSRF blocklist), connect to a *pinned* validated IP while
   keeping the hostname for TLS SNI/Host, follow up to 5 redirects with
   revalidation per hop, verify the TLS chain, read at most 2 MB with strict
   timeouts, parse the HTML read-only (never execute JS, never submit forms).
5. **Risk engine** (`backend/risk_engine.py`): converts inspection facts into
   findings (with severities), then applies a fixed decision ladder:
   - blocked / unreachable → **INSUFFICIENT EVIDENCE** (never SAFE, never MALICIOUS from a failure alone);
   - any high-severity finding → **MALICIOUS**;
   - any medium finding → **SUSPICIOUS**;
   - full benign inspection (reachable + TLS + real HTML content + complete read) → **SAFE** — this is the *only* path that can override a flagging model score, and it prints an explanation;
   - unrefuted model flag with thin/unavailable content → **SUSPICIOUS** (the model is the main named signal; page content could not confirm or refute it);
   - otherwise → **INSUFFICIENT EVIDENCE**.
6. **Response**: verdict + availability + evidence + findings + raw model, plus the full inspection dict for transparency.

A failed HTTP request never implies safe *or* malicious; a 404, missing title,
empty body or slow response never blocks URL-feature-based classification.

## 4. Key protection ideas (why this is safe to run)

- **Never visitors' machines at risk:** the inspector refuses to contact
  loopback/private/link-local/cloud-metadata destinations. The blocklist runs on
  the *resolved IP* at **every** redirect hop, including IPv4-mapped IPv6
  addresses on supported runtimes (tests assert the mapped forms are blocked).
- **DNS-rebinding resistant:** once DNS answers, the connection is pinned to a
  validated IP; the hostname is only used for TLS identity.
- **Never downloads or executes:** JS is parsed as text only; executable/binary
  content types are recorded and refused before deeper parsing.
- **Bounded network use:** 6 s connect / 12 s read / 25 s total; 2 MB response
  cap; first validated IP tried, remaining validated IPs used as fallback when a
  connection is refused (TLS failures stop immediately — they're host-specific).

Full details: `docs/SECURITY_AND_LIMITATIONS.md`.

## 5. What the project is and is not

| It IS | It is NOT |
|---|---|
| A research/scanner tool for URL triage | A browser sandbox or antivirus |
| Honest about uncertainty (INSUFFICIENT EVIDENCE) | Perfect-accuracy oracle |
| Source-grounded: every number traces to `model/model_metadata.json`, experiment artifacts, or `reports/qa_evaluation_2026-11-02.md` | A claim of live-corpus field performance (none measured) |

## 6. Folder map (what lives where)

| Path | Role |
|---|---|
| `backend/` | `app.py` (FastAPI), `predictor.py` (R2 engine), `feature_extractor.py` (14 features + contract self-checks), `inspector.py` (read-only inspection), `risk_engine.py` (decision layer), `url_policy.py` (YouTube/Instagram structural validation) |
| `frontend/` | `index.html`, `script.js`, `style.css` (static UI, served by FastAPI at `/`) |
| `tests/` | `test_inspection.py`, `test_url_policy.py`, `test_api_integration.py`, `test_evaluation.py` — 254 tests passing; predictor self-test 31/31 |
| `model/` | `fraudshield_model.pkl` (promoted R2 artifact — **never modify**), `model_metadata.json` (full training/provenance/metrics record), `experiment_c/`, `experiment_d/`, `experiment_six/`, `experiment_seven/` (feasibility probe), `stage7_study/` (data redesign study), `archive/` |
| `data/` | `malicious_phish.csv` (training-time Kaggle dataset), `tranco/top-1m.csv` (sampling frame), `test_urls.csv` (tiny pipeline fixture — *not* training data, see `data/README.md`) |
| `analysis/stage7/` | addendum scripts/stats of the Stage 7 study |
| `reports/` | QA evaluation reports (latest: `qa_evaluation_2026-11-02.md`) |
| `docs/` | This documentation set |

## 7. Glossary

- **R2** — the promoted model artifact: RandomForest, 100 trees, 14 lexical features (winner of Experiment C, variant `r2`).
- **model_score** — `predict_proba` malicious-class output. **Not calibrated**; treat as a risk signal, not a probability.
- **raw_prediction** — model's own argmax at 0.5; preserved everywhere.
- **fully inspected** — reachable + valid TLS + real HTML content + complete (untruncated) read. Only this state supports a SAFE override of a flagging score.
- **Availability** — transport/status-only field (see §2).
- **Policy layer** — deterministic shape validation for exact YouTube/Instagram hosts; lookalikes are never approved by it.
- **Punycode** — ASCII encoding of internationalized domain names (`xn--…`); frequent in homoglyph attacks, always flagged SUSPICIOUS here.
