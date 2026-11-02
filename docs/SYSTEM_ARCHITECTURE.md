# FraudShield — System Architecture

*Complete request flow: **User → Frontend → FastAPI → URL predictor → Website
inspector → Risk engine → Final verdict** — with each component's
responsibility and how evidence is combined. Verified against the source files
listed in each section.*

---

## 0. Component directory

| # | Component | File | Responsibility |
|---|---|---|---|
| 1 | Frontend | `frontend/index.html`, `frontend/script.js`, `frontend/style.css` | Collect URL, normalise, call API, render verdict/availability/evidence, handle loading + errors |
| 2 | API layer | `backend/app.py` | Route definitions, request validation, error mapping, serve static frontend |
| 3 | Predictor | `backend/predictor.py` | Load R2 once per process; validate URL string; extract 14 lexical features; raw score + argmax + risk-level bands; trusted-URL policy notes |
| 3a | Feature contract | `backend/feature_extractor.py` | THE official 14-feature order; used identically by training and prediction |
| 3b | Policy | `backend/url_policy.py` | Deterministic structural validation of exact YouTube/Instagram hosts and shapes |
| 4 | Inspector | `backend/inspector.py` | Read-only live website inspection with SSRF/rebinding/TLS/size/timeout protections |
| 5 | Risk engine | `backend/risk_engine.py` | Findings generation + decision ladder + evidence assembly; availability field |
| 6 | Model artifact | `model/fraudshield_model.pkl` | Promoted R2 (SHA-256 `d7419091…a1bf8`) — load-only, never modified |

---

## 1. Request flow, end to end (`POST /api/scan`)

### 1.1 Frontend
`script.js`:
- `normalizeUrl()` + `new URL()` validation (only `http:`/`https:`; hostname required). Invalid input → inline error, **no network call**.
- Loading state (`setLoading`, progress animation) while `POST /api/scan` runs.
- Error handling: HTTP error responses surface their `detail.message`/`code` (e.g. `non_web_scheme`); network/server errors surface a generic message.
- Rendering (`renderResult`): verdict badge mapped through `VERDICT_LEVELS`/`VERDICT_META` (`SAFE`/`SUSPICIOUS`/`MALICIOUS`/`INSUFFICIENT EVIDENCE` → css levels `safe/suspicious/malicious/insufficient`), risk meter filled by *category* (never a fabricated percentage), availability line via `AVAILABILITY_LABELS`, destination strip (redirects, DNS count, TLS subject/issuer), evidence list, per-finding list, and the raw R2 line (`model_score`, flagged/benign, `trusted-URL structure verified` note when policy approved).

### 1.2 FastAPI (`backend/app.py`)
- `POST /api/scan` (`scan()`): validates the request body via `URLRequest` → resolves invalid input to 422/400 with a `code` (`non_web_scheme`, `empty_input`, …).
- Calls `predict_url(request.url)`; if the string itself is unusable → 400, **nothing is contacted**.
- Calls `inspect_website(request.url)` — the only network-performing component.
- Calls `decide(url_model_result, inspection)` — pure, no I/O.
- Assembles the response (add `availability` from the assessment; pass through `findings/evidence/signals/url_model/threat_intelligence`); unexpected exceptions → 500 with a generic message (never raw internals).
- `POST /api/predict` (`predict()`): URL-string-only legacy contract; no live inspection.
- `GET /api/health`: model + engine readiness.
- The `frontend/` directory is mounted last so `/` serves the UI.

### 1.3 Predictor (`backend/predictor.py`)
Responsibilities and hard guarantees:
- **Load-once** singleton (`thread-safe`), `n_jobs` forced to 1 after load (measured: avoids 18–35 ms thread overhead per single-row call; predictions unchanged).
- **Strict input validation** before any scoring: empty/whitespace, non-string, non-web schemes → `valid=False` with an error code; never scored.
- **Feature vector built by name** in `FEATURE_NAMES` order (`model_metadata.json` → `active_feature_names`), eliminating silent reordering (`n_features_in_ = 14`).
- Outputs: `model_score` (malicious-class `predict_proba`), `raw_prediction` (argmax at 0.5), `risk_level` using display bands safe `<0.35`, suspicious `0.35–<0.65`, high ≥0.65 (presentation policy, **not** optimized thresholds), plus policy notes from `url_policy.py`.
- **No network access** — URL parsed as text only.

### 1.4 Policy layer (`backend/url_policy.py`)
- Exact hostname sets (`TRUSTED_YOUTUBE_HOSTS`, `TRUSTED_INSTAGRAM_HOSTS`) — full-host match after stripping one trailing dot; no wildcards/suffix matching (so `instagram.com.evil.com` never matches).
- HTTPS-only, standard ports, no embedded credentials, no percent-encoded path tricks, per-kind path/parameter allowlists (`/watch?v=`, `/playlist?list=`, `/shorts/`, `youtu.be/<11 chars>`, Instagram `p|reel|reels|tv`, `/shares/`, `/stories/`, single-segment profiles; optional `igsh`/`igshid` with strict token regex).
- Result is a `PolicyResult` (approved/rejected/platform/kind). **Used by the predictor for the URL-only verdict presentation and as explanatory notes in the layered scan — never as an auto-SAFE for the final verdict.**

### 1.5 Inspector (`backend/inspector.py`)
Per results in `docs/SECURITY_AND_LIMITATIONS.md` (source of numbers): 5 max redirects, 6 s connect / 12 s read / 25 s budget, 2 MB body cap, per-hop DNS+IP validation, IP pinning with hostname kept for SNI/Host, default strict TLS verification, multi-validated-IP connect fallback, full taxonomy of result fields (`reachable`, `blocked`, `error`, `availability`, `title_available`, `availability_detail`, `host_flags`, `hops`, `final_url`, `status_code`, TLS facts, parsed `page`), body parsed for **all** final statuses (executable/binary refused before parsing), meta-refresh/iframe/form/link extraction. Returns a plain dict; **never raises** for ordinary failures.

### 1.6 Risk engine (`backend/risk_engine.py`)
- **Findings generators**: `_base_findings` (unsafe destination, executable served, HTTP entry, downgrade, TLS error, punycode/non-ASCII hostname), `_page_findings` (password/sensitive fields without TLS, cross-domain meta-refresh, mixed-content images, cross-origin iframes, truncation), `_impersonation_findings` (brand keyword in title on a non-official host namespace via `extract_brand_claim` + `_host_is_official_brand`).
- **Decision ladder** (exact code path order in `decide()`):
  1. `blocked` → INSUFFICIENT EVIDENCE (refusal is not proof of harm),
  2. not `reachable` → INSUFFICIENT EVIDENCE (a failed fetch never yields SAFE/MALICIOUS alone),
  3. any high finding → MALICIOUS,
  4. any medium finding → SUSPICIOUS,
  5. `fully_inspected` (reachable + is_html + TLS + no error + text>0 + not truncated) and no findings → SAFE — this is the only override of a flagged model score, and the evidence explicitly explains the override (and cites policy approval when present),
  6. `raw_prediction == 1` with thin/unavailable content → SUSPICIOUS (URL-feature evidence stands; content couldn't refute it),
  7. otherwise → INSUFFICIENT EVIDENCE.
- **Availability** (`_availability_of`): taken from inspection transport/status facts (with a fallback mapping for pre-field data), one of the seven taxonomy values; never feeds the verdict.
- **Evidence assembly**: prelude, availability line, DNS validated-IPs line, redirect chain line, TLS line, content line, findings line (or explicit "content inspection produced no direct findings" when SUSPICIOUS is model-driven), R2 relationship line (override/agreement/sole-signal phrasing chosen by context).
- **Signals block**: content risk (severity-weighted, capped), finding counts, `r2_raw_decision`, `r2_model_score`, `full_inspection`, `redirects_followed`.

## 2. How evidence is combined (design principles)

1. **Raw model output is always preserved and never decisive.** `url_model.model_score`/`raw_prediction` echo the model verbatim (property-tested for all bank entries). The engine uses the *live destination's* verdict, and the model acts as (a) an override subject only when the destination checks out, (b) a sole-signal SUSPICIOUS driver when content cannot refute it.
2. **Every claim must trace to a finding or a measured fact.** Findings are generated only from recorded inspection fields; availability only from transport/status; verdict preludes explain the class chosen.
3. **Uncertainty is a first-class outcome.** Blocked, unreachable, TLS-error, thin-content → INSUFFICIENT EVIDENCE / SUSPICIOUS instead of fabricated SAFE/MALICIOUS.
4. **Overrides are loud.** The exact override sentence appears in `evidence` whenever a flagged R2 score is beaten by full inspection.
5. **No auto-whitelisting.** Brand namespaces use dot-anchored official-host semantics (`_host_is_official_brand`); the policy layer validates structure, not reputation.

## 3. `POST /api/predict` flow (URL-only path)

Same steps 1.1–1.4 **without** 1.5–1.6: the raw model contract returns
`valid`, `score`, `risk_level`, `prediction_label`, `model` metadata and policy
notes. Its verdict rules are predictor-policy bands (presentation). This is the
layer where the documented deep-path limitation is most visible — with live
inspection this limitation is compensated (not hidden).

## 4. Data flow diagram

```
            ┌──────────── frontend (script.js) ────────────┐
            │ normalizeUrl → POST /api/scan → renderResult │
            └───────────────────────┬──────────────────────┘
                                    ▼
                          FastAPI  (backend/app.py)
                    URLRequest validation → scan()/predict()
                                    ▼
                 predictor.predict_url(url)          ← text only, no network
                 ├─ FEATURE_NAMES × 14 ──► forest ──► model_score, raw_prediction
                 └─ url_policy.evaluate_trusted_url_policy(url) → notes
                                    ▼
                 inspector.inspect_website(url)      ← the ONLY network I/O
                   validate scheme/host/port ─►  (hop 0…5)
                     └─ resolve+validate IPs (SSRF blocklist, per hop)
                        └─ connect pinned IP (SNI=hostname), strict TLS
                           └─ redirects revalidated │ TLS facts │ body≤2MB
                              └─ HTMLParser (read-only) → page dict
                                    ▼
                 risk_engine.decide(url_model_result, inspection)
                   findings ─► severity ladder ─► verdict
                   transport facts ─► availability (independent)
                   evidence assembly ─► explanation lines
                                    ▼
       JSON: verdict · availability · evidence · findings · url_model(raw)
             inspection · security · signals · threat_intelligence
```
