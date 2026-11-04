# FraudShield

**A layered website security scanner** — live website inspection combined with an
R2 URL-string machine-learning model, producing an evidence-based risk verdict.

FraudShield addresses a real failure mode of single-signal URL scanners: a model
that scores URL *strings* cannot tell `https://example.org/about` from
`https://paypa1-login.xyz/verify` reliably, and a network scanner cannot judge a
URL it never reached. FraudShield does both, keeps the two evidence streams
separate, and refuses to claim more than the evidence supports.

- **Problem solved:** lexical URL classifiers flag ordinary legit deep links and cannot see live pages; naive whitelists cannot see lookalike phishing domains.
- **Approach:** independent raw-model score + read-only live inspection (DNS/SSRF validation, TLS check, redirect re-validation, HTML parsing) + structural policy layer, combined in an explicit risk engine whose evidence trail is visible.

## Objectives
1. Classify URLs **SAFE / SUSPICIOUS / MALICIOUS / INSUFFICIENT EVIDENCE** from combined, explainable evidence.
2. Report **website availability** (REACHABLE / PAGE_NOT_FOUND / TIMEOUT / CONNECTION_FAILED / DNS_FAILED / TITLE_UNAVAILABLE / UNKNOWN) independently of the security verdict.
3. Never claim SAFE or MALICIOUS from a failed fetch alone; never let a missing title or 404 block URL-feature classification.
4. Preserve the promoted R2 model artifact untouched (SHA-256 below) — fixes are made in the engine, not the model.
5. Every verdict carries human-readable `evidence` strings and per-severity `findings`.

## Key features
- **Layered scan** (`POST /api/scan`): raw R2 score (preserved, non-decisive) + full live inspection + policy notes → verdict with evidence.
- **URL-only prediction** (`POST /api/predict`): legacy R2 contract, raw `model_score`, `raw_prediction`, risk bands.
- **Read-only inspector**: no JavaScript execution, no form submission, no uploads, executable/binary downloads refused.
- **Availability taxonomy** mapped only from transport/status facts.
- **Trusted-URL structural policy** for YouTube and Instagram (exact hostnames, validated parameter shapes, `igsh`/`igshid` accepted) affecting the *URL-only* verdict; final scan verdicts always come from live evidence.
- **Brand-impersonation detection** from page titles with dot-anchored official-host namespaces (subdomains of `amazon.com` are Amazon; `paypal.com.evil.xyz` is not).
- **Punycode/IDN hostname flagging** (`deceptive_hostname` medium finding).
- **Modern UI** at `http://127.0.0.1:8000/` (verdict badge, availability line, evidence list, raw-model line).

## Technology stack
| Layer | Technology |
|---|---|
| Backend/API | Python 3.14, FastAPI, Uvicorn |
| Model | scikit-learn RandomForestClassifier (100 trees), loaded with joblib |
| URL features | Standard library only (`backend/feature_extractor.py`, 14 lexical features) |
| Inspector | Standard library only (`http.client`, `ssl`, `socket`, `html.parser`) — no third-party fetch stack |
| Frontend | Static HTML/CSS/JS (`frontend/`), optional Vite dev server |
| Testing | pytest, FastAPI TestClient |
| Datasets | Kaggle `malicious_phish.csv` (training-time only), Tranco top-1M (sampling frame), both stored locally under `data/` |

## System requirements
- Windows 10/11 with PowerShell (`pwsh`/Windows PowerShell 5.1+)
- Python 3.14 (venv at `venv\`; any modern CPython 3.10+ works for the code, docs are written against the shipped interpreter)
- Optional: Node.js for the Vite dev-server workflow only

## Installation (Windows PowerShell, Python 3.14)
```powershell
cd C:\fraudshield
python -m venv venv                       # first time only
.\venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # if activation is blocked
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn joblib scikit-learn numpy pytest httpx
```

## Launch
```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000** — enter a URL and scan. `--reload` can be added for development.

## Run tests
```powershell
.\venv\Scripts\Activate.ps1
python -m pytest tests\ -q                  # 254 tests currently pass
python backend\predictor.py --self-test     # 31/31 contract checks
```
Live-network tests are marked and automatically skipped when offline.

## Deployment (Render)

The repository ships a ready Render blueprint in **`render.yaml`**:

- Service: `web` (Python runtime, **free plan**), name `fraudshield`.
- Build: `pip install -r requirements.txt` (runtime dependencies pinned; dev/test tooling like pytest is not installed on Render).
- Start: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
- Health check: `/api/health`.
- The 29.3 MB model artifact `model/fraudshield_model.pkl` is committed to the repository, so no model download or upload step is needed at deploy time.

To deploy: push the repository to GitHub, then in the Render dashboard choose
"New Web Service" → "Blueprint" and point it at the repository (or use the
"Deploy to Render" flow with `render.yaml`). The free plan will sleep between
requests; the first request after sleeping incurs a cold start while the model
loads.

### Environment variables

| Variable | Where set | Values | Effect |
|---|---|---|---|
| `PORT` | Render (automatic) | port number | Port the app binds to in the start command. |
| `FRAUDSHIELD_ENV` | `render.yaml` (or dashboard) | default `development`; `production` / `prod` for the deployed service | In production mode the interactive API docs (`/docs`, `/redoc`, `/openapi.json`) are **disabled**; locally they remain available. |
| `FRAUDSHIELD_ENABLE_API_DOCS` | optional, dashboard only | `1` / `true` / `yes` | Force-re-enables the interactive API docs even when `FRAUDSHIELD_ENV=production` (explicit local override). |

**No secrets are required to deploy FraudShield** — there are no API keys, no
database credentials, and no third-party service tokens. Never add secret
values to this repository; if a token is ever needed, Render's dashboard
"Environment" tab (kept out of Git) is the only correct place for it.

## Architecture overview
```
User → Frontend (script.js) → FastAPI (backend/app.py)
     → predictor.py (raw R2 score, features, policy notes)
     → inspector.py (read-only live inspection: DNS/SSRF → connect → redirects → TLS → HTML)
     → risk_engine.py (findings + fully-inspection + raw R2 → verdict + availability + evidence)
     → response { verdict, availability, evidence, findings, inspection, security, url_model, signals }
```

- **Verdicts**: SAFE / SUSPICIOUS / MALICIOUS / INSUFFICIENT EVIDENCE.
- **Availability** is a separate field mapped from transport/status facts — a failed or unreadable page never becomes a security claim.
- Layered scan decision ladder (verified in tests): blocked/unreachable → INSUFFICIENT EVIDENCE; high findings → MALICIOUS; medium findings → SUSPICIOUS; full benign inspection → SAFE (string-model override explained); unrefuted model flag with thin content → SUSPICIOUS; otherwise INSUFFICIENT EVIDENCE.

Details: `docs/SYSTEM_ARCHITECTURE.md`, `docs/PROJECT_OVERVIEW.md`.

## Model information and actual evaluation metrics
- Artifact: `model\fraudshield_model.pkl` — **SHA-256
  `d74190913f656a4260578e4dfa1b4b8d17bc4b0681e1215d18974713876a1bf8`** (unchanged across all
  engineering phases; verified from disk after every phase).
- Kaggle `malicious_phish.csv`: 651,191 original rows, 641,084 usable (phishing 94,111 · benign 428,103 · defacement 96,457 · malware 32,520).
- Internal held-out test (128,214 rows): accuracy 0.8941 · precision 0.7946 · recall 0.9188 · F1 0.8522 · ROC AUC 0.9652.
- External 4,000-URL Tranco homepage holdout: **99.55%** URL safe, **99.1%** of 2,000 domains safe, apex safe rate 1.0.
- Raw `model_score` is a forest probability, **not calibrated**, and is never the final website verdict.
- Documented limitation: R2 was validated on homepage-form URLs; ordinary deep paths scored HIGH RISK in later probes (e.g. `https://www.google.com/search?q=fraud` → 0.9761). The layered scanner compensates by live inspection and explains every override; it does **not** hide the limitation.

Details and the full experiment history (B, C, D, S1/S2, Stage 7, S3 feasibility):
`docs/MODEL_METHODOLOGY.md`, `docs/TESTING_AND_EVALUATION.md`, `reports/qa_evaluation_2026-11-02.md`.

## Security protections
- Per-hop DNS validation; loopback / private / link-local (incl. cloud-metadata `169.254.169.254`) / CGNAT `100.64/10` / benchmark `198.18/15` / multicast / reserved / broadcast destinations **blocked before any connection**, including IPv4-mapped IPv6 addresses.
- Connections pinned to validated IPs with the real hostname kept for TLS SNI + Host (DNS-rebinding resistant); redirects revalidated at every hop; HTTPS→HTTP downgrades recorded; embedded credentials rejected; `Location` targets must be web schemes.
- TLS verified with the default strict context; a TLS failure is a connection failure, never proof of malware.
- Strict timeouts (6 s connect / 12 s read / 25 s total budget) and a hard 2 MB response cap; body-read timeouts are honest partial-content results.
- Input validation: non-web schemes, IP-host quirks, invalid ports and hostnames rejected before any network activity.

Full details and what the scanner cannot guarantee: `docs/SECURITY_AND_LIMITATIONS.md`.

## Known limitations (do not overstate readiness)
- All published accuracy figures are **synthetic/controlled-suite results**; no live corpus recall/FP field evaluation exists.
- Brand/impersonation detection is title-driven; JS-only SPA shells often yield no usable content (thin pages stay cautious, e.g. model-flagged shells remain SUSPICIOUS).
- Punycode/IDN domains are always flagged SUSPICIOUS — internationalized-but-legitimate domains receive cautious verdicts by design.
- Known legit deep URLs can still arrive SUSPICIOUS when content can't refute the string model.
- Threat-intelligence feeds are **not configured** (engine honestly reports `available: false`; Safe Browsing/URLhaus are future integration points).
- Inspection uses one browser-like request: cookie/JS-dependent checks (2FA walls, bot walls) are out of scope.

## Future improvements
- Genuine threat-intel integration (Google Safe Browsing / URLhaus) into one engine signal.
- Live phishing-corpus field evaluation (FP/recall on real URLs), kept separate from synthetic gates.
- Public-suffix-aware hostname grouping for future (re)training.
- Complete the S3 deep-URL dataset effort (promoted-report criteria in `model/stage7_study/`, probe verdict INCONCLUSIVE in `model/experiment_seven/`).
- Calibrated probability output (isotonic/Platt) presented separately from the raw forest score.

## Not a browser security product
FraudShield is a scanning/research tool for URL triage. It is not a replacement for
browser sandboxing, antivirus, or human judgment. It never claims perfect accuracy.
