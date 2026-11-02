# FraudShield — Model Methodology

*Everything below is grounded in `model/model_metadata.json` (the promoted
artifact's own provenance record), the experiment scripts under the project
root, and the artifacts under `model/`. No numbers were invented; where the
repository shows a limitation, it is stated here.*

---

## 1. Dataset and preprocessing

**Primary training data — Kaggle `malicious_phish.csv`** (provenance recorded in
`model/metadata.json` → `provenance`):

| Fact | Value |
|---|---|
| File | `data/malicious_phish.csv`, header `url,type` |
| Original rows | 651,191 (`kaggle_original_rows`) |
| Usable rows after loader validation | 641,084 (`kaggle_usable_rows`) |
| Class composition (recounted from the local file) | benign 428,103 · phishing 94,111 · defacement 96,457 · malware 32,520 |
| Dataset SHA-256 | `d83ce942…74d01` (recorded provenance; loader refuses undocumented data) |
| Train/test split | hostname-**grouped** held-out test: 128,214 rows; train 512,870 rows, minus 8,969 purged sanity rows → **503,901** train rows |
| Split mechanism | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, groups = parsed hostname (*not* registrable domain — no Public Suffix List dependency; parent/child overlaps are reported, not claimed zero) |

Loader guarantees (`backend/dataset_loader.py`, documented in `data/README.md`):
text labels mapped, unknown labels rejected/reported, exact duplicates
collapsed, conflicting labels per URL reported and excluded (leakage
protection), and `data/test_urls.csv` **refused** as it is a development
fixture, not training data.

**Legitimate homepage representations — Tranco top-1M** (sampling frame, not observed browser URLs):

| Fact | Value |
|---|---|
| File | `data/tranco/top-1m.csv`, 1,000,000 rows, SHA-256 recorded |
| Selection | deterministic salt `fraudshield-c-v1`, rank limit 50,000 |
| Purpose | generated `https://<domain>` (+ `https://www.<domain>` subset) representations so benign training/holdout rows exist in browser form |
| Train additions | 12,905 rows (10,000 domains) |
| Holdout | **4,000 rows / 2,000 domains, never used for fitting, threshold selection, feature selection or tuning** |
| Exclusions | every Tranco domain overlapping any Kaggle hostname (exact/parent/child) was excluded before selection |

**Final Experiment-C training population:** 503,901 + 12,905 =
**516,806 rows** (benign 348,404 · malicious 168,402).

## 2. Feature extraction (14 lexical features)

Defined once, in `backend/feature_extractor.py` → `FEATURE_NAMES`, and used
identically by training and prediction (order enforced, `n_features_in_`=14):

```
url_length  hostname_length  num_dots  num_hyphens  num_slashes  num_digits
num_special_chars  num_query_marks  num_equal_signs  has_at_symbol
uses_https  has_ip_host  num_subdomains  suspicious_keyword_count
```

- `suspicious_keyword_count` — case-insensitive substring count of 15 words:
  login, signin, verify, account, update, secure, password, bank, confirm,
  payment, security, wallet, authentication, bonus, prize (the module itself
  notes the FP potential, e.g. "accountant" contains "account").
- `num_special_chars` deliberately excludes characters counted by their own
  features; unparseable/non-web inputs yield the all-zero vector.
- The module carries its own contract self-checks (order/duplicates/HTTPS
  semantics/parse-fallbacks).

Text-only by design: **no network requests during training or prediction.**

## 3. Training approach

| Item | Value (from `model_metadata.json`) |
|---|---|
| Classifier | `RandomForestClassifier` |
| Hyperparameters | `n_estimators=100, max_depth=24, min_samples_leaf=5, max_features=sqrt, class_weight=balanced, random_state=42, n_jobs=-1` |
| Training time | 40.47 s (Python 3.14.10, scikit-learn 1.9.1) |
| Artifact size | 30,741,044 bytes (29.32 MB) |

**Leakage audit (`leakage_audit`):** zero exact train/test hostname overlap;
zero derived/Tranco/sanity overlaps; parent/child overlaps (2,979 / 2,858 pairs)
explicitly reported because no Public Suffix List is used.

**Pre-training gate (`pre_training_gate.passed = true`)** protects against the
HTTPS-label-proxy trap: `uses_https` univariate AUC 0.511 (≈ no signal),
HTTPS benign share 0.573 (in-gate 0.40–0.60), Cramér's V not worse than
baseline; the note explains why `num_dots/num_subdomains/num_slashes` retain
AUC (Kaggle malicious rows are genuinely deep/subdomained — a data property).

## 4. The R2 model (what actually got promoted)

- **Provenance:** winner variant `r2` of **Experiment C**; promoted in
  **Stage 5** by copy+verify only (`promotion.retrained = false`).
- Artifact SHA-256:
  **`d74190913f656a4260578e4dfa1b4b8d17bc4b0681e1215d18974713876a1bf8`**
  (unchanged through Stage 5 → layered-scanner phases → documentation phase;
  verified from disk repeatedly, most recently this documentation phase).
- Runtime: loaded by `backend/predictor.py` (singleton, thread-safe, `n_jobs=1`
  after load); feature order from metadata; network access: none.

## 5. Evaluation metrics (measured)

### 5.1 Internal held-out test (128,214 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.8941 |
| Precision | 0.7946 |
| Recall (malicious) | 0.9188 |
| F1 | 0.8522 |
| ROC AUC | 0.9652 |
| Confusion | TN 75,494 · FP 10,119 · FN 3,458 · TP 39,143 |

Per-style malicious recall: bare 0.830, http 0.963, https 0.986. Brand-subset
sanity check (13,056 benign / 1,071 malicious rows): accuracy 0.756, malicious
recall 0.491 — brand keywords make the string-only task harder, honestly reported.

### 5.2 External holdout (4,000 generated Tranco homepage URLs / 2,000 domains)

| Metric | Value |
|---|---|
| URL safe rate | 99.55% |
| Domain safe rate | 99.1% (1,982/2,000) |
| Apex safe rate | 1.0 |
| Score median | 0.0252 |

Worst-scoring holdout domains (real measurement, top: `ffe390afd658c19dcbf707e0597b846d.de` 0.728, `casasbahia-imagens.com.br` 0.699) show that **popularity ≠ automatic safety** and why the engine still inspects live.

### 5.3 Acceptance gates (`acceptance_gates`): 6/6 predeclared checks passed,
no quality warnings. Gates: https malicious recall ≥ 0.90, overall malicious
recall ≥ 0.87, external homepage safe ≥ 0.85, sanity 4/4 synthetic suspicious
URLs flagged, no leakage detected, pre-training distribution gate.

**Important scope qualifier:** these are controlled-suite numbers. They do not
measure live-phishing field recall; see `docs/TESTING_AND_EVALUATION.md`.

## 6. Why raw model probability is NOT the final website verdict

1. **String scores cannot see live pages.** The same lexical counts describe
   `paypa1-verify.xyz/login` and `github.com/team/repo/settings/secrets`; a
   string-only decision answers the wrong question half the time.
2. **The forest score is not calibrated.** `predict_proba` for a RandomForest
   votes towards training votes; the metadata states explicitly: *"predict_proba
   is a model risk score, not a calibrated real-world probability."* Display
   bands (0.35/0.65) are presentation policy, never optimized thresholds.
3. **Measured systematic failure — deep-path URLs.** Stage 5 evidence
   (`known_limitation.evidence_measured_in_stage_5`):
   `https://www.google.com/search?q=fraud` → 0.9761,
   `https://en.wikipedia.org/wiki/Phishing` → 0.9823,
   `https://github.com/microsoft/vscode` → 0.9565,
   `https://www.youtube.com/feed/...`-style pages repeatedly "HIGH RISK".
   Root cause attributed (Stage 5/7 analyses): 93.2% of benign Kaggle rows are
   written bare while 67.3% of malicious rows carry a scheme; only 2,263 benign
   `https+deep` rows existed in training versus 9,956 malicious; `num_slashes`
   has the highest feature importance (0.2618). The model learned the dataset's
   *writing convention*, incl. an ultra-narrow "https + exactly 2 slashes"
   benign cell (`https://example.com` → SAFE, `https://example.com/` → 0.93).
4. **The layered architecture therefore:**
   - preserves the raw score (`url_model.model_score`, `raw_prediction`),
   - lets the *live inspection* decide, and
   - only overrides a flagged score when the real destination fully checks out
     (reachable + TLS + real content) — and prints the override explanation.
   The layered scanner is measured to not produce that FP live:
   `https://github.com/microsoft/vscode` → SAFE; the probe URLs remain honest
   SUSPICIOUS/INSUFFICIENT states where content cannot refute the flag.

## 7. Why previous/beside experiments were not promoted

Every non-promotion is an *acceptance-gate failure against predeclared
criteria*, not a judgment call. (All gates were fixed in the experiment scripts
before running; artifacts and JSON comparison reports are preserved.)

| Experiment | Idea | Result | Why not promoted |
|---|---|---|---|
| **Experiment B** (benign scheme augmentation, `model/model_metadata_augmented.json`) | duplicate training-only bare benign rows as https/www forms | metrics **worse** (B-variant internal test: accuracy 0.8517, F1 0.7949, ROC AUC 0.9279; https malicious recall collapsed to **0.1525** for one candidate) | status recorded `"experimental - not approved for deployment"`; no promotion artifact |
| **Experiment D** (D1/D2, `model/experiment_d/comparison.json`) | derive *both*-scheme browser forms of benign train rows with real path/query text (leakage-safe; both schemes to prevent an HTTPS label proxy) | D1/D2 pass only **3/7** predeclared gates (overall + deep malicious recall fall below thresholds; legit-deep safe rate still < 0.90) | R2 keeps 6/7; the deep-URL fix must not destroy malicious recall — it did |
| **Stage 6 S1** (`model/experiment_six/comparison.json`) | scheme-blind semantic representation (hostname/path/query char & word n-grams + 45 keyword counts, LogisticRegression) | recorded verdict: *"FAIL: S1 adds measurable deep-URL separability but misses at least one predeclared gate; not promotable"* (induces a homepage/`www.`-normalisation collapse) | fails predeclared gate |
| **Stage 6 S2-A** (`experiment_six_s2.py`) | S1 with one `www.`-normalisation removal in the host block | script exists; **no promoted artifacts in the repository** | not promoted (no artifact, no accepted report) |
| **Stage 7** (read-only data redesign study, `model/stage7_study/stage7_report.md`) | diagnose the malicious-HTTPS gate (subtype/attribution/representation-gap analysis) | conclusion: *R2 REMAINS PROMOTED*; produced the S3 data spec + acceptance criteria for a *future* dataset redesign | study only — no training performed |
| **S3 / Experiment seven** (Common Crawl feasibility probe, `model/experiment_seven/feasibility_probe/report.md`) | assess whether Common Crawl index data can satisfy S3's deep-URL dataset criteria | verdict: **INCONCLUSIVE** (host cap destroys 93.54% of rows; query-bearing rate 16.04% vs ≥25% required; 16/22 criteria measured-pass) | no acquisition, no training; "requires targeted acquisition, not proven" |

**Net invariant across the whole history:** the promoted artifact is always the
original R2 file with the SHA-256 above; every rejected experiment remains
archived with its comparison JSON so the decision trail is auditable.

## 8. Current deployment relationship between model and app

`predictor.py` uses R2 exactly as trained (no retrain, no threshold change,
`--self-test` = 31/31 contract checks). The representation-convention
limitation is **handled, not hidden**, by the layered scanner: live inspection
overrides flagged legit deep pages with explanation, the policy layer
structurally validates known-good hosts, and unresolved thin pages remain
SUSPICIOUS/INSUFFICIENT EVIDENCE rather than guessed.
