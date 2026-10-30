# FraudShield — Stage 7: Read-Only Data-Redesign Study

**Status:** Stage 7 analysis complete. No training. No downloads. No artifact modification.
**Date:** 2026-10-30
**Mode:** READ-ONLY
**Promoted model:** R2 (unchanged, hash `d7419091…a1bf8`)
**Scope:** Sections A–H as specified.

---

## 0. Provenance and method

This study is **read-only**. The following were enforced throughout:

- No classifier, vectoriser or model was created, fitted or refitted. No `.fit()` or `.partial_fit()` call anywhere.
- No network, DNS, WHOIS, TLS, browser or external API access. No URL was visited.
- No dataset was downloaded.
- No pandas.
- All URLs processed strictly as text.
- Python 3.14.10, project venv (`numpy 2.5.4`, `scikit-learn 1.9.1`).

Inference on the **already-fitted** artifacts R2 / S1 / S2-A was performed to attribute existing
failures. No threshold was changed, and no score was used to refit anything.

**Two prior-session artifacts were reused rather than recomputed**, so that A/B/C remain
byte-consistent with the earlier run:

| Artifact | Role |
|---|---|
| `analysis/stage7/stage7_audit.py` | Imported (never modified). Supplies the corpus loader, URL parser, scheme/structure classifiers. |
| `analysis/stage7/stage7_stats.json` | Sections A, B, C — computed by the prior session, verified against the project's own artifacts. |

Everything in sections D–H and the addendum is new work in this session.

### Split reproduction check

The fixed split was reproduced exactly, confirming the analysis operates on the same data the
models were evaluated on:

| Quantity | Reproduced | R2 metadata | Match |
|---|---|---|---|
| Train after brand purge | 503,901 | 503,901 | ✅ |
| Test rows | 128,214 | 128,214 | ✅ |
| Usable corpus rows | 641,084 | 641,084 | ✅ |

---

## Executive conclusion

**The malicious-HTTPS failure is not a malicious-side problem. It is a benign-side coverage problem
compounded by a measurement artifact, and the gate that has been driving three rejected experiments
is measuring almost nothing.**

Four findings carry the study:

1. **The malicious-HTTPS gate is 42% one hostname.** Of 2,321 malicious HTTPS test rows, 976 (42.05%)
   are `pastebin.com`. Within malware specifically, 1,045 HTTPS rows come from **51 hosts**, and
   `pastebin.com` alone supplies **93.40%** — one host is ≥80% of the entire malware-HTTPS
   denominator. The "malware HTTPS recall 0.058" that rejected S1 and S2-A is a measurement of how
   one paste site's URLs score. It is not a generalisable estimate of malware-over-HTTPS detection.

2. **Defacement contributes exactly zero rows to the HTTPS gate and is 100% HTTP.** Verified across
   the full corpus and the test split: 0 HTTPS rows, 0 bare rows, 0 homepage rows. Defacement is
   therefore invisible to the HTTPS gate while dominating the *overall* malicious recall and
   *malicious deep recall* gates (19,268 of 37,601 deep malicious rows, 51.2%).

3. **The benign training corpus contains zero HTTPS homepages and 0.41% HTTPS deep rows**, while the
   legitimate-deep suite is scored at **93.01% HTTPS** after the `https://` rewrite. That is a
   ~225× representation gap. The models are being asked to recognise a benign form they have
   essentially never seen.

4. **R2's headline "legitimate deep safe 0.112" is an artifact of the evaluation harness, not a model
   property.** Scored in raw stored form R2 achieves **0.867752**; the harness rewrites 70,214 bare
   rows to `https://`, which perturbs exactly the features R2 weights most (`num_slashes`,
   `num_special_chars`, `num_dots`, `url_length`, `uses_https`). The gate that R2 fails is partly
   measuring the harness.

The scientific consequence: **three experiments have now been rejected against a gate whose
denominator is one paste site, while the actual defect — benign training distribution — has never
been addressed.** S2 established that deleting `www.` was insufficient. The evidence says the next
move is not another feature deletion and not another malicious source. It is benign data.

---

## A. Corpus audit by subtype

Corpus: `data/malicious_phish.csv`, SHA-256 `d83ce942…4d01`. Usable rows **641,084** after exact-URL
dedup and cross-label conflict removal. Full corpus (train + test), not the test split.

### A.1 Scheme

| Subtype | Rows | Bare | HTTP | HTTPS |
|---|---:|---:|---:|---:|
| benign | 428,074 | 392,696 (91.74%) | 33,428 (7.81%) | 1,950 (0.46%) |
| phishing | 94,057 | 69,224 (73.60%) | 17,869 (19.00%) | 6,964 (7.40%) |
| malware | 23,645 | 1,210 (5.12%) | 15,712 (66.45%) | 6,723 (28.43%) |
| defacement | 95,308 | 0 (0.00%) | 95,308 (100.00%) | **0 (0.00%)** |

Scheme is not a neutral field. It is a near-perfect subtype fingerprint: defacement is 100% HTTP,
malware is 66% HTTP / 28% HTTPS, benign is 92% bare. A scheme-blind model is being asked to
generalise across a variable that is almost deterministic given the label.

### A.2 Structure

| Subtype | Homepage | Path | Query | Path+query |
|---|---:|---:|---:|---:|
| benign | 47,638 (11.13%) | 307,698 (71.88%) | 4,574 (1.07%) | 68,164 (15.92%) |
| phishing | 23,547 (25.03%) | 62,370 (66.31%) | 338 (0.36%) | 7,802 (8.30%) |
| malware | 1,011 (4.28%) | 21,265 (89.93%) | 20 (0.08%) | 1,349 (5.71%) |
| defacement | **0 (0.00%)** | 41,348 (43.38%) | 0 (0.00%) | 53,960 (56.62%) |

Defacement has **no homepage rows at all**. Every defacement URL is a deep URL. This matters for the
"malicious deep recall" gate, which defacement dominates.

### A.3 Leading `www.`

| Subtype | `www.` rows | Share |
|---|---:|---:|
| benign | 13,228 | 3.09% |
| phishing | 36,452 | 38.76% |
| malware | 841 | 3.56% |
| defacement | 66,318 | **69.58%** |

This reproduces S1's discovered dataset-writing artifact and shows it is broader than reported:
defacement is the most `www.`-heavy class in the corpus at 69.58%, against benign at 3.09%.

### A.4 Hostname label count

| Subtype | 2 labels | 3 labels | 3+ labels | 4+ labels |
|---|---:|---:|---:|---:|
| benign | 301,617 | 107,927 | 29.54% | 4.33% |
| phishing | 26,282 | 53,788 | 72.01% | 14.82% |
| malware | 6,172 | 5,250 | 73.90% | **51.69%** |
| defacement | 25,953 | 55,689 | 72.77% | 14.34% |

Malware's 4+ label share (51.69%) is driven by IP-literal hosts, whose dotted quads inflate the count.

### A.5 TLD / suffix diversity

| Subtype | Distinct last-label TLDs | Distinct last-2 suffixes | Top TLD |
|---|---:|---:|---|
| benign | 220 | 456 | `com` 72.27% |
| phishing | 531 | 769 | `com` 52.51% |
| malware | 396 | 432 | `com` 25.42%, `jp` 12.32% |
| defacement | **74** | **109** | `com` 30.59%, `de` 11.30% |

Defacement is TLD-poor (74 TLDs vs benign's 220) and geographically European-weighted
(`de`, `nl`, `br`, `it`, `au`, `uk`), consistent with a scrape of a specific defacement archive
rather than a global sample.

### A.6 Hostname length

| Subtype | p10 | Median | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| benign | 9 | 15 | 24 | 45 | 248 |
| phishing | 12 | 17 | 28 | 70 | 236 |
| malware | 12 | 14 | 20 | 44 | 73 |
| defacement | 12 | 18 | 27 | 35 | 49 |

### A.7 Path depth

| Subtype | Median | p90 | p99 | Max |
|---|---:|---:|---:|---:|
| benign | 2 | 4 | 7 | 34 |
| phishing | 1 | 4 | 7 | 38 |
| malware | 2 | 2 | 6 | 11 |
| defacement | 1 | 4 | 6 | 12 |

### A.8 Query prevalence

| Subtype | Rows with query | Share | Median params |
|---|---:|---:|---:|
| benign | 72,738 | 16.99% | 1 |
| phishing | 8,140 | 8.65% | 1 |
| malware | 1,369 | 5.79% | 3 |
| defacement | 53,960 | **56.62%** | **4** |

Defacement is the most query-heavy class in the corpus by a wide margin (56.62% vs benign 16.99%),
with a median of 4 parameters. This is a CMS-signature effect, examined in section E.

### A.9 Measurable lexical characteristics (no fitting)

| Subtype | Suspicious-keyword hit rate | IP-literal hosts | Distinct hosts | Distinct approx domains | Rows/host |
|---|---:|---:|---:|---:|---:|
| benign | 5.90% | 26 (0.006%) | 134,065 | 104,520 | 3.19 |
| phishing | 11.07% | 329 (0.350%) | 55,810 | 45,548 | 1.69 |
| malware | 1.03% | **11,769 (49.77%)** | 7,364 | 6,887 | 3.21 |
| defacement | 0.88% | 0 (0.000%) | 2,122 | 2,121 | **44.91** |

Two structural facts stand out:

- **Malware is half IP-literal.** 49.77% of malware rows use a raw IP as host. A lexical model can
  separate these trivially via `has_ip_host`, which is why malware scores so well overall — and why
  its *HTTPS* subset (which is not IP-based) collapses.
- **Defacement averages 44.91 rows per host** across only 2,121 domains. This is a small number of
  compromised sites sampled exhaustively, not a broad URL population.

### A.10 Duplicate / cross-form relationships

| Subtype | Equivalent URL groups | Groups w/ multiple schemes | `www.` normalization collisions | Rows in collision groups | `www.` hosts without an apex row |
|---|---:|---:|---:|---:|---:|
| benign | 427,112 | 313 (0.07%) | 712 | 1,424 | 6,667 |
| phishing | 93,042 | 579 (0.62%) | 697 | 1,394 | 18,460 |
| malware | 22,880 | 44 (0.19%) | 14 | 28 | 298 |
| defacement | 95,307 | **0** | **0** | 0 | 1,436 |

- Defacement has **zero** multi-scheme groups and **zero** apex/www collisions: each defacement URL
  exists in exactly one written form. There is no cross-form redundancy to exploit.
- Phishing has the highest multi-scheme rate (0.62%), meaning the same page appears under both
  `http://` and `https://` — a real source-convention artifact.
- Benign's 6,667 `www.` hosts with no apex row confirm that apex/www pairing is largely absent in the
  benign corpus, which is why S2's normalization had so little benign material to work with.

### A.11 Bare / HTTP / HTTPS equivalent representations

Only 313 benign groups (0.07%) and 579 phishing groups (0.62%) appear in more than one scheme. The
corpus is overwhelmingly **single-representation per URL**: a given page is stored once, in whatever
scheme the source happened to write. This is the root of the scheme confound — there is almost no
material teaching a model that `example.com/x`, `http://example.com/x` and `https://example.com/x`
are the same thing.

---

## B. Malicious HTTPS attribution

Malicious HTTPS test rows: **2,321**.

### B.1 Composition

| Subtype | HTTPS rows | Share of malicious HTTPS | R2 recall | S1 recall | S2-A recall |
|---|---:|---:|---:|---:|---:|
| phishing | 1,276 | 54.98% | 0.974922 | 0.775078 | 0.789969 |
| malware | 1,045 | 45.02% | 0.999043 | **0.060287** | **0.058373** |
| defacement | **0** | 0.00% | n/a | n/a | n/a |
| **total** | **2,321** | 100% | **0.985782** | **0.453253** | **0.460577** |

### B.2 Defacement HTTPS: explicitly verified zero

Confirmed independently on both the full corpus and the test split:

- Full corpus defacement rows: 95,308 — HTTPS rows: **0**
- Test-split defacement rows: 19,268 — HTTPS rows: **0**
- `defacement_https_zero_confirmed.is_zero = true`

**Defacement contributes exactly zero rows to the malicious-HTTPS gate.** It cannot be the cause of
the HTTPS failure, and it is not affected by any HTTPS-side fix.

### B.3 Contribution to the global gate

Leave-one-subtype-out on the HTTPS gate:

| Model | Without phishing | Without malware | Without defacement |
|---|---:|---:|---:|
| R2 | 0.999043 | 0.974922 | 0.985782 |
| S1 | **0.060287** | 0.775078 | 0.453253 |
| S2-A | **0.058373** | 0.789969 | 0.460577 |

For S1 and S2-A the gate is **dominated by malware**: remove malware and recall rises from 0.453 to
0.775. Remove phishing instead and it collapses to 0.060. Malware's 1,045 rows are what drag the gate
down, and they are 45% of the denominator.

### B.4 Effective sample size — the decisive measurement

Rows are not independent. Measured clustering of the malicious-HTTPS test rows:

| Group | Rows | Distinct hosts | Rows/host | Hosts for 80% of rows | Largest host share |
|---|---:|---:|---:|---:|---:|
| all malicious HTTPS | 2,321 | 816 | 2.84 | 352 | `pastebin.com` 42.05% |
| phishing | 1,276 | 765 | 1.67 | 510 | `fareast.qa` 2.04% |
| malware | 1,045 | **51** | **20.49** | **1** | `pastebin.com` **93.40%** |

**One hostname, `pastebin.com`, supplies 976 of 1,045 malware-HTTPS rows (93.40%) and 976 of 2,321
all-malicious-HTTPS rows (42.05%).**

The malware-HTTPS recall of 0.058 is therefore not a measurement of malware-over-HTTPS detection. It
is, to a first approximation, a measurement of **how a single paste site's URLs score**. S1's scores
on those rows are perfectly flat (p10 = median = p90 = 0.300623), which is the signature of a model
receiving no usable evidence — a paste URL is a bare opaque identifier with no lexical content.

### B.5 Verdict: subtype-driven, formatting-driven, or both?

**Both, and they are separable — but the dominant term is source-convention, not subtype.**

Evidence for **formatting/source-convention driven**:

- Malware HTTPS rows come from 51 hosts, 93.40% from one. That is a source artifact, not a
  population.
- Defacement is 100% HTTP / 0% bare / 0% HTTPS — pure convention.
- Malware is 66% HTTP / 28% HTTPS while benign is 92% bare: the *writing convention* differs by
  label, so scheme leaks label information (residual scheme probe AUC **0.880385**).
- Phishing HTTPS recall is *higher* than phishing bare recall for every model (R2 0.975 vs 0.829;
  S1 0.775 vs 0.750; S2-A 0.790 vs 0.687), so HTTPS is not intrinsically harder for phishing.

Evidence for **subtype driven**:

- Malware-over-HTTPS genuinely differs from malware-over-HTTP: the HTTP subset is 49.77% IP-literal
  and scores 0.924–0.953 recall, while the HTTPS subset is domain-based with no lexical signal.
- Phishing HTTPS remains the largest single block (54.98%) and is the only subtype where all three
  models do reasonably well.

**Conclusion:** the malicious-HTTPS gate is failing because (a) its denominator is dominated by one
paste hostname whose URLs carry no lexical evidence, and (b) the corpus teaches scheme as a label
proxy. It is **not** primarily evidence that the models cannot detect malicious HTTPS URLs.

---

## C. Benign coverage audit

Purpose: establish exactly what benign examples the current training corpus is missing.

### C.1 Measured representation

| Category | Kaggle benign train (335,499) | Total benign train incl. Tranco (348,404) | Legit-deep suite (76,039) | Legit-deep in **evaluated** form |
|---|---:|---:|---:|---:|
| HTTPS homepage | **0 (0.00%)** | 12,905 (3.70%) | 0 (0.00%) | 0 |
| HTTPS deep, any | 1,439 (0.43%) | 1,439 (0.41%) | 509 (0.67%) | **70,723 (93.01%)** |
| HTTPS deep path | 615 (0.18%) | 615 (0.18%) | 118 (0.16%) | — |
| HTTPS deep query | 53 (0.02%) | 53 (0.02%) | 0 (0.00%) | — |
| HTTPS deep path+query | 771 (0.23%) | 771 (0.22%) | 391 (0.51%) | — |
| HTTPS `www.` | 94 (0.03%) | 2,999 (0.86%) | 0 (0.00%) | — |
| HTTPS non-`www.` | 1,345 (0.40%) | 11,345 (3.26%) | 509 (0.67%) | — |
| HTTP any | 27,863 (8.30%) | 27,863 (8.00%) | 5,316 (6.99%) | — |
| Bare any | 306,197 (91.27%) | 306,197 (87.89%) | 70,214 (92.34%) | — |
| Deep any | 297,449 (88.66%) | 297,449 (85.37%) | 76,039 (100%) | — |
| Homepage any | 38,050 (11.34%) | 50,955 (14.63%) | 0 (0.00%) | — |

### C.2 The gap, quantified

The legitimate-deep suite is **scored** after `browser_form()`, which prepends `https://` to bare
rows. 70,214 of 76,039 rows are rewritten. So the suite the models are graded on is **93.01% HTTPS
deep**, while the training corpus they learned from is **0.41% HTTPS deep**.

**Representation gap: ~225×.**

| Comparison | Train share | Eval share | Ratio |
|---|---:|---:|---:|
| HTTPS deep benign | 0.4130% | 93.01% | **225.2×** |
| HTTPS homepage benign | 3.70% | 0.00% | — (suite has no homepages) |
| Bare benign | 87.89% | 7.69% (post-rewrite) | 0.09× |

### C.3 Hostname / TLD diversity

| Block | Distinct hosts | Distinct approx domains | Distinct TLDs | Rows/host |
|---|---:|---:|---:|---:|
| Kaggle benign train | 107,073 | 84,420 | 419 | 3.13 |
| Tranco generated train | 12,905 | 9,980 | 465 | 1.00 |
| Total benign train | 119,978 | 94,393 | 672 | 2.90 |
| Legit-deep suite | 18,629 | 16,084 | 239 | 4.08 |
| Tranco holdout | 4,000 | 1,998 | 228 | 2.00 |

TLD diversity in training (672) exceeds the evaluation suites (239 / 228), so TLD coverage is *not*
the binding constraint. Hostname diversity is adequate in count but wrong in **kind**: the training
hosts are overwhelmingly bare, homepage-adjacent, and `www.`-free.

### C.4 What is actually missing

Stated plainly, the benign training corpus is missing:

1. **HTTPS homepages entirely** (0 rows in Kaggle; the 12,905 Tranco rows are synthetic
   `https://` + apex/`www.` constructions, not observed URLs).
2. **HTTPS deep URLs at any meaningful rate** (0.41% vs 93.01% required).
3. **HTTPS query URLs** (53 rows total in Kaggle benign training).
4. **`www.` + HTTPS combinations** (94 rows in Kaggle, 0.03%).
5. **Real observed deep URLs under HTTPS** — the dominant real-world benign form.

The single largest gap is not diversity of hosts or TLDs. It is that **the corpus contains almost no
real HTTPS deep URLs**, which is precisely the form the evaluation suite is built from.

---

## D. Benign data redesign requirements

Each requirement below is labelled **MEASURED** (derived from the numbers above) or
**DESIGN RECOMMENDATION** (a judgement call, with its reasoning shown). Precision is not claimed
beyond what the data supports.

### D.1 Usable sample size — DESIGN RECOMMENDATION, anchored on measurement

Measured baseline: 348,404 benign training rows, of which 1,439 are HTTPS-deep (0.41%).

Scenarios to raise the HTTPS-deep share, computed arithmetically on the measured population:

| Target HTTPS-deep share | Additional HTTPS-deep rows needed | Resulting benign train rows |
|---|---:|---:|
| 5% | 16,823 | 365,226 |
| 10% | 37,113 | 385,516 |
| 20% | 85,303 | 433,706 |
| 30% | 147,261 | 495,664 |

**Recommended range: 40,000–90,000 usable new benign HTTPS-deep rows** (targeting a 10–20% share).

Reasoning: 5% is too thin to move a 93%-HTTPS evaluation suite; 30% would require 147k rows and
would swamp the malicious class (currently 166,546 rows in S2 training), pushing the benign:malicious
ratio far from the corpus's own 428,074:213,010. 10–20% is the range that closes most of the gap
without unbalancing the task. This is a recommendation, not a measured requirement — the data cannot
tell us the exact optimum without training, which is out of scope here.

### D.2 Minimum HTTPS coverage — MEASURED requirement

- **100% of new benign rows must carry an explicit `https://` scheme.** MEASURED: the corpus's
  central defect is that scheme is a label proxy (residual scheme probe AUC 0.880385) and that
  benign training is 87.89% bare while the evaluation suite is 93.01% HTTPS.
- **Zero rows may be scheme-synthesised from bare input.** MEASURED: Tranco's 12,905 rows are
  `https://` + apex/`www.` constructions with no path; they raised the HTTPS-homepage share to 3.70%
  but left the HTTPS-deep share at 0.41%, and S2's Tranco safety still landed at 0.5955. Synthetic
  scheme prefixes do not substitute for observed URLs.

### D.3 Homepage / deep balance — DESIGN RECOMMENDATION

- **≥90% deep (path and/or query), ≤10% homepage.** Reasoning: the evaluation suite is 100% deep and
  0% homepage, and the corpus already has 50,955 benign homepage rows (14.63%). Adding more
  homepages addresses a gap that does not exist. Note this deliberately inverts the natural
  web-crawl distribution, which is homepage-heavy — the source must be filtered, not taken raw.

### D.4 Path / query diversity — MEASURED + DESIGN

- **MEASURED:** benign training has only 72,738 query rows (16.99%) and 53 HTTPS-query rows. The
  evaluation suite's query and path+query blocks (864 + 12,477 rows) are therefore thinly supported.
- **DESIGN:** require **≥25% of new rows to contain a query string**, and **≥1,000 distinct query
  parameter names**. Reasoning: the corpus's query vocabulary is narrow and the models have never
  seen HTTPS+query at scale.
- **DESIGN:** require **median path depth ≥2** and **p90 ≥4**, matching the measured benign corpus
  (median 2, p90 4) so the new data does not shift the benign path-depth distribution.

### D.5 Apex / `www.` balance — MEASURED requirement

- **Both forms must be present for a substantial share of domains, and both must be HTTPS.**
  MEASURED: benign training has 2,999 HTTPS-`www.` rows (0.86%) and 6,667 `www.` hosts with no apex
  row; S2's apex-vs-`www.` score delta is exactly 0 over 2,000 domains, which is correct behaviour
  but was achieved by normalisation, not by data.
- **DESIGN:** require **≥20% of new domains to appear in both apex and `www.` form**, so the model
  can learn the equivalence from data rather than from a hard-coded normalisation rule.

### D.6 Hostname diversity — MEASURED requirement

- **≥10,000 distinct registrable domains**, and **≤5 rows per host on average**. MEASURED: the
  corpus's worst clustering is defacement at 44.91 rows/host; benign training sits at 2.90. A new
  source that averages >5 rows/host would import the same defect.
- **≥150 distinct TLDs.** MEASURED: benign training has 672, evaluation suites 239/228 — so 150 is a
  floor that comfortably covers the evaluated distribution without demanding the full 672.

### D.7 Duplicate / cross-form controls — MEASURED requirement

- **Exact-URL dedup, then normalised-host+path+query dedup.** MEASURED: the corpus required removal
  of 10,066 exact duplicate rows and 12 conflicting rows; S2 required purging 16,172 rows for
  normalisation-induced leakage alone.
- **Cross-form collapse:** where the same path exists under both `http://` and `https://`, keep the
  HTTPS form only. MEASURED: 313 benign and 579 phishing groups currently exist in multiple schemes,
  which is exactly the redundancy that teaches scheme as a label proxy.

### D.8 Provenance requirements — DESIGN RECOMMENDATION

- A documented collection method, collection date, and a stable public identifier (version or
  snapshot id).
- **No rows whose only provenance is "scraped from a search engine result page."** Reasoning: the
  `www.` artifact (defacement 69.58%, phishing 38.76%, benign 3.09%) is a writing-convention
  fingerprint of how a source emitted URLs, and SERP-derived data is the usual cause.
- Rows must be **observed URLs**, not constructed ones. MEASURED justification: Tranco's constructed
  rows moved HTTPS-homepage share but not HTTPS-deep share, and did not fix Tranco safety.

### D.9 Licensing requirements — DESIGN RECOMMENDATION

- A licence that permits research use and redistribution of derived features, or at minimum
  local use without redistribution.
- No licence that forbids use for security research or model training.
- Recorded per-source, with the licence text or URL stored alongside the data manifest.

### D.10 Timestamp / version requirements — DESIGN RECOMMENDATION

- A single collection window, recorded, ideally ≤12 months wide.
- **Version pinning by content hash**, matching this project's existing practice
  (`data/tranco/tranco_manifest.json`).
- Reasoning: the corpus's own defacement block is a historical archive (European TLD weighting, 74
  TLDs), and mixing eras would confound "benign" with "old".

### D.11 Leakage requirements — MEASURED requirement

- **Zero hostname overlap with the fixed test split**, checked both raw and normalised. MEASURED:
  S2's preflight asserts `raw_train_test_hostname_overlap = 0` and
  `normalized_train_test_hostname_overlap = 0`; the normalised check is mandatory because
  `www.`-normalisation created 16,172 collisions in S2.
- **Zero overlap with the Tranco holdout domains** (`tranco_holdout_domains_in_train = 0`).
- **Zero overlap with the legitimate-deep suite's 18,629 hostnames.**
- **Zero overlap with the sanity brand list.**
- Any new benign source must be split **by hostname group**, never by row.

---

## E. Malicious-side / defacement study

### E.1 What the defacement label actually represents

Measured profile of the 95,308 defacement rows:

| Property | Value |
|---|---|
| Scheme | 100% `http://`, 0% bare, 0% HTTPS |
| Structure | 0% homepage, 43.38% path, 56.62% path+query |
| Leading `www.` | 69.58% |
| Distinct hosts / domains | 2,122 / 2,121 |
| Rows per host | **44.91** |
| IP-literal hosts | 0 |
| Distinct TLDs | 74 |
| Suspicious-keyword hit rate | 0.88% (lowest of all subtypes) |
| Median query params | 4 |

The dominant path tokens are **`index`, `php`, `option`, `id`, `itemid`, `view`, `html`, `content`,
`article`, `component`, `catid`, `tmpl`, `category`, `page`, `layout`, `print`, `default`**.

That is the **Joomla CMS URL grammar** (`index.php?option=com_content&view=article&id=…&Itemid=…`).

**The defacement label in this corpus represents "this page was defaced", not "this URL is lexically
malicious."** The URLs are the ordinary, well-formed URLs of legitimate CMS-driven sites that were
subsequently compromised. The maliciousness lives in the page content and the compromise event, not
in the URL string.

### E.2 Does URL text contain measurable discriminative information?

Holding the scheme confound constant (HTTP + deep only), defacement (6,000 sampled) vs benign
HTTP-deep (5,316):

| Non-fitted rule | Defacement rate | Benign rate | Gap |
|---|---:|---:|---:|
| Has query string | 0.5632 | 0.2571 | **+0.3060** |
| Path ends in `.php/.html/.htm/.asp/.jsp` | 0.7458 | 0.1606 | **+0.5852** |
| Path depth ≥3 | 0.2452 | 0.6934 | −0.4482 |
| Host contains a digit | 0.0152 | 0.0587 | −0.0435 |
| Suspicious keyword present | 0.0088 | 0.0237 | −0.0149 |

Path-token Jaccard (defacement vs benign, HTTP+deep): **0.0768** — near-disjoint vocabularies.

Model separability with scheme held constant:

| Model | AUC (defacement vs benign, HTTP+deep) | Defacement recall @0.5 | Benign FPR @0.5 |
|---|---:|---:|---:|
| R2 | 0.975893 | 0.9785 | 0.2022 |
| S1 | 0.962902 | 0.9152 | 0.1603 |
| S2-A | 0.926845 | 0.8697 | 0.1964 |

**Yes, there is real measurable signal** — but read it carefully. The strongest discriminators are
*CMS URL grammar* (`.php` extension, `option=`/`view=` query structure) and *path depth*, not
malicious intent markers. Suspicious keywords are **less** common in defacement than in benign
(0.88% vs 2.37%). A model that learns "Joomla URL shape ⇒ malicious" has learned to recognise a CMS,
not an attack.

### E.3 How much apparent predictability comes from source/writing convention?

A large share. Three conventions stack:

1. **Scheme:** 100% HTTP vs benign's 7.81%. Scheme alone separates most of the class.
2. **`www.`:** 69.58% vs benign's 3.09%.
3. **Query structure:** 56.62% with a median of 4 params vs benign's 16.99% with a median of 1.

None of these three is a property of malicious intent. They are properties of **how the defacement
archive was written and scraped**. The residual scheme probe AUC of 0.880385 confirms the models can
reconstruct the written scheme from the features, so this convention is available to them as a
shortcut.

### E.4 Is treating all defacement URLs as lexically malicious label noise?

**Yes — it is an ill-posed target, for a specific and demonstrable reason.**

The label asserts "this URL is malicious." The evidence says the URL is an ordinary CMS URL on a
site that was compromised. Two consequences follow:

- **Within-class inconsistency:** the same Joomla URL pattern appears as *benign* elsewhere in the
  corpus (benign HTTP-deep rows share the `.php`/query grammar at 16.06% and 25.71% respectively).
  The model is asked to call one malicious and the other benign on the basis of a distinction the URL
  string does not carry.
- **Inflated apparent capability:** defacement is 19,268 of 37,601 deep malicious rows (51.2%) and
  drives "malicious deep recall" (S2-A 0.875 on defacement vs 0.686 phishing, 0.711 malware). A
  large part of that gate is being satisfied by CMS-shape recognition.

This is **label noise in the technical sense**: the target is not a function of the input.

### E.5 Defensible future options (defacement is NOT removed)

Per instruction, defacement is retained. Four options, with consequences:

**Option 1 — Retain as a difficult/noisy malicious class (status quo).**
*Consequence:* preserves historical comparability and the 7 gates exactly. Keeps the label-noise
problem and keeps CMS-shape recognition contributing to recall. Zero migration cost.
*Assessment:* acceptable as a baseline, not as a fix.

**Option 2 — Retain in training, report separately in evaluation.**
*Consequence:* the model still sees defacement, but gates are reported per subtype so CMS-shape
recognition cannot silently satisfy a global recall gate. Requires new diagnostics (already
recommended in G), no gate changes. **Lowest-risk improvement.**

**Option 3 — Build a separate task.**
*Consequence:* "was this site compromised" is a site-reputation question, not a URL-lexical one.
Would need host-level features and temporal data, and is out of scope for a text-only URL classifier.
*Assessment:* scientifically correct, but a different project.

**Option 4 — Obtain malicious sources whose labels correspond to URL behaviour.**
*Consequence:* phishing and malware URLs are malicious *as URLs* (credential-harvesting paths,
payload-delivery paths). Adding such sources would improve the malicious class's construct validity.
*Assessment:* **supported by the evidence for phishing/malware, not for defacement.**

**Recommendation: Option 2 now, Option 4 for phishing/malware only if a suitable source is found
(section F). Do not pursue Option 3 in this project.**

---

## F. External dataset research plan

**Nothing was downloaded. Nothing was chosen.** This section specifies source *types* and their
acceptance criteria, and lists public candidates only as candidates.

### F.1 Benign HTTPS homepages

| Field | Specification |
|---|---|
| Purpose | Close the 0-row HTTPS-homepage gap in Kaggle benign training; support the Tranco-safety gate. |
| Required fields | Full URL with explicit scheme; observation timestamp; source identifier. |
| Label semantics | Benign by construction (top-ranked legitimate site) — not by classifier. |
| Desired size | 10,000–25,000 rows, ≥8,000 distinct domains. |
| Provenance / licence | Public ranking list with a citable snapshot; permissive or research-permitting licence. |
| Timestamp / version | Pinned snapshot id + content hash. |
| Leakage risks | **High** — a popularity list overlaps any popularity-based holdout. Must be split by domain against the Tranco holdout. |
| Dedup | Exact URL, then registrable domain. |
| Text-only | Yes — ranking list is plain text. |
| Candidates (not chosen) | Tranco (already used), Cisco Umbrella top-1M, Majestic Million. |

**Caution, measured:** Tranco is already in the training set as 12,905 synthetic rows and Tranco
safety is still 0.5955. A second popularity list of the same kind is unlikely to fix the gate. This
source type is **lowest priority** of the five.

### F.2 Benign real deep URLs

| Field | Specification |
|---|---|
| Purpose | **Primary gap.** Close the 0.41% → 93.01% HTTPS-deep representation gap. |
| Required fields | Full observed URL with explicit `https://` scheme; path and/or query present; timestamp. |
| Label semantics | Benign by construction (from a curated legitimate corpus), not by inference. |
| Desired size | **40,000–90,000 rows** (section D.1), ≥10,000 distinct domains, ≤5 rows/host. |
| Provenance / licence | Documented collection method and date; licence permitting research use. |
| Timestamp / version | Single collection window ≤12 months; pinned by hash. |
| Leakage risks | **High** — must not overlap the 18,629 legitimate-deep suite hostnames or the fixed test split. Group-split by hostname. |
| Dedup | Exact URL → normalised host+path+query → cross-form collapse to HTTPS. |
| Text-only | Yes. |
| Candidates (not chosen) | Common Crawl URL index (filtered to high-reputation domains), public bookmark/link corpora, DMOZ-derived URL sets, Wikipedia external-link dumps. |

**This is the highest-value source type in the study.**

### F.3 Benign query URLs

| Field | Specification |
|---|---|
| Purpose | Close the HTTPS+query gap (53 rows in Kaggle benign training). |
| Required fields | URL with explicit scheme, non-empty query string, parameter names preserved. |
| Label semantics | Benign by construction. |
| Desired size | ≥10,000 rows, ≥1,000 distinct parameter names. |
| Provenance / licence | As F.2. |
| Timestamp / version | As F.2. |
| Leakage risks | Medium — search-result URLs can embed the query that produced them; strip referrer parameters. |
| Dedup | Normalise parameter order before comparing. |
| Text-only | Yes. |
| Candidates (not chosen) | Common Crawl query-bearing URLs; public analytics/link datasets. |

### F.4 Phishing URLs

| Field | Specification |
|---|---|
| Purpose | Improve phishing recall (S2-A deep phishing recall 0.686) with URLs malicious *as URLs*. |
| Required fields | Full URL, explicit scheme, label, first-seen timestamp. |
| Label semantics | Must mean "this URL is a phishing URL", not "this host was reported". |
| Desired size | 20,000–50,000 rows, ≥5,000 distinct hosts, ≤5 rows/host. |
| Provenance / licence | Documented feed with a citable snapshot; research-permitting licence. |
| Timestamp / version | Pinned snapshot; note that phishing URLs are short-lived — record first-seen. |
| Leakage risks | **High** — public blocklists are widely mirrored and may overlap the existing corpus. Must be checked against the fixed test split by hostname. |
| Dedup | Exact URL + normalised host+path+query. |
| Text-only | Yes. |
| Candidates (not chosen) | PhishTank, OpenPhish, URLhaus (phishing subset), APWG feeds. |

### F.5 Malware URLs

| Field | Specification |
|---|---|
| Purpose | Replace the `pastebin.com`-dominated malware-HTTPS block with a real population. |
| Required fields | Full URL, explicit scheme, label, timestamp. |
| Label semantics | Must mean "this URL distributes malware", not "this host appeared in a feed". |
| Desired size | 20,000–50,000 rows, **≥1,000 distinct hosts**, ≤5 rows/host. |
| Provenance / licence | Documented feed; research-permitting licence. |
| Timestamp / version | Pinned snapshot. |
| Leakage risks | **High** — same mirroring problem; also high risk of re-importing paste-site URLs. |
| Dedup | Exact URL + host cap per domain. |
| Text-only | Yes. |
| Candidates (not chosen) | URLhaus (malware subset), MalwareBazaar URL records, Abuse.ch feeds. |

**Measured constraint on F.5:** any malware source must be checked for host concentration *before*
use. The current block is 1,045 rows / 51 hosts / 93.40% one host. A source with >5 rows/host
reproduces the defect.

### F.6 Explicit non-recommendation

No source is selected here. Selection requires the acceptance criteria in D to be applied and the
leakage checks in D.11 to pass. Popularity is not a criterion.

---

## G. Evaluation / gate review

The existing 7 gates are **unchanged and retained** for historical comparability. Nothing is removed,
relaxed or silently reinterpreted.

### G.1 Existing promotion gates (unchanged)

| # | Gate | Threshold |
|---|---|---|
| 1 | Overall malicious recall | ≥ 0.87 |
| 2 | Malicious HTTPS recall | ≥ 0.90 |
| 3 | Tranco homepage safe | ≥ 0.95 |
| 4 | Suspicious synthetic | 4/4 |
| 5 | Legitimate deep safe | ≥ 0.90 |
| 6 | Malicious deep recall | ≥ 0.90 |
| 7 | No leakage | 0 leaked rows |

### G.2 Current standing

| Gate | R2 | S1 | S2-A |
|---|---|---|---|
| 1 Overall malicious recall ≥0.87 | ✅ 0.918828 | ❌ 0.816460 | ❌ 0.775968 |
| 2 Malicious HTTPS recall ≥0.90 | ✅ 0.985782 | ❌ 0.453253 | ❌ 0.460577 |
| 3 Tranco homepage safe ≥0.95 | ✅ 0.991000 | ❌ 0.025500 | ❌ 0.595500 |
| 4 Suspicious synthetic 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| 5 Legitimate deep safe ≥0.90 | ❌ 0.112074 | ✅ 0.941412 | ✅ 0.918239 |
| 6 Malicious deep recall ≥0.90 | ✅ 0.975001 | ❌ 0.840536 | ❌ 0.785617 |
| 7 No leakage | ✅ | ✅ | ✅ |
| **Passed** | **6/7** | **3/7** | **3/7** |

### G.3 Gate sensitivity — how far each gate is from passing (S2-A)

Derived arithmetically from the recorded evaluation, so the size of each deficit is explicit:

| Gate | Observed | Correct now | Correct needed | **Shortfall (rows)** | Denominator |
|---|---:|---:|---:|---:|---:|
| Overall malicious recall ≥0.87 | 0.775968 | 33,057 | 37,063 | **4,006** | 42,601 |
| Malicious HTTPS recall ≥0.90 | 0.460577 | 1,069 | 2,089 | **1,020** | 2,321 |
| Tranco homepage safe ≥0.95 | 0.595500 | 2,382 | 3,800 | **1,418** | 4,000 |
| Legitimate deep safe ≥0.90 | 0.918239 | 69,822 | 68,436 | **0 (surplus 1,386)** | 76,039 |
| Malicious deep recall ≥0.90 | 0.785617 | 29,540 | 33,841 | **4,301** | 37,601 |
| Suspicious synthetic 4/4 | 4/4 | — | — | 0 | 4 |
| No leakage | 0 | — | — | 0 | — |

Reading: the HTTPS gate needs 1,020 more correct rows out of 2,321 — but 976 of those rows are one
paste hostname. The Tranco gate needs 1,418 of 4,000 — and Tranco rows have no path or query, so the
model has only the host block and the intercept to work with.

### G.4 Additional Stage 7 diagnostics (recommended, not gates)

These do not change promotion. They make future results interpretable.

1. **Per-subtype recall** — phishing, malware, defacement reported separately. Prevents CMS-shape
   recognition from silently satisfying a global recall gate.
2. **Per-subtype HTTPS recall** — with **host counts alongside row counts**, and the largest-host
   share. *Mandatory given B.4.*
3. **HTTPS-only deep ROC-AUC** — R2 0.757214, S1 0.951609, S2-A 0.924561.
4. **All-deep ROC-AUC** — S1 0.964609, S2-A 0.947931.
5. **Threshold frontier / 90-90 reachability** — currently `both_0.90_reachable = false` for every
   model on both suites. Report the best-balanced operating point and its threshold.
6. **Tranco apex / `www.` / domain safety** — S2-A reports 0.5955 for all three with an
   apex-vs-`www.` score delta of exactly 0 over 2,000 pairs. Report all three, plus the delta.
7. **OOV rate** — host n-gram coverage: benign 0.807352, phishing 0.712708, malware 0.682299,
   defacement 0.783017.
8. **Zero-feature rate** — share of rows producing no known features at all.
9. **Intercept / no-evidence score** — S2-A intercept +0.301758; **score with no features = 0.574872
   → predicted malicious**. This is a defect, not a diagnostic: absence of evidence is being read as
   evidence of malice. 2 Tranco rows and 6 deep-benign rows hit it.
10. **Calibration** — reliability curve and Brier score per suite.
11. **Score distributions** — p10/median/p90 per subtype and scheme. Flat distributions
    (S1 malware-HTTPS p10 = median = p90 = 0.300623) are the signature of no usable evidence.

### G.5 Gates whose scientific interpretation may need reconsideration

Flagged for review. **Not changed here.**

**(a) Malicious HTTPS recall ≥0.90 — interpretation must change.**
The gate's denominator is 2,321 rows but effectively ~352 hosts, and 42.05% of it is one paste
hostname. A 0.90 threshold over this denominator is not a stable estimate of anything. The gate can
stay as a *historical* number, but it must be reported with host-level clustering and a
host-capped variant, or it will keep rejecting models for reasons unrelated to their quality.

**(b) Legitimate deep safe ≥0.90 — the harness must be examined.**
R2 scores 0.112074 in evaluated form and **0.867752 in raw stored form**. The difference is entirely
the `https://` rewrite of 70,214 bare rows, which perturbs `url_length`, `num_slashes`, `num_dots`,
`special_chars` and `uses_https` — R2's five highest-weighted features. The gate is measuring the
harness's rewriting convention as much as the model. The gate should stay, but the suite should be
reported in **both** forms, and the rewrite should be documented as part of the evaluation definition.

**(c) Malicious deep recall ≥0.90 — construct validity.**
51.2% of the denominator is defacement, whose label is "site was compromised" rather than "URL is
malicious" (section E). The gate is partly measuring CMS-URL recognition. Report per subtype.

**(d) Overall malicious recall ≥0.87 — same issue, diluted.**
Defacement is 19,268 of 42,601 malicious test rows (45.2%).

**(e) Tranco homepage safe ≥0.95 — the suite is synthetic.**
All 4,000 rows are `https://` + apex or `www.` with no path or query, generated from a popularity
list. It tests one very narrow form. It is a legitimate gate but should not be read as "general
benign safety."

**(f) No leakage — definition needs the normalised check.**
S2 required purging 16,172 rows for *normalised* hostname collisions. The gate must assert both raw
and normalised overlap, as S2's preflight now does.

---

## H. Proposed next experiment (exactly one)

**Designation: S3 — Benign Distribution Repair.**
**Status: proposed, not executed. Requires approval.**

### H.1 Objective

Test whether repairing the **benign training distribution** — specifically the HTTPS-deep
representation gap — raises Tranco safety and legitimate-deep safety without sacrificing malicious
recall, using the S2-A representation and model family **unchanged**.

This addresses training-data distribution. It does **not** delete, add or reweight any individual
lexical feature.

### H.2 Role of existing Kaggle data

- **Retained as the backbone.** All 487,729 S2 Kaggle training rows stay, with the S2-A
  `www.`-normalisation and the 16,172-row leakage purge intact.
- **Not re-split.** The fixed test split (128,214 rows) and all evaluation suites are untouched.
- **Down-weighted only if required by class balance**, and only by a single documented factor —
  no per-row or per-feature manipulation.

### H.3 Proposed external benign source type

**Type F.2 — benign real deep URLs with explicit `https://`** (section F.2). This is the only source
type that closes the measured gap.

Acceptance criteria, applied before any training:

| Criterion | Requirement | Basis |
|---|---|---|
| Rows | 40,000–90,000 usable | D.1 (measured scenarios) |
| HTTPS coverage | 100% explicit `https://` | D.2 (measured) |
| Scheme synthesis | 0 synthesised rows | D.2 (measured) |
| Deep share | ≥90% deep | D.3 |
| Query share | ≥25% with query | D.4 |
| Distinct domains | ≥10,000 | D.6 |
| Rows per host | ≤5 mean | D.6 (measured vs defacement 44.91) |
| Distinct TLDs | ≥150 | D.6 (measured) |
| Apex/www pairs | ≥20% of domains in both forms | D.5 |
| Test-split overlap | 0 raw **and** 0 normalised | D.11 (measured) |
| Tranco holdout overlap | 0 | D.11 |
| Legit-deep hostname overlap | 0 of 18,629 | D.11 |
| Provenance | documented method + date + hash | D.8 |
| Licence | research-permitting, recorded | D.9 |

**If no source meets all criteria, S3 does not run.** That is a valid outcome.

### H.4 Proposed malicious source

**Not proposed for S3.** Evidence does not support adding one yet:

- Malware-HTTPS is a 51-host measurement artifact, not a population deficiency that more rows of the
  same kind would fix.
- Phishing is the best-performing subtype on HTTPS (0.790).
- Adding malicious data would confound the experiment: S3 must isolate the benign lever.

A malicious source (F.4/F.5) is deferred to a later stage, contingent on S3's result.

### H.5 Target training composition

| Component | Rows | Share |
|---|---:|---:|
| Kaggle (S2-A, purged) | 487,729 | ~78% |
| Tranco generated (existing) | 12,905 | ~2% |
| **New benign HTTPS-deep** | **40,000–90,000** | **~6–13%** |
| **Total** | **540,634–590,634** | 100% |

Benign HTTPS-deep share moves from 0.41% to **7.7–15.7%** of benign training — a 19–38× improvement
on the measured baseline, targeting the 10–20% band from D.1.

Class balance is checked and reported; if benign:malicious drifts beyond the corpus's own
428,074:213,010 ratio by more than 10%, the new rows are subsampled to the lower bound (40,000).

### H.6 Treatment of defacement

**Retained unchanged in training.** Per section E.5, Option 2:

- Defacement stays a malicious class with its existing 76,040 training rows.
- Evaluation reports defacement **separately** (new diagnostic), so CMS-shape recognition cannot
  silently satisfy a global recall gate.
- No defacement rows are removed, relabelled or reweighted.
- The 7 gates are unchanged.

### H.7 Canonicalization

Identical to S2-A, with no new rules:

- Host block: lowercase, remove exactly one leading literal `www.`.
- Path / query / keyword blocks: unchanged.
- Scheme stripped before featurisation (scheme-blind), as in S1/S2.
- **No new normalisation of any kind.** Adding normalisation here would confound the benign-data
  lever with a feature change.

### H.8 Deduplication

1. Exact-URL dedup within the new source.
2. Normalised host+path+query dedup against the existing training set.
3. Cross-form collapse: where the same path exists as both `http://` and `https://`, keep HTTPS.
4. Host cap: no host may contribute >5× the mean rows/host.
5. Re-run the S2 normalised-collision purge and record the count.

### H.9 Group-based leakage prevention

- Split **by hostname group**, never by row (reuse `train.split_by_hostname`).
- Assert, before fitting: raw train/test hostname overlap = 0; normalised overlap = 0; Tranco
  holdout domains in train = 0; legit-deep hostnames in train = 0; sanity brand hosts in train = 0.
- Write these to `preflight.json` **before** the classifier is fitted, as S2 did.
- Any non-zero assertion aborts the run before training.

### H.10 Representation

**Unchanged from S2-A.** Field-separated TF-IDF with `www.`-normalised host:

| Block | Config | Vocab |
|---|---|---:|
| host | `char_wb`, 3–5 grams, min_df 3, sublinear | 30,000 |
| path | `word`, 1–2 grams, min_df 5 | 50,000 |
| query | `word`, 1–2 grams, min_df 5 | 7,500 |
| keyword counts | fixed vocabulary, no fitting | 45 |

Total ~87,545 features. Vocabulary fitted on S3 training rows only.

### H.11 Model family

**Unchanged: L2 logistic regression**, `max_iter=400`, same solver and C as S2-A. No tuning, no C
search, no ensemble. The experiment varies data, not model.

### H.12 Threshold policy

**Deployed threshold stays 0.5.** No tuning. Threshold sweeps are diagnostic only, as in S1/S2.

One bounded, explicitly-flagged sub-change is proposed for separate approval, because B.4/G.4 show
it is a defect rather than a tuning choice: **the no-evidence intercept.** S2-A scores 0.574872 —
predicted malicious — when a URL produces no features at all. That is absence of evidence being read
as evidence of malice, and it directly harms the Tranco gate. If approved, S3 reports the
no-evidence score and, as a **separate reported variant only**, a prior-corrected intercept. The
primary S3 result keeps the intercept untouched so the benign-data effect is isolated.

### H.13 Unchanged evaluation suites

- Fixed test split: 128,214 rows (unchanged, never resplit).
- Tranco holdout: 4,000 rows / 2,000 domains (unchanged).
- Legitimate-deep suite: 76,039 rows / 18,629 hosts (unchanged, evaluated in browser form).
- Malicious-deep suite: 37,601 rows (unchanged).
- Sanity suite: 10 legit + 4 suspicious (unchanged).
- Famous-URL diagnostics (unchanged).

### H.14 All existing 7 gates

Applied unchanged, with the same thresholds, in the same order, and reported as passed/failed
exactly as S1/S2 did. **No gate is relaxed.**

### H.15 Additional diagnostics

All of G.4, plus the two mandatory additions:

- **Host-clustered HTTPS recall** — per subtype, with host counts, largest-host share, and a
  host-capped recall variant.
- **Legitimate-deep safety in both raw and evaluated form** — to separate model behaviour from the
  `https://` rewrite.

### H.16 Artifact paths

```
model/experiment_seven/
  preflight.json          # written before any fit; aborts on any leakage assertion
  preflight.log
  s3.metadata.json
  s3.pkl
  feature_config.json
  evaluation.json
  comparison.json
  protected_before.json
  protected_after.json
  protected_diff.json
  run.log
```

No file is written into R2 / C / D / S1 / S2 directories.

### H.17 Expected resources

**Runtime — projected from measured S2 timings** (S2: corpus load 67.2 s, vectoriser fit 80.22 s,
train transform 57.9 s, classifier fit 19.12 s, test inference 19.69 s ≈ 245 s total for 500,634
rows):

- S3 at 540,634–590,634 rows ≈ 1.08–1.18× the rows.
- **Projected: 5–8 minutes** total, plus ~70–90 s corpus load.

**RAM — PROJECTION, not measured.** No memory instrumentation exists in the S1/S2 logs, so this is
arithmetic, not observation:

- Sparse CSR at ~87,545 features, ~600k rows, ~120 non-zeros/row ≈ 72M nnz.
- float32 data ≈ 288 MB + int32 indices ≈ 288 MB ≈ **~0.6 GB per materialised matrix**.
- With train + test matrices, TF-IDF intermediates and the classifier, **peak ≈ 3–6 GB**.
- **S3 must instrument peak RSS** (e.g. `tracemalloc` / `psutil`) and record it, so this projection
  is replaced by a measurement. This is a gap in the existing experiment records.

### H.18 Stopping / failure policy

S3 stops and reports **failure without promotion** if any of the following occurs. No retry with a
different setting, no threshold adjustment, no second candidate.

1. Any preflight leakage assertion is non-zero → **abort before fitting.**
2. The benign source fails any acceptance criterion in H.3 → **abort before fitting.**
3. Any of the 7 gates fails → **reject**, artifacts preserved.
4. Legitimate-deep safety or Tranco safety *falls* relative to S2-A → **reject**, even if other gates
   improve (the experiment's purpose is benign repair; a regression there falsifies the hypothesis).
5. Malicious recall drops by more than 0.02 absolute vs S2-A → **reject** as an unacceptable
   trade.
6. Runtime exceeds 60 minutes or peak RSS exceeds 12 GB → **abort** and report.

**Exactly one candidate is trained.** If it is rejected, S3 is closed and the result recorded; no
variant sweep follows in the same stage.

### H.19 Falsifiable hypothesis

> If the benign-HTTPS-deep representation gap is the binding constraint, then adding 40,000–90,000
> real benign HTTPS-deep rows, with representation, model family, threshold and evaluation suites all
> unchanged, will raise Tranco homepage safety from 0.5955 toward ≥0.95 and legitimate-deep safety
> above 0.918239, while malicious recall moves by less than 0.02.

If Tranco safety does not improve materially, the benign-coverage hypothesis is **falsified**, and
the remaining explanation is the no-evidence intercept and host-concentration artifacts — a
different, narrower problem.

---

## Files created

All new files are confined to `model/stage7_study/`. Nothing was written into R2 / C / D / S1 / S2
directories.

| File | Purpose |
|---|---|
| `model/stage7_study/stage7_analysis.py` | Addendum analysis: subtype×scheme profiles, evaluated vs raw deep suite, defacement study, integrity hashing. |
| `model/stage7_study/stage7_derived.py` | Sections D and G: gate sensitivity arithmetic, benign-gap targets. Consumes JSON only. |
| `model/stage7_study/stage7_https_concentration.py` | Effective sample size of the malicious-HTTPS gate (host clustering). |
| `model/stage7_study/stage7_addendum.json` | Output of the addendum analysis. |
| `model/stage7_study/stage7_derived.json` | Output of the derived analysis. |
| `model/stage7_study/stage7_https_concentration.json` | Output of the concentration analysis. |
| `model/stage7_study/protected_before.json` | SHA-256 of 40 protected artifacts before analysis. |
| `model/stage7_study/protected_after.json` | SHA-256 of the same 40 artifacts after analysis. |
| `model/stage7_study/protected_diff.json` | Before/after comparison. |
| `model/stage7_study/stage7_report.md` | This report. |
| `model/stage7_study/run.log` | Console log of all three analysis runs. |

---

## Protected-artifact integrity result

40 protected artifacts hashed before and after the full study.

**Result: 0 changed / 0 missing.**

Verified unchanged, including:

- `model/fraudshield_model.pkl` = `d74190913f656a4260578e4dfa1b4b8d17bc4b0681e1215d18974713876a1bf8` (R2, promoted)
- `model/model_metadata.json`
- `backend/feature_extractor.py`, `backend/predictor.py`, `backend/dataset_loader.py`
- `train.py`, `experiment_c.py`, `experiment_d.py`, `experiment_six.py`, `experiment_six_s2.py`
- `data/malicious_phish.csv` = `d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01`
- `data/tranco/top-1m.csv` = `c13d972cb8deb00bcada3d2087c283a067e7235e7acebd596ab21c0126cd95c5`
- `data/tranco/selection.json`
- All Experiment C, D, S1 and S2 artifacts (`*.pkl`, `*.metadata.json`, `comparison.json`,
  `evaluation.json`, `feature_config.json`, `host_artifact_audit.json`, `preflight.json`,
  `protected_before.json`, `protected_diff.json`, `deep_separability.json`)

---

## Unresolved scientific risks and questions

1. **The HTTPS gate may be unfixable as currently defined.** 42.05% of its rows are one paste
   hostname and 93.40% of its malware block is that same host. No amount of benign data changes what
   a paste URL looks like. This gate likely needs a host-capped reformulation before it can
   meaningfully pass — which is a change to the evaluation, and therefore needs explicit approval.

2. **The legitimate-deep harness rewrite is unstudied.** R2 scores 0.112 in evaluated form and 0.868
   in raw form. Which form represents deployment is a product question, not a modelling one, and it
   changes which model looks best. This has never been examined in the project.

3. **No memory measurements exist for any experiment.** S1/S2 recorded timings but not peak RSS. The
   H.17 RAM figure is a projection and is labelled as such.

4. **The benign source may not exist.** The acceptance criteria in H.3 are strict (100% explicit
   HTTPS, ≥90% deep, ≤5 rows/host, ≥10,000 domains, zero leakage against three holdouts). A
   compliant source may not be available under a suitable licence. If so, S3 cannot run, and the
   project needs a different route — likely the intercept correction plus a host-capped gate.

5. **Defacement's label remains ill-posed.** Option 2 (report separately) mitigates the measurement
   problem but does not fix the target. The class still teaches CMS-shape recognition. Whether that
   is acceptable for a deployed product is a decision above this study.

6. **The `www.` artifact is not fully explained.** S2 normalised it away, but 2,194 rows (0.450%)
   still contain `www` after normalisation, and defacement's 69.58% `www.` rate is far higher than
   any other class. The upstream cause (source scraping convention) is inferred, not proven.

7. **Class balance after adding benign data is unvalidated.** The 10% drift rule in H.5 is a
   judgement call. Adding 90,000 benign rows shifts the benign:malicious ratio materially, and its
   effect on precision is unknown without training.

8. **Scheme leakage is mitigated, not removed.** The residual scheme probe AUC is 0.880385 even
   under scheme-blind featurisation. Adding HTTPS-deep benign rows will reduce the *benign* scheme
   imbalance but does nothing about defacement's 100% HTTP.

---

NO MODEL TRAINED.
NO DATA DOWNLOADED.
NO EXISTING MODEL/EXPERIMENT ARTIFACT MODIFIED.
R2 REMAINS PROMOTED.
WAITING FOR APPROVAL.
