# FraudShield — `data/` folder

## `test_urls.csv` is NOT a training dataset

This file is a **tiny pipeline fixture for development/testing only** (20 lines).

It intentionally contains:

- 7 legitimate, well-known HTTPS sites (label `0`)
- 7 suspicious-looking URLs on reserved domains (label `1`)
- 1 URL containing a comma, to exercise CSV quoting
- 1 blank line, 1 invalid URL, 1 invalid label
- 2 exact duplicates and 1 label conflict

Do **not** quote accuracy numbers, class balance, or model quality from this file.
It exists only to prove that `backend/dataset_loader.py` validates, de-duplicates,
and converts rows correctly.

## Format for the real dataset

```
url,label
https://www.example.com,0
http://secure-login.example/verify-account,1
```

- `0` = legitimate / safe, `1` = malicious / suspicious
- Text labels such as `safe` / `phishing` also work (see `DEFAULT_LABEL_MAP`)
- Unknown labels are rejected and reported, never guessed
- Exact duplicate URLs are collapsed; conflicting labels for the same URL are
  reported and excluded (data-leakage protection)

## Provenance rule

Any dataset placed here for Stage 4 must come from a **documented, legitimate
source** and its origin/licence must be recorded before training. No dataset has
been downloaded automatically.
