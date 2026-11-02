# FraudShield — Security Architecture and Limitations

*What protects the machine running the scanner, what protects the accuracy of
the verdict, and — just as important — what the scanner **cannot** guarantee.
Every mechanism below is verifiable in `backend/inspector.py`,
`backend/risk_engine.py`, `backend/url_policy.py` and the test suite; source
functions/lines are named so claims can be checked.*

---

## 1. The two security roles of this project

FraudShield has to be safe in two directions:

1. **Operator safety (SSRF class):** the inspector is a *server-side* fetcher.
   Without protections, an attacker-supplied URL could make your machine call
   itself (`http://127.0.0.1:8000/api/scan`), probe an internal network
   (`http://192.168.x.x/`), or hit a cloud-metadata endpoint
   (`http://169.254.169.254/…`). This class of bug is prevented *before any
   connection is made*.
2. **Verdict integrity:** the output must not over-claim. A failed fetch, a
   TLS error, or a thin page is **not** evidence of safety **or** of malice —
   it produces `INSUFFICIENT EVIDENCE` or, when a string-model flag stands
   unrefuted, a cautious `SUSPICIOUS`.

---

## 2. SSRF protection (`inspector.py`)

### 2.1 Blocklist — `_ip_is_unsafe()`

Every resolved address is parsed with `ipaddress` and refused if it is:

| Range class | Examples | Covered via |
|---|---|---|
| Loopback | `127.0.0.1`, `::1` | `ip.is_loopback` |
| Private (RFC 1918 / ULA) | `10/8`, `172.16/12`, `192.168/16`, `fc00::/7` | `ip.is_private` |
| Link-local | `169.254/16` — **includes the cloud-metadata endpoint** | `ip.is_link_local` |
| Cloud metadata, explicit | `169.254.169.254` | explicit check |
| Multicast / reserved / unspecified | `224/4`, `240/4`, `::` etc. | `is_multicast` / `is_reserved` / `is_unspecified` |
| CGNAT (RFC 6598) | `100.64.0.0/10` | explicit network check (Python's `is_private` does not cover it) |
| Benchmarking | `198.18.0.0/15` | explicit network check |
| Broadcast | `255.255.255.255` | explicit check |

**Literal-IP hostnames** (`http://127.0.0.1/…`) go straight to the safety
check in `_validate_destination()` — they never touch DNS. A refused
destination yields `blocked = "unsafe_destination:<ip>"` and **nothing is
contacted** (the risk engine treats `blocked` as INSUFFICIENT EVIDENCE:
"a refusal is not proof of harm").

**IPv4-mapped IPv6** (`::ffff:127.0.0.1`, `::ffff:169.254.169.254`) is parsed
by `ipaddress` into an IPv4 address on supported runtimes and therefore hits
the same blocklist. Tests explicitly assert these forms are blocked
(`tests/test_evaluation.py`, `tests/test_inspection.py`).

### 2.2 Resolver-level bypass tricks

Decimal/hex/octal spellings such as `0177.0.0.1`, `0x7f.1`, `2130706433` are
handled at the *resolver* boundary: they either fail to resolve
(`dns_resolution_failed`) or resolve to an IP that the blocklist refuses.
Tests cover both outcomes. Non-`gaierror` `OSError`s from `getaddrinfo` are
handled as `dns_resolution_failed` — a production fix (previously could
crash), verified by a dedicated test.

### 2.3 DNS rebinding protection

`_validate_destination()` returns the **full list of distinct validated IPs**.
The connection is then **pinned to a validated IP** (`_HTTPSConnection` /
`_HTTPConnection` connect to the pinned address), while the hostname is kept
only for the TLS SNI and the HTTP `Host` header. So the name→IP mapping is
checked once per hop and never re-resolved behind the scanner's back.
At **every** redirect hop the destination is re-validated from scratch
(`hop_index` loop, "destination validation (EVERY hop)").

### 2.4 Multi-address connect fallback

If the first validated IP refuses the connection, the next validated IP is
tried (`for pinned_ip in dest["ips"]`). **TLS errors do not hop**: a
certificate/SSL failure is treated as host-specific and stops immediately
(code comment: other addresses should not silently retry the cert handshake).

---

## 3. Input validation (`_validate_scheme_url()`, predictor, API)

Before any network activity:

- Only `http`/`https` schemes accepted (`javascript:`, `file:`, `data:`… →
  `non_web_scheme`); empty input; non-string input.
- Embedded credentials (`user:pass@host`) → `embedded_credentials`.
- Invalid port or hostname (spaces, empty) → clean error.
- IP-host quirks are handled through the literal-IP path above.

Errors surface as clean 4xx with a machine-readable `code`
(`tests/test_api_integration.py`); **no request is issued when input is
invalid**.

---

## 4. Redirect policy

`hop_index in range(MAX_REDIRECTS + 1)` — up to **MAX_REDIRECTS = 5** hops:

| Rule | Behaviour |
|---|---|
| Per-hop revalidation | every hop's scheme/host/port re-parsed; DNS re-validated |
| Scheme whitelist | `Location` must be http/https → otherwise `redirect_to_non_http` (covers `javascript:`/`data:`/`ftp:` redirect tricks) |
| HTTPS→HTTP downgrade | recorded as `redirect_scheme_downgrade = True` → produces a finding/risk in the engine |
| Credentials in redirect | `redirect_to_credentials` |
| Redirect cap | exceeded → `too_many_redirects` |
| Budget | each hop shares the single 25 s total budget |

Important nuance: **a redirect chain that ends at a legitimate page is not
evidence of malice, and a downgrade alone is a finding but not an automated
MALICIOUS verdict** — the engine severity ladder decides
(`docs/SYSTEM_ARCHITECTURE.md` §2).

---

## 5. TLS

- Connections wrap with **`ssl.create_default_context()`** — the platform
  default: certificate chain AND hostname verification enabled (SNI = the
  real hostname; the socket dials the pinned IP).
- Recorded facts: protocol version, cipher, subject CN (falling back to the
  first SAN `DNS` entry for modern SAN-only certs), issuer organisation,
  `notAfter`.
- A TLS failure ⇒ `tls_certificate_error` → availability
  `CONNECTION_FAILED` → **INSUFFICIENT EVIDENCE** in the engine.
  A bad certificate is reported honestly; it is *never* interpreted as proof
  of malware, and a strictly-verified failure can never be waved through to
  claim a page "checked out".

---

## 6. Timeouts, budget, and size limits

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_CONNECT_TIMEOUT` | 6.0 s | per TCP/TLS connect attempt |
| `DEFAULT_READ_TIMEOUT` | 12.0 s | per response read (socket read timeout) |
| `DEFAULT_TOTAL_BUDGET` | 25.0 s | whole-inspection deadline (checked before every hop; overshoot → `inspection_timeout`, availability `TIMEOUT`) |
| `MAX_BODY_BYTES` | 2,000,000 (2 MB) | hard cap on the payload considered |

- Body-read timeouts are **honest partial-content results**: `body_error =
  "body_read_timeout"`, availability `TIMEOUT` — the engine treats a timed-out
  body as *not fully inspected*, so content can never silently count as
  benign evidence for the SAFE override.
- Oversize responses are **truncated at 2 MB** with
  `content_truncated = true`; truncation is itself a finding input
  (an incompletely-read page cannot support the full SAFE override either).
- A request includes `Connection: close`, a descriptive `User-Agent`
  (`FraudShieldWebsiteInspector/1.0 (+read-only safety inspection)`) and
  standard HTML accept headers — one browser-like GET per hop.

---

## 7. Content handling — read-only guarantees

- The body is parsed by a **stdlib `html.parser.HTMLParser`** subclass:
  text extraction, title, forms/password/sensitive inputs, links, external
  scripts, iframes, `<meta http-equiv=refresh>`, `img` without https.
- **JavaScript is never executed.** External scripts are only *counted*.
- **Forms are never submitted**; the scanner performs exactly one GET per hop.
- Executable/binary content types (per `_is_executable_content`, incl.
  Content-Disposition sniffing) are **refused before deeper parsing** and
  recorded as `served_executable` — nothing is downloaded further.
- Malformed HTML is tolerated (`parser.feed` wrapped in try/except; parser
  exceptions never crash a scan).
- Non-HTML final bodies (JSON APIs, error pages) are still read (up to
  16 KB decoded sample) and recorded as evidence — a 404 or an empty 200
  **never blocks URL-feature-based classification** (deliberate design:
  thin/absent content ≠ guilt, but also never the SAFE override).

---

## 8. IDN / punycode / non-ASCII handling

`_hostname_flags(host)` records two informational flags:

- `punycode_host` — any dot-label starting `xn--`.
- `non_ascii_host` — any character above `\x7f` in the host.

The risk engine converts these into a **medium finding
(`deceptive_hostname`)**, i.e. every punycode / non-ASCII hostname is
verdict-capped at **SUSPICIOUS** (never auto-SAFE). This is a deliberate,
documented trade-off: homoglyph attacks (`аpple.com` with Cyrillic а, …)
are a real phishing pattern, and today's engine cannot reliably distinguish
a legitimate internationalized domain from a deceptive lookalike — so it
chooses caution over guesswork. Consequence: legitimate IDNs (common in some
country-code spaces) receive cautious verdicts.

Related, structural: brand impersonation uses a **dot-anchored official-host
check** (`_host_is_official_brand`): only subdomains *of* the official domain
match (e.g. `amazon.com` / `www.amazon.com` are Amazon; `paypal.com.evil.xyz`
is *not* PayPal). The URL policy layer matches trusted platforms by **exact
host** (no suffix wildcards), so `instagram.com.evil.com` never gets
structural approval.

---

## 9. What the scanner cannot guarantee

Stated plainly, per project rule 6 (no overstating):

1. **No field-proven detection rate for real phishing.** All detection/FP
   figures in this project come from *synthetic* crafted banks
   (`docs/TESTING_AND_EVALUATION.md`); no live phishing-corpus trial exists.
   A real adversary with patterns absent from our bank may evade the current
   findings.
2. **No threat-intelligence feeds.** The response honestly reports
   `threat_intelligence.available = false` (`risk_engine.py` `THREAT_INTEL`).
   Known-URL databases (Safe Browsing, URLhaus) are future work; until then,
   a known-bad URL with a slow/unreachable host can end up INSUFFICIENT
   EVIDENCE rather than MALICIOUS.
3. **One browser-like request per hop.** No JavaScript execution, no
   cookies/login flows, no form submission — so JS-gated or login-gated
   attacker content (password forms behind a bot wall, post-render DOM
   injection) cannot be seen.
4. **A SAFE verdict is a statement about *evidence*, not a guarantee of
   safety.** It means: reachable, valid TLS, real HTML content, complete
   2 MB-capped read, and zero findings *from the checks performed*. Content
   delivered conditionally (JS), Cloaking per user-agent, or payloads above
   the considered window are out of scope. It is not an antivirus, sandbox,
   or browser-security product.
5. **Availability ≠ safety, in both directions.** `REACHABLE` says only that
   transport and status worked. `DNS_FAILED`/`TIMEOUT`/`CONNECTION_FAILED`
   say nothing about the URL's intent — that's why the engine refuses to
   convert them into verdicts (INSUFFICIENT EVIDENCE) and why a lookalike
   domain that fails to resolve is still recorded with its (flagging) model
   score.
6. **Inspection completeness is bounded.** 2 MB window, 25 s budget, 5 hops.
   Larger/slower/more-deeply-redirected sites yield honest partial results
   (truncation finding, TIMEOUT availability) — deliberately **not** rounded
   up to a full judgement.
7. **Anti-fingerprinting limitations.** A fixed, self-identifying User-Agent
   is sent; sophisticated hosting can serve different content to scanners
   than to browsers. The scanner cannot detect per-client cloaking in
   general (this is a fundamental limit of single-request fetchers).
8. **IDN trade-off** (see §8): legitimate internationalized domains get
   cautious verdicts. Precision was consciously traded for recall here.
9. **Model-side limitation remains.** R2 flags ordinary deep-path URLs
   (`docs/MODEL_METHODOLOGY.md` §6). The scanner compensates by live
   inspection but does not fix the model; URLs no inspection can reach will
   show the model's flag alone (cautious SUSPICIOUS) or INSUFFICIENT
   EVIDENCE.

---

## 10. Quick map: threat → protection

| Threat | Protection (where) |
|---|---|
| SSRF to internal/metadata services | `_ip_is_unsafe` blocklist pre-contact, every hop |
| IPv4-mapped IPv6 bypass | `ipaddress` parse normalises; tests assert blocked |
| Resolver trick spellings | resolver-first design: unresolvable or refused |
| DNS rebinding | per-hop DNS validation + IP pinning (SNI/Host keep hostname) |
| Redirect loops / meta-hops | cap 5 + total 25 s budget |
| Redirect to `javascript:`/non-web | `redirect_to_non_http` |
| HTTPS downgrade | recorded finding (`redirect_scheme_downgrade`) |
| Bad certificate | strict default context; honest failure, never waved through |
| Huge downloads | 2 MB cap + `content_truncated` finding input |
| Slow/hanging servers | 6 s / 12 s / 25 s timeouts; partial bodies recorded as such |
| Drive-by downloads / JS execution | no execution, no submission; executables refused |
| Credential-harvesting pages | password/sensitive-field findings (only with http/no-TLS confidence) |
| Brand impersonation | dot-anchored brand namespace check on page title |
| Punycode homoglyphs | `xn--` / non-ASCII → medium finding (SUSPICIOUS cap) |
| Model FP/FN on strings | raw score preserved but never decisive; live evidence rules |
