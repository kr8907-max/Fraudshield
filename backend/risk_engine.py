"""
FraudShield - combined risk engine
==================================

Combines independent signals into one final verdict:

    * R2 URL-string model (ONE input among several; never the decider)
    * live website content inspection (backend.inspector)
    * HTTPS / TLS certificate facts
    * redirect behaviour
    * destination DNS facts
    * trusted-structure policy notes (backend.url_policy)

Known-threat-intelligence is NOT available in this build (external API
key required: e.g. Google Safe Browsing or URLhaus). It is reported as
"not configured" rather than faked.

Verdicts (security assessment)
-----------------------------
    SAFE                  sufficient evidence of low risk (full inspection
                          passed, valid HTTPS certificate, no significant
                          findings)
    SUSPICIOUS            concerning indicators found (medium findings,
                          no HTTPS, or model/inspection disagreement)
    MALICIOUS             strong phishing / malware evidence
    INSUFFICIENT EVIDENCE inspection incomplete or refused (site
                          unreachable, SSRF blocked, invalid certificate,
                          timeout, executable download, over-thin
                          evidence); no safety claim and no risk claim

Website availability (independent of the security verdict)
----------------------------------------------------------
Mapped ONLY from transport/status facts in backend.inspector:
    REACHABLE / PAGE_NOT_FOUND / TIMEOUT / CONNECTION_FAILED /
    DNS_FAILED / TITLE_UNAVAILABLE / UNKNOWN

A failed HTTP request never implies a URL is safe or malicious, and
HTTP 404, a missing HTML title, an empty body, or a slow response never
block URL-feature-based classification.

Every verdict carries explicit human-readable evidence strings. Raw model
output is always preserved and never used to call a URL malicious on
string structure alone.
"""

from __future__ import annotations

from urllib.parse import urlsplit

VERDICT_SAFE = "SAFE"
VERDICT_SUSPICIOUS = "SUSPICIOUS"
VERDICT_MALICIOUS = "MALICIOUS"
VERDICT_INSUFFICIENT = "INSUFFICIENT EVIDENCE"

# Backwards-compatible alias (old name mapped textually).
VERDICT_HIGH = VERDICT_MALICIOUS

AVAILABILITY_REACHABLE = "REACHABLE"
AVAILABILITY_PAGE_NOT_FOUND = "PAGE_NOT_FOUND"
AVAILABILITY_TIMEOUT = "TIMEOUT"
AVAILABILITY_CONNECTION_FAILED = "CONNECTION_FAILED"
AVAILABILITY_DNS_FAILED = "DNS_FAILED"
AVAILABILITY_TITLE_UNAVAILABLE = "TITLE_UNAVAILABLE"
AVAILABILITY_UNKNOWN = "UNKNOWN"

_AVAILABILITY_VALUES = frozenset(
    {
        AVAILABILITY_REACHABLE,
        AVAILABILITY_PAGE_NOT_FOUND,
        AVAILABILITY_TIMEOUT,
        AVAILABILITY_CONNECTION_FAILED,
        AVAILABILITY_DNS_FAILED,
        AVAILABILITY_TITLE_UNAVAILABLE,
        AVAILABILITY_UNKNOWN,
    }
)

THREAT_INTEL = {
    "available": False,
    "providers": [],
    "note": (
        "No threat-intelligence feed is configured. This build performs "
        "local inspection only; feeds such as Google Safe Browsing or "
        "URLhaus require an API key and have not been added."
    ),
}

# Findings carry severities; these weights turn them into a content-risk
# signal for transparency (not a probability).
_SEVERITY_WEIGHTS = {"high": 55, "medium": 25, "low": 8, "info": 0}


def _host_of(url: str | None) -> str:
    try:
        return (urlsplit(url or "").hostname or "").lower()
    except ValueError:
        return ""


def _base_findings(inspection: dict, tls: dict | None) -> list[dict]:
    """Structural/certificate/transport findings, independent of page text."""
    findings: list[dict] = []

    if inspection.get("blocked"):
        findings.append(
            {
                "severity": "high",
                "code": "unsafe_destination",
                "message": "Inspection refused: destination resolves to a forbidden "
                "range (private, loopback, link-local, metadata or reserved "
                f"address): {inspection['blocked']}. Nothing was contacted.",
            }
        )
        return findings

    if inspection.get("redirect_scheme_downgrade"):
        findings.append(
            {
                "severity": "medium",
                "code": "https_downgraded_on_redirect",
                "message": "Redirect chain downgrades from HTTPS to plain HTTP.",
            }
        )

    if inspection.get("served_executable"):
        findings.append(
            {
                "severity": "high",
                "code": "executable_download",
                "message": "The destination served an executable/binary "
                "attachment; nothing was executed or downloaded.",
            }
        )

    flags = inspection.get("host_flags") or []
    if flags:
        _idn_note = (
            "Hostname uses IDN/punycode or non-ASCII characters "
            f"({', '.join(flags)}); visually confusable lookalike domains "
            "often use these. Not proof of anything by itself."
        )
        findings.append(
            {
                "severity": "medium",
                "code": "deceptive_hostname",
                "message": _idn_note,
            }
        )

    first_hop_scheme = (inspection.get("hops") or [{}])[0].get("url", "")
    if first_hop_scheme.startswith("http:"):        findings.append(
            {
                "severity": "medium",
                "code": "no_https_entry",
                "message": "Inspection URL entered over unencrypted HTTP; traffic "
                "could be observed or altered in transit.",
            }
        )

    if tls is None and first_hop_scheme.startswith("https:"):
        findings.append(
            {
                "severity": "high",
                "code": "tls_error",
                "message": "HTTPS connection could not be verified "
                f"({inspection.get('error') or inspection.get('ssl_error') or 'TLS failure'}).",
            }
        )
    return findings


def _page_findings(inspection: dict) -> list[dict]:
    """Content findings from the parsed page (phase-2 signal)."""
    findings: list[dict] = []
    page = inspection.get("page") or {}
    final_host = _host_of(inspection.get("final_url"))

    password_fields = page.get("password_fields") or 0
    sensitive = page.get("sensitive_inputs") or 0
    forms = page.get("forms") or 0
    title = (page.get("title") or "").lower()

    if password_fields and not inspection.get("tls"):
        findings.append(
            {
                "severity": "high",
                "code": "password_over_http",
                "message": "Password field on a page served WITHOUT verified HTTPS; "
                "credentials could be intercepted.",
            }
        )

    sensitive_total = sensitive + password_fields
    if sensitive_total and not password_fields and not inspection.get("tls"):
        findings.append(
            {
                "severity": "medium",
                "code": "sensitive_form_over_http",
                "message": f"Sensitive form inputs ({sensitive_total}) on a page "
                "without verified HTTPS.",
            }
        )

    if page.get("meta_refresh"):
        target = page["meta_refresh"]
        if isinstance(target, str) and "=" in target:
            # Parse "0; url=https://..." content into the bare URL.
            from backend.inspector import _parse_meta_refresh_target

            target = _parse_meta_refresh_target(target) or target
        target_host = _host_of(target) if "://" in target else final_host
        if isinstance(target, str) and target.lower().startswith("http"):
            if target_host and target_host != final_host:
                findings.append(
                    {
                        "severity": "medium",
                        "code": "meta_refresh_redirect",
                        "message": "Meta-refresh redirect to a different domain: "
                        f"{target[:120]}.",
                    }
                )

    if (page.get("http_images") or 0) > 0 and inspection.get("tls"):
        findings.append(
            {
                "severity": "low",
                "code": "mixed_content",
                "message": "Page loads some images over plain HTTP (mixed content).",
            }
        )

    if (page.get("iframes") or 0) and len(page.get("iframes") or []) and any(
        _host_of(src) not in ("", final_host) for src in page.get("iframes") or []
    ):
        findings.append(
            {
                "severity": "low",
                "code": "cross_origin_iframes",
                "message": "Cross-origin iframe embedding detected.",
            }
        )

    if inspection.get("content_truncated"):
        findings.append(
            {
                "severity": "info",
                "code": "content_truncated",
                "message": "Response exceeded the inspection size cap; analysis "
                "used the first portion of the document.",
            }
        )
    return findings


def _impersonation_findings(inspection: dict) -> list[dict]:
    """Brand-name-on-wrong-host detection (imported lazily to avoid cycles)."""
    from backend.inspector import extract_brand_claim

    findings: list[dict] = []
    brand = extract_brand_claim(inspection.get("page") or {}, _host_of(inspection.get("final_url")))
    if brand:
        findings.append(
            {
                "severity": "high",
                "code": "brand_impersonation",
                "message": f"The page title references '{brand}' but the hostname is "
                f"not an official {brand} domain "
                f"({_host_of(inspection.get('final_url'))}). Classic phishing "
                "impersonation pattern.",
            }
        )
    return findings


def _summarize_tls(tls: dict | None) -> list[str]:
    if not tls:
        return []
    lines = [
        "HTTPS certificate verified during the connection: subject "
        f"'{tls.get('subject_cn')}', issuer '{tls.get('issuer')}'."
    ]
    if tls.get("not_after"):
        lines.append(f"Certificate expires {tls.get('not_after')} (server {tls.get('hostname')}).")
    if tls.get("tls_version"):
        lines.append(f"Transport: {tls.get('tls_version')} ({tls.get('cipher')}).")
    return lines


def _redirect_evidence(inspection: dict) -> list[str]:
    lines: list[str] = []
    hops = inspection.get("hops") or []
    if len(hops) > 1:
        chain = " -> ".join(
            f"{hop.get('status')} {_host_of(hop.get('url')) or 'hop'}" for hop in hops
        )
        lines.append(f"Redirect chain followed and re-validated at every hop: {chain}.")
    final_url = inspection.get("final_url")
    if final_url and hops and _host_of(final_url) != _host_of(hops[0].get("url")):
        lines.append(
            "Destination leaves the original hostname; the new host "
            f"({_host_of(final_url)}) was independently DNS-validated before use."
        )
    return lines


def _content_evidence(inspection: dict) -> list[str]:
    page = inspection.get("page") or {}
    lines: list[str] = []
    if page.get("title"):
        lines.append(f"Live page title: \"{page['title'][:120]}\".")
    if page.get("text_length"):
        lines.append(
            f"Readable page content inspected ({page['text_length']} characters, "
            f"{page.get('forms') or 0} form(s), {page.get('password_fields') or 0} "
            "password field(s))."
        )
    return lines


def _dns_evidence(inspection: dict) -> list[str]:
    dns = inspection.get("dns") or {}
    ips = dns.get("validated_ips") or []
    if ips:
        shown = ", ".join(ips[:3])
        more = "..." if len(ips) > 3 else ""
        return [
            f"Destination '{dns.get('host')}' resolved and validated to "
            f"{len(ips)} address(es) ({shown}{more}); no private, loopback, "
            "link-local, cloud-metadata or reserved ranges present."
        ]
    return []


def _content_risk(findings: list[dict]) -> int:
    score = sum(_SEVERITY_WEIGHTS.get(f["severity"], 0) for f in findings)
    return min(score, 100)


def _availability_of(inspection: dict) -> str:
    """Availability comes from transport/status facts, never from security."""
    value = inspection.get("availability")
    if value in _AVAILABILITY_VALUES:
        return value

    # Fallback mapping for inspections produced before the availability
    # field existed (tests with synthetic dicts, etc.).
    if inspection.get("blocked"):
        return AVAILABILITY_UNKNOWN
    if not inspection.get("reachable"):
        error = inspection.get("error") or ""
        if "timeout" in error:
            return AVAILABILITY_TIMEOUT
        if "dns" in error or "resolve" in error:
            return AVAILABILITY_DNS_FAILED
        if error:
            return AVAILABILITY_CONNECTION_FAILED
        return AVAILABILITY_UNKNOWN
    status = inspection.get("status_code")
    if status in (404, 410):
        return AVAILABILITY_PAGE_NOT_FOUND
    if (inspection.get("page") or {}).get("title"):
        return AVAILABILITY_REACHABLE
    return AVAILABILITY_TITLE_UNAVAILABLE


def decide(url_model_result: dict | None, inspection: dict | None) -> dict:
    """
    Combine the R2 model output and the live inspection into a verdict.

    Always returns a dict with:
        verdict, availability, evidence (list[str]), findings (list[dict]),
        signals (dict: per-source summaries), url_model (dict or None),
        threat_intelligence (dict)

    Security verdict rules (availability never feeds the verdict):
        * a refused/unreachable fetch yields at most INSUFFICIENT EVIDENCE
          (never SAFE and never MALICIOUS from the failure alone);
        * reaching the destination (even a 404/500/empty/JSON/slow body)
          never blocks URL-feature-based classification;
        * high findings -> MALICIOUS, medium findings -> SUSPICIOUS;
        * a full benign inspection overrides string-model suspicion
          (with explanation); raw model output is always preserved.
    """
    inspection = inspection or {}
    findings: list[dict] = []
    findings.extend(_base_findings(inspection, inspection.get("tls")))
    if inspection.get("reachable"):
        findings.extend(_page_findings(inspection))
        findings.extend(_impersonation_findings(inspection))

    has_high = any(f["severity"] == "high" for f in findings)
    has_medium = any(f["severity"] == "medium" for f in findings)

    raw_prediction = (url_model_result or {}).get("raw_prediction")
    model_score = (url_model_result or {}).get("model_score")
    policy = (url_model_result or {}).get("policy") or {}

    fully_inspected = (
        inspection.get("reachable")
        and inspection.get("is_html")
        and bool(inspection.get("tls"))
        and not inspection.get("error")
        and bool(inspection.get("page"))
        and (inspection.get("page") or {}).get("text_length", 0) > 0
        and not inspection.get("content_truncated")
    )

    # ------------------------------------------------------------------
    # Availability first (pure status facts, independent of the verdict)
    # ------------------------------------------------------------------
    availability = _availability_of(inspection)

    # ------------------------------------------------------------------
    # Security verdict (R2 is one input, never the decider)
    # ------------------------------------------------------------------
    if inspection.get("blocked"):
        # Nothing was contacted: inspection refusal is not proof of harm.
        verdict = VERDICT_INSUFFICIENT
    elif not inspection.get("reachable"):
        # Failed fetch (timeout / DNS / connection / TLS): never SAFE and
        # never MALICIOUS from the failure alone.
        verdict = VERDICT_INSUFFICIENT
    elif has_high:
        verdict = VERDICT_MALICIOUS
    elif has_medium:
        verdict = VERDICT_SUSPICIOUS
    elif fully_inspected:
        verdict = VERDICT_SAFE
    elif raw_prediction == 1:
        # Thin/unavailable page content cannot refute the string-model
        # flag; URL-feature evidence stands on its own.
        verdict = VERDICT_SUSPICIOUS
    else:
        verdict = VERDICT_INSUFFICIENT

    # ------------------------------------------------------------------
    # Assemble the evidence text for the reported verdict
    # ------------------------------------------------------------------
    evidence: list[str] = []
    verdict_prelude = {
        VERDICT_SAFE: "Verdict SAFE: full inspection found no significant risk "
        "indicators.",
        VERDICT_SUSPICIOUS: "Verdict SUSPICIOUS: concerning indicators were found.",
        VERDICT_MALICIOUS: "Verdict MALICIOUS: strong phishing/malware indicators found.",
        VERDICT_INSUFFICIENT: "Verdict INSUFFICIENT EVIDENCE: inspection incomplete or "
        "unavailable; no safety claim and no risk claim is made.",
    }
    evidence.append(verdict_prelude[verdict])
    evidence.append(f"Website availability: {availability}.")
    evidence.extend(_dns_evidence(inspection))
    evidence.extend(_redir := _redirect_evidence(inspection))
    evidence.extend(_summarize_tls(inspection.get("tls")))
    evidence.extend(_content_evidence(inspection))

    if findings:
        findings_text = "; ".join(f["message"] for f in findings if f["severity"] != "info")
        if findings_text:
            evidence.append("Findings: " + findings_text)
    elif verdict == VERDICT_SUSPICIOUS:
        evidence.append(
            "Content inspection produced no direct findings; the caution "
            "level comes from the URL-string model flag and from evidence "
            "that could not be fully verified."
        )
    else:
        evidence.append(
            "No suspicious findings: no password fields, no brand impersonation, "
            "no cross-domain redirects, no executable downloads."
        )

    # R2 relationship, stated explicitly everywhere except when unverified.
    if raw_prediction is not None:
        if raw_prediction == 1 and verdict == VERDICT_SAFE:
            evidence.append(
                "Note: the URL-string model (R2) flagged this URL "
                f"(model score {model_score:.3f}) because of lexical features such "
                "as path depth and length. Live inspection of the actual "
                "destination found no phishing indicators on a valid HTTPS "
                "certificate, so the string-model score was overridden."
                + (" Trusted-URL structural policy also verified the URL shape."
                   if policy.get("approved") else "")
            )
        elif raw_prediction == 1:
            if findings:
                evidence.append(
                    "The URL-string model (R2) also flagged this URL "
                    f"(model score {model_score:.3f}); it agrees with the "
                    "inspection findings but did not determine the verdict."
                )
            else:
                evidence.append(
                    "The URL-string model (R2) flagged this URL "
                    f"(model score {model_score:.3f}); it is the main risk "
                    "signal here, and the returned page content could not "
                    "confirm or refute it."
                )
        else:
            evidence.append(
                f"URL-string model (R2) score: {model_score:.3f} (malicious-class "
                "probability, not a calibrated probability); reported separately "
                "from the final verdict."
            )

    signals = {
        "content_risk": _content_risk(findings),
        "finding_counts": {
            "high": sum(1 for f in findings if f["severity"] == "high"),
            "medium": sum(1 for f in findings if f["severity"] == "medium"),
            "low": sum(1 for f in findings if f["severity"] == "low"),
        },
        "r2_raw_decision": raw_prediction,
        "r2_model_score": model_score,
        "full_inspection": fully_inspected,
        "redirects_followed": max(len(inspection.get("hops") or []) - 1, 0),
    }

    return {
        "verdict": verdict,
        "availability": availability,
        "evidence": evidence,
        "findings": findings,
        "signals": signals,
        "threat_intelligence": THREAT_INTEL,
    }
