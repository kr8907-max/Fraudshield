"""
FraudShield - safe website inspector
====================================

Performs a read-only, no-frills inspection of a real destination website:

    1. Resolves the destination hostname and validates every IP address.
    2. Blocks loopback, private, link-local (incl. 169.254.169.254 cloud
       metadata), CGNAT, benchmark, multicast, reserved and broadcast
       destinations (SSRF protection).
    3. Connects to the VALIDATED address only, with the original hostname
       kept for TLS SNI / the Host header.
    4. Follows up to MAX_REDIRECTS redirects, revalidating every hop's
       destination exactly the same way. Never downgrades https -> http
       silently; scheme changes are recorded and host changes re-validated.
    5. Strict connect/read timeouts and a hard response-size cap.
    6. Parses the returned HTML with the standard-library HTMLParser.
       JavaScript is NEVER executed; forms are NEVER filled or submitted;
       nothing is uploaded and no executables are downloaded.

The module only READS pages. It returns a plain dict (JSON-friendly) and
never raises for ordinary failures -- unreachable sites, timeouts, TLS
failures and unsafe destinations are reported in the result.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

MAX_REDIRECTS = 5
DEFAULT_CONNECT_TIMEOUT = 6.0   # seconds per TCP/TLS connection attempt
DEFAULT_READ_TIMEOUT = 12.0      # seconds for one response read
DEFAULT_TOTAL_BUDGET = 25.0     # seconds for the whole inspection
MAX_BODY_BYTES = 2_000_000      # hard cap on response payload considered
MAX_TEXT_SAMPLE = 4_000         # characters of visible text kept in result

USER_AGENT = "FraudShieldWebsiteInspector/1.0 (+read-only safety inspection)"

# Hostname reference table used ONLY to detect brand impersonation
# (title/hostname claims a brand without being that brand's domain).
# This is NOT a safe-list: it never approves anything; it only raises
# suspicion when a brand keyword appears on a non-official hostname.
BRAND_OFFICIAL_HOSTS: dict[str, frozenset[str]] = {
    "chatgpt": frozenset({"chatgpt.com", "chat.openai.com"}),
    "openai": frozenset({"openai.com", "chat.openai.com", "api.openai.com"}),
    "github": frozenset({"github.com", "gist.github.com"}),
    "paypal": frozenset({"paypal.com", "www.paypal.com"}),
    "apple": frozenset({"apple.com", "icloud.com"}),
    "google": frozenset({"google.com", "mail.google.com", "drive.google.com"}),
    "gmail": frozenset({"mail.google.com"}),
    "microsoft": frozenset({"microsoft.com", "office.com", "outlook.com", "login.microsoftonline.com"}),
    "amazon": frozenset({"amazon.com", "amazon.co.uk", "amazon.de"}),
    "netflix": frozenset({"netflix.com"}),
    "facebook": frozenset({"facebook.com", "fb.com"}),
    "instagram": frozenset(
        {
            "instagram.com",
            "www.instagram.com",
            "m.instagram.com",
            "help.instagram.com",
            "login.instagram.com",
            "ig.me",
        }
    ),
    "youtube": frozenset({"youtube.com", "www.youtube.com", "youtu.be", "music.youtube.com"}),
    "binance": frozenset({"binance.com", "accounts.binance.com"}),
    "steam": frozenset({"steampowered.com", "store.steampowered.com"}),
    "whatsapp": frozenset({"whatsapp.com", "web.whatsapp.com"}),
}

# Content types we treat as executable/binary and refuse to parse deeply.
_EXEC_CONTENT_TYPES = (
    "application/octet-stream",
    "application/x-msdownload",
    "application/x-msi",
    "application/x-exe",
    "application/vnd.microsoft.portable-executable",
)


class UnsafeDestinationError(Exception):
    """Raised when the destination resolves to a forbidden IP range."""


class _PageParser(HTMLParser):
    """Read-only HTML collector. Never evaluates anything."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.in_title = False
        self.in_skip = 0  # depth inside <script>/<style>
        self.visible_text_parts: list[str] = []
        self.forms = 0
        self.form_actions: list[str] = []
        self.password_inputs = 0
        self.sensitive_inputs = 0
        self.links: list[str] = []
        self.external_scripts = 0
        self.iframes: list[str] = []
        self.meta_refresh: str | None = None
        self.img_without_https = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrd = {k: (v or "") for k, v in attrs}

        if tag in ("script", "style"):
            self.in_skip += 1
            if tag == "script" and attrd.get("src"):
                self.external_scripts += 1
            return

        if tag == "title" and not self.in_title:
            self.in_title = True

        if tag == "form":
            self.forms += 1
            action = attrd.get("action", "").strip()
            if action:
                self.form_actions.append(action)

        if tag == "input":
            input_type = attrd.get("type", "").strip().lower()
            if input_type == "password":
                self.password_inputs += 1
            name_value = (attrd.get("name", "") + " " + attrd.get("id", "")).lower()
            if any(word in name_value for word in ("pass", "pwd", "card", "otp", "cvv", "ssn")):
                self.sensitive_inputs += 1

        if tag == "a" and attrd.get("href"):
            self.links.append(attrd["href"].strip())

        if tag == "iframe" and attrd.get("src"):
            self.iframes.append(attrd["src"].strip())

        if tag == "img" and attrd.get("src", "").startswith("http://"):
            self.img_without_https += 1

        if tag == "meta" and attrd.get("http-equiv", "").lower() == "refresh":
            if self.meta_refresh is None:
                self.meta_refresh = attrd.get("content", "").strip()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("script", "style"):
            self.in_skip = max(self.in_skip - 1, 0)
        elif tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_skip:
            return
        if self.in_title:
            self.title_parts.append(data)
        else:
            stripped = data.strip()
            if stripped:
                self.visible_text_parts.append(stripped)


def _ip_is_unsafe(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for loopback / private / link-local / metadata / special ranges."""
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local          # 169.254/16 incl. cloud metadata
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    if ip.version == 4:
        if ip in ipaddress.ip_network("100.64.0.0/10"):  # CGNAT
            return True
        if ip in ipaddress.ip_network("198.18.0.0/15"):  # benchmarking
            return True
        if ip == ipaddress.IPv4Address("255.255.255.255"):  # broadcast
            return True
        if ip == ipaddress.IPv4Address("169.254.169.254"):  # cloud metadata
            return True
    return False


def _hostname_flags(host: str) -> list[str]:
    """Homoglyph/IDN deception signals for a hostname (informational; the
    risk engine turns them into a medium finding so punycode never passes
    silently as a trusted-looking domain)."""
    flags: list[str] = []
    if any(label.startswith("xn--") for label in host.split(".")):
        flags.append("punycode_host")
    if any(ch > "\x7f" for ch in host):
        flags.append("non_ascii_host")
    return flags


def _validate_scheme_url(url: str) -> tuple[str, str | None, int]:
    """Return (scheme, hostname, port) or raise ValueError for non-web URLs."""
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"non_web_scheme:{scheme or 'missing'}")
    if parts.username or parts.password:
        raise ValueError("embedded_credentials")
    try:
        port = parts.port  # may raise ValueError on bad port
    except ValueError:
        raise ValueError("invalid_port")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host or " " in host:
        raise ValueError("invalid_hostname")
    return scheme, host, 443 if (port is None and scheme == "https") else 80 if port is None else port


def _validate_destination(host: str, port: int) -> dict:
    """
    Resolve and validate a hostname. Returns {"ips": [...]} where every IP
    is confirmed safe, or {"error": code} when the destination must not be
    contacted.
    """
    # Literal IP hostname goes straight to the safety check.
    try:
        literal = ipaddress.ip_address(host)
        if _ip_is_unsafe(literal):
            return {"error": f"unsafe_destination:{host}"}
        return {"ips": [str(literal)]}
    except ValueError:
        pass  # ordinary hostname -> DNS resolution below

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return {"error": "dns_resolution_failed"}
    except OSError:
        # Resolver/network failure that is not a name-resolution result
        # (e.g. platform socket errors): treat as unreachable, never crash.
        return {"error": "dns_resolution_failed"}

    seen: list[str] = []
    safe_ips: list[str] = []
    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip_value = sockaddr[0]
        if ip_value in seen:
            continue
        seen.append(ip_value)
        try:
            ip = ipaddress.ip_address(ip_value)
        except ValueError:
            continue
        if _ip_is_unsafe(ip):
            return {"error": f"unsafe_destination:{ip_value}"}
        safe_ips.append(ip_value)

    if not safe_ips:
        return {"error": "no_resolvable_address"}
    return {"ips": safe_ips}


class _HTTPSConnection(http.client.HTTPSConnection):
    """https connection pinned to a pre-validated IP address."""

    def __init__(self, hostname, pinned_ip, connect_timeout, read_timeout):
        super().__init__(hostname, timeout=connect_timeout)
        self._pinned_ip = pinned_ip
        self._read_timeout = read_timeout
        self._tls_info: dict = {}

    def connect(self):
        sock = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout
        )
        sock.settimeout(self._read_timeout)
        context = ssl.create_default_context()
        self.sock = context.wrap_socket(sock, server_hostname=self.host)
        try:
            cipher = self.sock.cipher()
            tls_version = self.sock.version()
        except (ValueError, OSError):
            cipher, tls_version = None, None
        self._tls_info = {
            "tls_version": tls_version,
            "cipher": cipher[0] if cipher else None,
        }
        try:
            # Capture the certificate immediately: with `Connection: close`
            # the base class nulls self.sock as soon as the response begins.
            self._tls_info["peer_cert"] = self.sock.getpeercert() or {}
        except (ValueError, OSError):
            self._tls_info["peer_cert"] = {}


class _HTTPConnection(http.client.HTTPConnection):
    """http connection pinned to a pre-validated IP address."""

    def __init__(self, hostname, pinned_ip, connect_timeout, read_timeout):
        super().__init__(hostname, timeout=connect_timeout)
        self._pinned_ip = pinned_ip
        self._read_timeout = read_timeout
        self._tls_info = {}

    def connect(self):
        sock = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout
        )
        sock.settimeout(self._read_timeout)
        self.sock = sock


def _read_body_max(response, max_bytes: int) -> tuple[bytes, bool, int | None]:
    """Read up to max_bytes. Returns (payload, truncated, error_code|None).
    On a mid-read timeout the bytes received so far are kept and marked
    truncated so inspection stays honest about partial content."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    error_code: int | None = None
    while True:
        try:
            chunk = response.read(65_536)
        except (socket.timeout, TimeoutError, ssl.SSLError, OSError):
            error_code = "body_read_timeout"
            truncated = True
            break
        if not chunk:
            break
        total += len(chunk)
        if total >= max_bytes:
            chunks.append(chunk[: max_bytes - (total - len(chunk))])
            truncated = True
            break
        chunks.append(chunk)
    return b"".join(chunks), truncated, error_code


_META_REFRESH_TARGET_RE = re.compile(r"url=(.*?)\s*$", re.IGNORECASE)


def _parse_meta_refresh_target(content: str) -> str | None:
    match = _META_REFRESH_TARGET_RE.search(content)
    if match:
        raw = match.group(1).strip().strip("'\"")
        return raw or None
    return None


def _decode_body(payload: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", content_type or "", re.IGNORECASE)
    if match:
        charset = match.group(1)
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _looks_like_html(body: bytes, content_type: str) -> bool:
    if content_type and "html" in content_type.lower():
        return True
    head = body[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<body" in head[:512]


def _is_executable_content(content_type: str, disposition: str) -> bool:
    lowered = (content_type or "").lower()
    if any(marker in lowered for marker in _EXEC_CONTENT_TYPES):
        return True
    if "attachment" in (disposition or "").lower():
        return True
    return False


def inspect_website(url: str) -> dict:
    """
    Inspect one URL. Returns a JSON-friendly dict; NEVER raises for
    ordinary failures (network, DNS, TLS, unsafe destinations).
    """
    started = time.monotonic()
    result: dict = {
        "requested_url": url,
        "reachable": False,
        "blocked": None,          # unsafe-destination or input-refusal code
        "error": None,            # transport/fetch failure code
        "availability": "UNKNOWN",
        "title_available": False,
        "availability_detail": {"status_code": None, "title_available": False, "availability": "UNKNOWN"},
        "hops": [],
        "final_url": None,
        "status_code": None,
        "content_type": None,
        "is_html": False,
        "served_executable": False,
        "content_truncated": False,
        "tls": None,
        "dns": None,
        "host_flags": [],
        "redirect_scheme_downgrade": False,
        "page": {},
        "elapsed_ms": None,
    }

    # -- input validation -------------------------------------------------
    try:
        scheme, host, port = _validate_scheme_url(url)
    except ValueError as exc:
        code = str(exc)
        if code.startswith("non_web_scheme"):
            result["blocked"] = "non_http_url"
        else:
            result["blocked"] = code
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
        return result

    current_url = url
    reused_tls: dict | None = None
    reused_dns: dict | None = None

    for hop_index in range(MAX_REDIRECTS + 1):
        # Overall deadline check.
        if time.monotonic() - started > DEFAULT_TOTAL_BUDGET:
            result["error"] = "inspection_timeout"
            result["availability"] = "TIMEOUT"
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
            return result

        try:
            scheme, host, port = _validate_scheme_url(current_url)
        except ValueError as exc:
            code = str(exc)
            if code == "embedded_credentials":
                result["error"] = "redirect_to_credentials"
            elif code.startswith("non_web_scheme"):
                result["error"] = "redirect_to_non_http"
            else:
                result["error"] = code
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
            return result

        if hop_index > 0 and scheme == "http" and reused_tls == {"https": True}:
            result["redirect_scheme_downgrade"] = True

        # -- destination validation (EVERY hop) ---------------------------
        dest = _validate_destination(host, port)
        reused_dns = {"host": host, "validated_ips": dest.get("ips", []), "error": dest.get("error")}
        result["host_flags"] = _hostname_flags(host)
        if dest.get("error"):
            result["blocked"] = dest["error"]
            if "dns" in dest["error"]:
                result["availability"] = "DNS_FAILED"
            result["dns"] = reused_dns
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
            return result

        parts = urlsplit(current_url)
        path_and_query = parts.path or "/"
        if parts.query:
            path_and_query += "?" + parts.query

        # -- connect (try every validated address until one answers) --------
        response = None
        conn = None
        last_failure: dict = {}
        for pinned_ip in dest["ips"]:
            try:
                if scheme == "https":
                    conn = _HTTPSConnection(host, pinned_ip, DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
                else:
                    conn = _HTTPConnection(host, pinned_ip, DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
                conn.request(
                    "GET",
                    path_and_query if path_and_query.startswith("/") else "/" + path_and_query,
                    headers={
                        "Host": host if port in (80, 443) else f"{host}:{port}",
                        "User-Agent": USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml;q=0.8,*/*;q=0.5",
                        "Accept-Language": "en",
                        "Connection": "close",
                    },
                )
                response = conn.getresponse()
                break
            except socket.timeout:
                last_failure = {"error": "connection_timeout", "availability": "TIMEOUT"}
            except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                ssl_was_the_cause = (
                    isinstance(exc, ssl.SSLError)
                    or "certificate" in str(exc).lower()
                    or "ssl" in str(exc).lower()
                )
                if ssl_was_the_cause:
                    # TLS-level failures are host-specific: other addresses
                    # of the same hostname should not silently retry the
                    # certificate handshake.
                    last_failure = {
                        "error": "tls_certificate_error",
                        "availability": "CONNECTION_FAILED",
                        "ssl_error": str(exc),
                    }
                    break
                last_failure = {
                    "error": "connection_failed",
                    "availability": "CONNECTION_FAILED",
                    "ssl_error": str(exc),
                }
        if response is None:
            result["error"] = last_failure.get("error") or "connection_failed"
            result["availability"] = last_failure.get("availability") or "CONNECTION_FAILED"
            if last_failure.get("ssl_error"):
                result["ssl_error"] = last_failure["ssl_error"]
            result["dns"] = reused_dns
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
            return result

        # -- TLS / certificate facts -----------------------------------------
        hop_record = {
            "url": current_url,
            "status": response.status,
            "ip": pinned_ip,
            "location": None,
        }
        result["hops"].append(hop_record)

        if scheme == "https":
            tls_info = getattr(conn, "_tls_info", {}) or {}
            cert = tls_info.get("peer_cert") or {}
            not_after = cert.get("notAfter")
            subject = cert.get("subject") or []
            issuer = cert.get("issuer") or []
            sans = cert.get("subjectAltName") or []
            subject_cn = next((v for rdn in subject for k, v in rdn if k == "commonName"), None)
            if not subject_cn:
                # Modern certificates are often SAN-only (no commonName).
                subject_cn = next(
                    (value for kind, value in sans if kind == "DNS"), None
                )
            issuer_org = next((v for rdn in issuer for k, v in rdn if k == "organizationName"), None)
            result["tls"] = {
                "handshake": "ok",
                "tls_version": tls_info.get("tls_version"),
                "cipher": tls_info.get("cipher"),
                "subject_cn": subject_cn,
                "issuer": issuer_org,
                "not_after": not_after,
                "hostname": host,
            }
            reused_tls = {"https": True}
        else:
            reused_tls = {"https": False}

        # -- redirects ---------------------------------------------------------
        status = response.status
        location = response.getheader("Location")
        content_type = (response.getheader("Content-Type") or "").strip()
        disposition = response.getheader("Content-Disposition") or ""
        content_length = response.getheader("Content-Length")

        if status in (301, 302, 303, 307, 308) and location:
            next_url = urljoin(current_url, location.strip())
            hop_record["location"] = next_url
            result["final_url"] = next_url
            current_url = next_url
            try:
                next_scheme, _nh, _np = _validate_scheme_url(next_url)
            except ValueError:
                result["error"] = "redirect_to_non_http"
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
                return result
            if scheme == "https" and next_scheme == "http":
                result["redirect_scheme_downgrade"] = True
            if hop_index == MAX_REDIRECTS:
                result["error"] = "too_many_redirects"
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
                return result
            continue

        # -- final response: body ---------------------------------------------
        result["reachable"] = True
        result["final_url"] = current_url
        result["status_code"] = status
        result["content_type"] = content_type
        result["dns"] = reused_dns
        executable = _is_executable_content(content_type, disposition)
        result["served_executable"] = executable

        # All final statuses: read the body — error pages, JSON APIs, and
        # other non-HTML responses still carry evidence (recorded as such)
        # and must never block URL-feature-based classification.
        payload, truncated, body_error = _read_body_max(response, MAX_BODY_BYTES)
        result["content_truncated"] = truncated
        if body_error:
            result["body_error"] = body_error
        if not executable and _looks_like_html(payload, content_type):
            html = _decode_body(payload, content_type)
            parser = _PageParser()
            try:
                parser.feed(html)
                parser.close()
            except Exception:  # malformed HTML tolerated
                pass
            title_text = " ".join(" ".join(parser.title_parts).split())
            visible_text = " ".join(parser.visible_text_parts)
            result["is_html"] = True
            result["page"] = {
                "title": title_text[:300],
                "text_length": len(visible_text),
                "text_sample": visible_text[:MAX_TEXT_SAMPLE],
                "forms": parser.forms,
                "form_actions": parser.form_actions[:10],
                "password_fields": parser.password_inputs,
                "sensitive_inputs": parser.sensitive_inputs,
                "links": len(parser.links),
                "external_scripts": parser.external_scripts,
                "iframes": parser.iframes[:10],
                "meta_refresh": _parse_meta_refresh_target(parser.meta_refresh) if parser.meta_refresh else None,
                "http_images": parser.img_without_https,
            }
        elif executable:
            result["is_html"] = False
            result["page"] = {
                "title": None,
                "text_length": 0,
                "text_sample": "",
                "forms": 0,
                "form_actions": [],
                "password_fields": 0,
                "sensitive_inputs": 0,
                "links": 0,
                "external_scripts": 0,
                "iframes": [],
                "meta_refresh": None,
                "http_images": 0,
                "note": "Executable/binary download refused for inspection (never downloaded further).",
            }
        else:
            text = _decode_body(payload[:16_000], content_type)
            result["is_html"] = False
            result["page"] = {
                "title": None,
                "text_length": len(text),
                "text_sample": text[:MAX_TEXT_SAMPLE],
                "forms": 0,
                "form_actions": [],
                "password_fields": 0,
                "sensitive_inputs": 0,
                "links": 0,
                "external_scripts": 0,
                "iframes": [],
                "meta_refresh": None,
                "http_images": 0,
            }
        # Availability is a pure transport/status fact, independent of
        # content: never a security claim.
        result["error"] = f"http_status_{status}" if status not in (200, 201, 202, 203, 204, 206, 403, 405, 406, 410) else result["error"]
        result["title_available"] = bool((result.get("page") or {}).get("title"))
        if result.get("body_error") == "body_read_timeout":
            result["availability"] = "TIMEOUT"
        elif status in (404, 410):
            result["availability"] = "PAGE_NOT_FOUND"
        elif result["title_available"]:
            result["availability"] = "REACHABLE"
        else:
            result["availability"] = "TITLE_UNAVAILABLE"

        result["availability_detail"] = {
            "status_code": status,
            "title_available": result["title_available"],
            "availability": result["availability"],
        }

        conn.close()
        break

    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
    return result


def _host_is_official_brand(brand: str, host_l: str) -> bool:
    """True when the hostname belongs to the brand's real namespace.

    Exact match OR a dot-anchored subdomain of an official host, e.g.
    www.amazon.com ends-with ".amazon.com" (only the real Amazon controls
    DNS under amazon.com). Lookalikes can never match this way:
    "paypal.com.evil.xyz" does not end with ".paypal.com".
    """
    for official in BRAND_OFFICIAL_HOSTS.get(brand, frozenset()):
        if host_l == official or host_l.endswith("." + official):
            return True
    return False


def extract_brand_claim(page: dict, host: str | None) -> str | None:
    """
    Return a brand name whose wording appears in the page title while the
    hostname is NOT that brand's official domain namespace (impersonation
    hint). Purely raises suspicion; never approves a site.

    Path slugs such as "microsoft/vscode" style owner/repo references in
    titles are stripped first so a GitHub-hosted repository is not flagged
    for merely mentioning its owner's name.
    """
    title = page.get("title") or ""
    # Remove owner/repo path mentions and bare path fragments of a brand.
    title = re.sub(r"\b[a-z0-9_-]+/[a-z0-9_.\-]+", " ", title, flags=re.IGNORECASE)
    title_l = title.lower()
    host_l = (host or "").lower()
    for brand, official in BRAND_OFFICIAL_HOSTS.items():
        if brand in title_l and not _host_is_official_brand(brand, host_l):
            return brand
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FraudShield website inspector")
    parser.add_argument("url", help="URL to inspect (read-only)")
    args = parser.parse_args()
    print(json.dumps(inspect_website(args.url), indent=2, ensure_ascii=False))
