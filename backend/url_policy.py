"""
FraudShield - trusted URL policy layer
======================================

This module is a deterministic STRUCTURAL verification layer that runs
AFTER the R2 model scoring. It can only ever *downgrade* a malicious
verdict to SAFE -- it can never upgrade a benign verdict or change a raw
model score.

Design rules (hard constraints):
    * Exact hostname whitelisting. Suffix/substring matches are forbidden,
      so "youtube.com.evil-example.com" or "youtube-login.example.com"
      can never be trusted.
    * HTTPS is mandatory. HTTP trusted-platform URLs are NOT approved and
      fall back to the raw model verdict.
    * Embedded credentials ("user:pass@host"), explicit non-default ports,
      IP hosts, and percent-encoded paths are never approved.
    * Only narrowly defined URL kinds are approved (homepage, /watch,
      /playlist, /shorts, youtu.be short links), and their query
      parameters are name- and value-validated against a strict allowlist.
      Redirect endpoints (/redirect) and unknown parameters are never
      approved.
    * The raw R2 model output is always preserved and reported alongside
      the final policy-adjusted verdict.

This layer is presentation/policy logic only. It never retrains,
re-thresholds, or re-weights the model, and performs no network access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlsplit

# Exact, case-insensitive trusted hostnames. LOOKALIKE REJECTION RULE:
# membership must match the FULL hostname after stripping a single
# trailing dot only. No wildcard, suffix, or parent/child matching.
TRUSTED_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
        "youtu.be",
        "www.youtu.be",
    }
)

# Instagram: exact hosts only. Lookalikes such as
# "instagram.com.evil.com" or "instagram-login.example.com" never match.
TRUSTED_INSTAGRAM_HOSTS = frozenset(
    {
        "instagram.com",
        "www.instagram.com",
        "m.instagram.com",
        "l.instagram.com",
        "help.instagram.com",
        "ig.me",
        "www.ig.me",
    }
)

_STANDARD_PORTS = frozenset({None, 443})

# Compact canonical ID shapes observed on YouTube.
_PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,80}$")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_SHORTS_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_SI_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_FEATURE_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_APP_RE = re.compile(r"^[a-z]{1,16}$")
_INDEX_RE = re.compile(r"^[0-9]{1,5}$")
_START_RADIO_RE = re.compile(r"^[01]$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{1,10}[smh]?$")
_SEARCH_IS_RE = re.compile(r"^[a-z0-9_+-]{1,32}$")

# Instagram shapes.
_IG_PICTURE_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{5,64}$")   # /p/, /reel/, /reels/, /tv/
_IG_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{1,30}$")       # profile handle
_IG_SHARE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")    # /shares/<token>
_IG_STORY_ID_RE = re.compile(r"^[0-9]{1,15}$")               # /stories/<user>/<id>
_IG_SHARE_PARAM_RE = re.compile(r"^[A-Za-z0-9_-]{2,96}$")    # igsh / igshid values

# Optional non-required parameter names per URL kind. ANY other name is
# rejected ("deceptive query parameters must not be automatically
# approved"). Every allowed name is validated with the regexes above.
_COMMON_PARAMS = ("si", "app", "feature")

_PARAM_VALIDATORS: dict[str, re.Pattern[str]] = {
    "si": _SI_RE,
    "app": _APP_RE,
    "feature": _FEATURE_RE,
    "v": _VIDEO_ID_RE,
    "list": _PLAYLIST_ID_RE,
    "index": _INDEX_RE,
    "start_radio": _START_RADIO_RE,
    "t": _TIMESTAMP_RE,
    "search_query": _SEARCH_IS_RE,
    "start": _TIMESTAMP_RE,
    "rel": _START_RADIO_RE,
    "hl": re.compile(r"^[a-zA-Z-]{1,16}$"),
    "igsh": _IG_SHARE_PARAM_RE,
    "igshid": _IG_SHARE_PARAM_RE,
}

_POLICY_NOTE_APPROVED_FORMAT = (
    "Trusted {platform} URL structure verified: exact https hostname, "
    "no embedded credentials or IP host, recognized path, and validated "
    "query parameters. Structural verification overrides the raw model "
    "score for this verdict."
)


@dataclass(frozen=True)
class PolicyResult:
    """Outcome of the structural policy check for one URL."""

    platform: str | None
    """'youtube' when the exact trusted hostname matched, else None."""

    url_kind: str | None
    """homepage | watch | playlist | shorts | short_link when matched."""

    approved: bool
    rejection: str | None

    @property
    def is_trusted_platform(self) -> bool:
        return self.platform is not None

    @property
    def note(self) -> str:
        platform_name = (self.platform or "platform").capitalize()
        return _POLICY_NOTE_APPROVED_FORMAT.format(platform=platform_name)


def _reject(kind: str | None, reason: str, platform: str = "youtube") -> PolicyResult:
    return PolicyResult(platform=platform, url_kind=kind, approved=False, rejection=reason)


def _approve(kind: str, platform: str = "youtube") -> PolicyResult:
    return PolicyResult(platform=platform, url_kind=kind, approved=True, rejection=None)


def _check_params(
    kind: str,
    pairs: list[tuple[str, str]],
    required: dict[str, re.Pattern[str]],
    optional: tuple[str, ...],
    platform: str = "youtube",
) -> PolicyResult:
    """Validate one query string against a strict per-kind allowlist."""
    names = [name for name, _value in pairs]
    if len(set(names)) != len(names):
        return _reject(kind, "duplicate_query_parameter", platform)

    for name, value in pairs:
        if name not in required and name not in optional:
            return _reject(kind, "unexpected_query_parameter", platform)
        validator = _PARAM_VALIDATORS.get(name)
        if validator is None or not value or not validator.match(value):
            return _reject(kind, f"invalid_parameter_value:{name}", platform)

    for name, validator in required.items():
        actual = next((value for param, value in pairs if param == name), "")
        if not isinstance(actual, str) or not validator.match(actual):
            # Distinguish mild malformed cases from deceptive redirects.
            return _reject(kind, f"missing_or_invalid:{name}", platform)

    return _approve(kind, platform)


def evaluate_trusted_url_policy(url: str) -> PolicyResult:
    """
    Verify one URL string against the trusted-platform structural rules.

    Returns a PolicyResult. ``platform is None`` means "not a trusted
    platform URL": the caller must keep the raw model verdict unchanged.
    ``approved=True`` means the caller may safely downgrade the verdict
    to SAFE. Anything else means "trusted hostname but unapproved
    structure": keep the raw model verdict and surface the rejection.
    """
    if not isinstance(url, str) or not url.strip():
        return PolicyResult(None, None, False, None)

    text = url.strip()

    # Rule 0: HTTPS scheme MUST be explicit. Bare "youtube.com/x" input
    # proves nothing about transport security and is never approved.
    if not text.lower().startswith("https://"):
        # Hostname lookalikes still count as "trusted-platform-looking"
        # only when the host itself matches; a plain http youtube.com is
        # checked below. Without //, urlsplit cannot parse reliably for
        # policy purposes, so treat bare input as its own host check.
        split_candidate = text if "://" in text else None
        if split_candidate is None:
            bare = text.split("/")[0].split("?")[0].lower().rstrip(".")
            if bare in TRUSTED_YOUTUBE_HOSTS:
                return _reject("homepage", "https_required", "youtube")
            if bare in TRUSTED_INSTAGRAM_HOSTS:
                return _reject("homepage", "https_required", "instagram")
            return PolicyResult(None, None, False, None)

    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower().rstrip(".")
        port = parts.port  # May raise ValueError on invalid port syntax.
    except ValueError:
        return _reject(None, "unparseable_url")

    # Rule 1: exact hostname whitelist only.
    is_youtube_host = host in TRUSTED_YOUTUBE_HOSTS
    is_instagram_host = host in TRUSTED_INSTAGRAM_HOSTS
    if not is_youtube_host and not is_instagram_host:
        return PolicyResult(None, None, False, None)

    platform = "youtube" if is_youtube_host else "instagram"

    # Rule 2: HTTPS only.
    if (parts.scheme or "").lower() != "https":
        return _reject(None, "https_required", platform)

    # Rule 3: standard port only (443 or no explicit port).
    if port not in _STANDARD_PORTS:
        return _reject(None, "non_standard_port", platform)

    # Rule 4: no embedded credentials, ever.
    if parts.username or parts.password:
        return _reject(None, "embedded_credentials", platform)

    # Rule 5: decoded path must equal the raw path (blocks %-encoding
    # tricks such as "/playlist%3Flist=" or "/watch%2Fx").
    raw_path = parts.path or "/"
    if "%" in raw_path and unquote(raw_path) != raw_path:
        return _reject(None, "percent_encoded_path", platform)

    path = raw_path.rstrip("/") if raw_path != "/" else raw_path
    pairs: list[tuple[str, str]] = []
    if parts.query:
        try:
            pairs = list(parse_qsl(parts.query, keep_blank_values=True))
        except ValueError:
            return _reject(None, "unparseable_query", platform)

    is_youtu_be = host in ("youtu.be", "www.youtu.be")
    is_cookie_host = host in ("youtube-nocookie.com", "www.youtube-nocookie.com")

    # ------------------------------------------------------------------
    # Instagram structural kinds (exact hosts validated above).
    # Runs BEFORE the YouTube checks so its own homepage/profile kinds
    # can never be absorbed by generic YouTube path rules.
    # ------------------------------------------------------------------
    if is_instagram_host:
        if path in ("", "/"):
            return _check_params(
                "homepage", pairs, required={}, optional=("hl",), platform=platform
            )

        media_match = re.match(r"^/(p|reel|reels|tv)/([A-Za-z0-9_-]+)$", path)
        if media_match and _IG_PICTURE_CODE_RE.match(media_match.group(2)):
            kind = "reel" if media_match.group(1) in ("reel", "reels") else "post"
            return _check_params(
                kind,
                pairs,
                required={},
                optional=("igsh", "igshid", "hl"),
                platform=platform,
            )

        share_match = re.match(r"^/shares/([A-Za-z0-9_.-]+)$", path)
        if share_match and _IG_SHARE_TOKEN_RE.match(share_match.group(1)):
            return _check_params(
                "share",
                pairs,
                required={},
                optional=("igsh", "igshid", "hl"),
                platform=platform,
            )

        story_match = re.match(r"^/stories/([A-Za-z0-9_.]+)/([0-9]{1,15})$", path)
        if story_match and _IG_USERNAME_RE.match(story_match.group(1)):
            return _check_params(
                "story",
                pairs,
                required={},
                optional=("igsh", "igshid"),
                platform=platform,
            )

        # A single path segment is an Instagram profile handle.
        segment = path.lstrip("/")
        if "/" not in segment and _IG_USERNAME_RE.match(segment):
            return _check_params(
                "profile",
                pairs,
                required={},
                optional=("igsh", "igshid", "hl"),
                platform=platform,
            )

        # Anything else on an Instagram hostname (e.g. /accounts/, unknown
        # deep paths with arbitrary parameters) is NOT auto-approved.
        return _reject(None, "unrecognized_path", platform)

    # Rule 6: per-type path/param validation (YouTube) ---------------------
    if is_youtu_be:
        # Short link form: https://youtu.be/<11-char video id>
        video_id = path[1:] if path.startswith("/") else path
        if _VIDEO_ID_RE.match(video_id):
            return _check_params(
                "short_link",
                pairs,
                required={},
                optional=("t", "si", "feature", "list", "start_radio", "index"),
            )

    if path in ("", "/"):
        return _check_params(
            "homepage",
            pairs,
            required={},
            optional=_COMMON_PARAMS + ("hl",),
        )

    if path == "/playlist" or (is_cookie_host and path == "/playlist"):
        return _check_params(
            "playlist",
            pairs,
            required={"list": _PLAYLIST_ID_RE},
            optional=("v", "index", "start_radio", *_COMMON_PARAMS),
        )

    if path == "/watch":
        return _check_params(
            "watch",
            pairs,
            required={"v": _VIDEO_ID_RE},
            optional=("t", "list", "index", "start_radio", *_COMMON_PARAMS),
        )

    shorts_match = re.match(r"^/shorts/([A-Za-z0-9_-]+)$", path)
    if shorts_match and _SHORTS_ID_RE.match(shorts_match.group(1)):
        return _check_params(
            "shorts",
            pairs,
            required={},
            optional=_COMMON_PARAMS,
        )

    # /embed/<id> for nocookie or www hosts.
    embed_match = re.match(r"^/embed/([A-Za-z0-9_-]+)$", path)
    if embed_match and _VIDEO_ID_RE.match(embed_match.group(1)):
        return _check_params(
            "embed",
            pairs,
            required={},
            optional=_COMMON_PARAMS + ("start", "rel"),
        )

    # Anything else on a trusted hostname (e.g. /redirect, /results with
    # arbitrary values, /account...) is NOT auto-approved: the raw model
    # verdict stands.
    return _reject(None, "unrecognized_path", platform)


def policy_approval_reason(platform: str | None, url_kind: str | None) -> str:
    """Human-readable explanation used in the API 'reasons' list."""
    platform_name = (platform or "trusted").capitalize()
    kind = url_kind or "trusted"
    return (
        f"Trusted-platform URL structure verified ({platform_name} {kind} format "
        "via exact https hostname; no credentials, IP host, redirect trick, or "
        "unvalidated parameters). Policy layer overrides the raw model score "
        "for this verdict."
    )
