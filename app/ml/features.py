"""Feature engineering for API request anomaly detection.

The features below are picked to be (a) cheap to compute, (b) interpretable
in the dissertation write-up, and (c) actually predictive of the abuse
patterns produced by the rule-based-server's mock attack traffic.

Every feature is numeric; the order in `FEATURE_ORDER` is the contract
between the trainer and the runtime scorer — do not reorder without
retraining.
"""

from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import unquote

from app.schemas import DetectRequest, FeatureBreakdown

FEATURE_ORDER: list[str] = [
    "request_count_1min",
    "request_count_burst",
    "distinct_paths",
    "content_length",
    "path_length",
    "query_key_count",
    "body_key_count",
    "suspicious_keyword_score",
    "user_agent_risk",
    "method_risk",
]

# Keywords that frequently appear in attack payloads. Each match contributes
# 1.0 to the keyword score (saturating so that one extreme payload doesn't
# dominate; see `_keyword_score`).
_SUSPICIOUS_KEYWORDS = (
    re.compile(r"\bunion\b\s+\bselect\b", re.I),
    re.compile(r"\bor\b\s*\d+\s*=\s*\d+", re.I),
    re.compile(r"['\"]\s*or\s*['\"]?\s*\d+", re.I),
    re.compile(r"<script\b", re.I),
    re.compile(r"javascript:", re.I),
    re.compile(r"\bon\w+\s*=", re.I),
    re.compile(r"\.\./"),
    re.compile(r"%2e%2e%2f", re.I),
    re.compile(r"/etc/passwd", re.I),
    re.compile(r";\s*(rm|cat|ls|wget|curl|sh|bash)\b", re.I),
    re.compile(r"\$\([^)]+\)"),
    re.compile(r"`[^`]+`"),
    re.compile(r"\bsleep\s*\(\s*\d+\s*\)", re.I),
    re.compile(r"\bxp_cmdshell\b", re.I),
)

_BAD_USER_AGENT_PATTERNS = (
    re.compile(r"sqlmap", re.I),
    re.compile(r"nikto", re.I),
    re.compile(r"nmap", re.I),
    re.compile(r"masscan", re.I),
    re.compile(r"acunetix", re.I),
    re.compile(r"\bcurl/", re.I),
    re.compile(r"\bwget/", re.I),
    re.compile(r"python-requests", re.I),
    re.compile(r"go-http-client", re.I),
    re.compile(r"libwww-perl", re.I),
)

# Methods rarely seen in normal API traffic; we don't penalise heavily because
# legitimate clients do use PUT/PATCH/DELETE.
_RISKY_METHODS = {"TRACE": 1.0, "CONNECT": 1.0, "OPTIONS": 0.0}


def _safe_unquote(s: str) -> str:
    """URL-decode tolerantly. Attackers commonly use percent-encoding to
    obfuscate payloads (`%27 OR %271%27%3D%271`); detection must see through
    that. We unquote twice to handle double-encoding (a known WAF bypass)."""
    try:
        return unquote(unquote(s))
    except Exception:
        return s


def _keyword_score(haystack: Iterable[str]) -> float:
    """Saturating count of suspicious keyword matches across all strings.

    Each haystack entry is URL-decoded before matching so that percent-encoded
    payloads still trigger the patterns. A single match contributes a high
    baseline (0.7) because a single well-formed SQLi/XSS/traversal pattern is
    already strong evidence on its own; subsequent matches add diminishing
    returns up to a saturation of 1.0.
    """
    hits = 0
    for raw in haystack:
        if not raw:
            continue
        s = _safe_unquote(raw)
        for pat in _SUSPICIOUS_KEYWORDS:
            if pat.search(s):
                hits += 1
                break  # one hit per haystack entry is enough
    if hits == 0:
        return 0.0
    return min(0.7 + 0.15 * (hits - 1), 1.0)


def _user_agent_risk(ua: str | None) -> float:
    if not ua:
        return 1.0  # missing UA is itself suspicious
    if any(p.search(ua) for p in _BAD_USER_AGENT_PATTERNS):
        return 1.0
    if len(ua) < 8:
        return 0.6
    return 0.0


def _method_risk(method: str) -> float:
    return _RISKY_METHODS.get((method or "").upper(), 0.0)


def extract(req: DetectRequest) -> FeatureBreakdown:
    """Compute every feature for a single request.

    Each feature is non-negative; the model normalises internally so we don't
    have to carry a scaler artefact.
    """
    haystack = [
        req.path or "",
        req.endpoint or "",
        " ".join(req.query_keys),
        " ".join(req.body_keys),
        req.user_agent or "",
    ]

    return FeatureBreakdown(
        request_count_1min=float(req.history.requests_1min),
        request_count_burst=float(req.history.requests_burst),
        distinct_paths=float(req.history.distinct_paths),
        content_length=float(req.content_length),
        path_length=float(len(req.path or "")),
        query_key_count=float(len(req.query_keys)),
        body_key_count=float(len(req.body_keys)),
        suspicious_keyword_score=_keyword_score(haystack),
        user_agent_risk=_user_agent_risk(req.user_agent),
        method_risk=_method_risk(req.method),
    )


def to_vector(features: FeatureBreakdown) -> list[float]:
    """Project feature dict onto the canonical `FEATURE_ORDER` vector."""
    d = features.model_dump()
    return [float(d[k]) for k in FEATURE_ORDER]


def explain(features: FeatureBreakdown) -> list[str]:
    """Human-readable summary of the most-firing signals.

    Used by the analyser frontend to render *why* the AI flagged something,
    so reviewers don't see a black-box score.
    """
    notes: list[str] = []
    f = features
    if f.suspicious_keyword_score >= 0.4:
        notes.append("matches known attack-keyword patterns")
    if f.user_agent_risk >= 0.6:
        notes.append("suspicious or missing user-agent")
    if f.request_count_burst >= 15:
        notes.append("high burst rate from this IP")
    if f.request_count_1min >= 60:
        notes.append("high sustained rate from this IP")
    if f.distinct_paths >= 10:
        notes.append("scanning many endpoints")
    if f.path_length >= 80:
        notes.append("unusually long path")
    if f.content_length >= 50_000:
        notes.append("large payload")
    if f.method_risk >= 1.0:
        notes.append("rarely-used HTTP method")
    return notes
