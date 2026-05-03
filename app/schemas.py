"""Wire schemas for the AI detection service.

These are the contracts the rule-based-server's `aiClient` speaks to. Keep
them small, explicit, and stable — adding new optional fields is fine, but
renaming or removing fields is a breaking change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryFeatures(BaseModel):
    """Per-IP rolling counters supplied by the upstream rule server.

    The AI server does not maintain its own state because:
      1. requests can be load-balanced across replicas, but the rule server
         is the canonical source of truth for per-IP behaviour;
      2. it keeps the AI server stateless and trivially horizontally
         scalable.
    """

    requests_1min: int = Field(default=0, ge=0, description="Total requests from this IP in the last minute.")
    requests_burst: int = Field(default=0, ge=0, description="Requests from this IP in the last burst window.")
    distinct_paths: int = Field(default=0, ge=0, description="Distinct paths visited recently.")


class DetectRequest(BaseModel):
    """Input to /detect.

    All fields except `path` are optional so that callers can iterate quickly
    without breaking the contract.
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str | None = None
    method: str = "GET"
    path: str = "/"
    endpoint: str | None = None
    ip: str | None = None
    user_agent: str | None = None
    content_length: int = Field(default=0, ge=0)
    has_body: bool = False
    query_keys: list[str] = Field(default_factory=list)
    body_keys: list[str] = Field(default_factory=list)
    history: HistoryFeatures = Field(default_factory=HistoryFeatures)


class FeatureBreakdown(BaseModel):
    """The numeric features that the model actually saw.

    Exposed so the analyser dashboard can show *why* a score was high without
    re-implementing feature extraction client-side.
    """

    request_count_1min: float
    request_count_burst: float
    distinct_paths: float
    content_length: float
    path_length: float
    query_key_count: float
    body_key_count: float
    suspicious_keyword_score: float
    user_agent_risk: float
    method_risk: float


class DetectResponse(BaseModel):
    """Output of /detect.

    `is_anomaly` is the binary decision the rule server uses; `score` is the
    calibrated [0, 1] anomaly probability — higher is more anomalous. The
    upstream `AI_SCORE_THRESHOLD` setting on the rule server can override the
    binary cut-off without retraining the model.
    """

    is_anomaly: bool
    score: float = Field(ge=0.0, le=1.0)
    label: Literal["anomaly", "normal"]
    threshold: float
    model: str
    model_version: str
    features: FeatureBreakdown
    explain: list[str] = Field(default_factory=list, description="Top contributing signals.")
    inference_ms: float
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    uptime_sec: float


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    service: str
    version: str
    model_loaded: bool
    model_path: str
    threshold: float


class InfoResponse(BaseModel):
    service: str
    version: str
    env: str
    model: dict[str, Any]
