"""Application configuration.

All knobs are env-driven via pydantic-settings, so the same image runs
unchanged in dev, CI, and on the deployed Render instance. Defaults are tuned
for local development.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = Field(default="ai-based-server", alias="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", alias="SERVICE_VERSION")
    env: Literal["development", "production", "test"] = Field(
        default="development", alias="ENV"
    )
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Model ────────────────────────────────────────────────────────────
    model_path: Path = Field(
        default=Path("./models/isolation_forest.joblib"), alias="MODEL_PATH"
    )
    anomaly_threshold: float = Field(default=0.55, alias="ANOMALY_THRESHOLD")

    # Simulated inference latency (used to study the latency-accuracy
    # trade-off without retraining a heavier model).
    simulated_latency_ms: float = Field(default=0.0, alias="SIMULATED_LATENCY_MS")
    simulated_latency_jitter_ms: float = Field(
        default=0.0, alias="SIMULATED_LATENCY_JITTER_MS"
    )

    @property
    def cors_origins_list(self) -> list[str] | str:
        """CORS origins as either a list or the wildcard string."""
        if self.cors_origins.strip() == "*":
            return "*"
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor; safe to call from request paths."""
    return Settings()
