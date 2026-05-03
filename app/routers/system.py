"""System endpoints: /health, /ready, /version, /info."""

from __future__ import annotations

import os
import platform
import time

from fastapi import APIRouter, Request

from app import __version__
from app.schemas import HealthResponse, InfoResponse, ReadyResponse

router = APIRouter()
_STARTED_AT = time.monotonic()


@router.get("/", include_in_schema=False)
def root(request: Request) -> dict:
    return {
        "service": request.app.state.settings.service_name,
        "version": request.app.state.settings.service_version,
        "docs": ["/docs", "/health", "/ready", "/info", "/detect"],
    }


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    s = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=s.service_name,
        version=s.service_version,
        uptime_sec=round(time.monotonic() - _STARTED_AT, 3),
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    s = request.app.state.settings
    detector = request.app.state.detector
    # We are "ready" as long as the process is up; the model may be missing
    # (in which case we fall back to the heuristic scorer). The flag is
    # surfaced so deployers can choose to fail-fast on a missing artefact.
    status = "ready" if detector.model_loaded else "degraded"
    return ReadyResponse(
        status=status,
        service=s.service_name,
        version=s.service_version,
        model_loaded=detector.model_loaded,
        model_path=str(detector.model_path),
        threshold=detector.threshold,
    )


@router.get("/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    s = request.app.state.settings
    detector = request.app.state.detector
    return InfoResponse(
        service=s.service_name,
        version=s.service_version,
        env=s.env,
        model={
            **detector.model_metadata,
            "python": platform.python_version(),
            "pid": os.getpid(),
            "host": platform.node(),
            "package_version": __version__,
        },
    )
