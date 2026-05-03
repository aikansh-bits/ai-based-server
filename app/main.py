"""FastAPI application factory and uvicorn entrypoint.

Run with:

    python -m app.main          # native
    uvicorn app.main:app        # via uvicorn
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.logging_config import configure_logging
from app.routers.detect import router as detect_router
from app.routers.system import router as system_router
from app.services.detector import DetectorService

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wire up the detector and any other singletons at startup."""
    settings = get_settings()
    detector = DetectorService(
        model_path=settings.model_path,
        threshold=settings.anomaly_threshold,
        simulated_latency_ms=settings.simulated_latency_ms,
        simulated_latency_jitter_ms=settings.simulated_latency_jitter_ms,
    )
    detector.load()
    app.state.settings = settings
    app.state.detector = detector
    log.info(
        "service_started",
        extra={
            "service": settings.service_name,
            "version": settings.service_version,
            "env": settings.env,
            "model_loaded": detector.model_loaded,
            "threshold": detector.threshold,
            "package_version": __version__,
        },
    )
    yield
    log.info("service_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, dev=not settings.is_prod)

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if origins == "*" else origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id", "x-inference-ms"],
    )

    @app.middleware("http")
    async def correlation_and_timing(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_exception", extra={"request_id": rid, "path": request.url.path})
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error", "request_id": rid},
                headers={"x-request-id": rid},
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["x-request-id"] = rid
        response.headers["x-inference-ms"] = f"{elapsed_ms:.3f}"
        if not _is_noisy(request.url.path):
            log.info(
                "request",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 3),
                },
            )
        return response

    app.include_router(system_router)
    app.include_router(detect_router)

    return app


def _is_noisy(path: str) -> bool:
    return path in {"/", "/health", "/ready", "/info", "/openapi.json"} or path.startswith(
        "/docs"
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        reload=not settings.is_prod,
    )
