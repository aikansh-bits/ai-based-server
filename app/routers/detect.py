"""POST /detect — score a single API request for abuse."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import DetectRequest, DetectResponse

router = APIRouter()


@router.post("/detect", response_model=DetectResponse, response_model_exclude_none=True)
async def detect(payload: DetectRequest, request: Request) -> DetectResponse:
    """Score a single request.

    The detector is attached to `app.state` at startup; we resolve it via the
    request to keep route handlers free of module-level globals (helpful for
    testability).
    """
    detector = request.app.state.detector
    return await detector.detect(payload)
