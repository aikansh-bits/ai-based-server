"""Anomaly-detection orchestration.

Glues feature extraction, the calibrated Isolation Forest model, and the
heuristic safety net into a single `detect(...)` function consumed by the
HTTP layer.

Stateful concerns (model loading, threshold) live on the `DetectorService`
instance so they can be replaced/reloaded without bouncing the process and
so tests can construct a service with synthetic fixtures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path

from app.ml import features as feat
from app.ml import model as mdl
from app.schemas import DetectRequest, DetectResponse, FeatureBreakdown

log = logging.getLogger(__name__)


class DetectorService:
    def __init__(
        self,
        *,
        model_path: Path,
        threshold: float,
        simulated_latency_ms: float = 0.0,
        simulated_latency_jitter_ms: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.simulated_latency_ms = max(0.0, simulated_latency_ms)
        self.simulated_latency_jitter_ms = max(0.0, simulated_latency_jitter_ms)
        self._model: mdl.CalibratedModel | None = None
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load (or reload) the on-disk model artefact."""
        self._model = mdl.load(self.model_path)
        self._loaded = self._model is not None
        if not self._loaded:
            log.warning(
                "detector_running_heuristic_only",
                extra={
                    "model_path": str(self.model_path),
                    "hint": "run `python -m app.ml.train` to enable the ML scorer",
                },
            )

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def model_metadata(self) -> dict:
        meta = mdl.metadata(self._model)
        meta["threshold"] = self.threshold
        meta["simulated_latency_ms"] = self.simulated_latency_ms
        meta["simulated_latency_jitter_ms"] = self.simulated_latency_jitter_ms
        return meta

    # ── inference ────────────────────────────────────────────────────────

    async def detect(self, req: DetectRequest) -> DetectResponse:
        start = time.perf_counter()

        await self._maybe_simulate_latency()

        features = feat.extract(req)
        ml_score = (
            mdl.score(self._model, feat.to_vector(features))
            if self._model is not None
            else 0.0
        )
        heur = mdl.heuristic_score(features.model_dump())
        final = mdl.combine(ml_score, heur)

        is_anomaly = final >= self.threshold
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DetectResponse(
            is_anomaly=is_anomaly,
            score=round(final, 4),
            label="anomaly" if is_anomaly else "normal",
            threshold=self.threshold,
            model=mdl.MODEL_NAME,
            model_version=mdl.MODEL_VERSION,
            features=features,
            explain=feat.explain(features),
            inference_ms=round(elapsed_ms, 3),
            request_id=req.request_id,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    async def _maybe_simulate_latency(self) -> None:
        if self.simulated_latency_ms <= 0:
            return
        jitter = self.simulated_latency_jitter_ms
        target = self.simulated_latency_ms
        if jitter > 0:
            target += random.uniform(-jitter, jitter)
        target = max(0.0, target)
        await asyncio.sleep(target / 1000.0)
