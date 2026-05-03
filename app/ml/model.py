"""Anomaly-detection model wrapper.

Architecture
------------
- **Isolation Forest** (scikit-learn) trained on synthetic *legitimate* traffic.
  This is an unsupervised, tree-based detector that excels at flagging points
  in low-density regions of feature space — a good default for "I have lots
  of normal data and a few anomalies" problems, which mirrors real API
  abuse-detection deployments.
- **Calibrated score in [0, 1]**: the raw decision function from the forest
  is mapped through a logistic squash with parameters fitted at training time
  so we can compare scores across runs and apply a single threshold.
- **Heuristic safety net**: a small, hand-tuned scorer runs alongside the ML
  model. Whichever score is higher wins. This guarantees that obvious attacks
  (e.g. SQLi keywords) are not missed even if the IF underestimates them, and
  it ensures the service degrades gracefully if the model artefact is missing.

Persistence
-----------
The trainer (`app.ml.train`) writes a single `.joblib` artefact containing
the fitted Isolation Forest plus its calibration parameters and metadata.
Loading is lazy: the first call to `score()` warms the model up.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from app.ml.features import FEATURE_ORDER

log = logging.getLogger(__name__)

MODEL_NAME = "isolation_forest"
MODEL_VERSION = "1.0.0"


@dataclass
class CalibratedModel:
    """Bundle of artefacts that get persisted together as a single .joblib."""

    forest: IsolationForest
    calibration_a: float  # logistic slope
    calibration_b: float  # logistic intercept
    feature_order: list[str] = field(default_factory=lambda: list(FEATURE_ORDER))
    trained_at: str = ""
    train_samples: int = 0
    train_anomaly_fraction: float = 0.0


# ─── runtime API ────────────────────────────────────────────────────────────


def load(path: Path) -> CalibratedModel | None:
    """Load a trained model from disk. Returns None if the file is missing."""
    if not path.exists():
        log.warning(
            "model_missing",
            extra={"path": str(path), "hint": "run `python -m app.ml.train`"},
        )
        return None
    try:
        model: CalibratedModel = joblib.load(path)
        if model.feature_order != FEATURE_ORDER:
            log.error(
                "model_feature_mismatch",
                extra={
                    "expected": FEATURE_ORDER,
                    "got": model.feature_order,
                },
            )
            return None
        log.info(
            "model_loaded",
            extra={
                "path": str(path),
                "trained_at": model.trained_at,
                "train_samples": model.train_samples,
            },
        )
        return model
    except Exception as exc:  # noqa: BLE001
        log.error("model_load_failed", extra={"path": str(path), "error": str(exc)})
        return None


def save(model: CalibratedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("model_saved", extra={"path": str(path)})


def _logistic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Numerically-stable logistic squash."""
    z = -(a * x + b)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(z)), np.exp(-z) / (1.0 + np.exp(-z)))


def score(model: CalibratedModel, features: list[float]) -> float:
    """Calibrated anomaly score in [0, 1] for a single feature vector."""
    x = np.array(features, dtype=float).reshape(1, -1)
    # Higher decision_function -> more "normal"; we negate so higher = more anomalous.
    raw = -float(model.forest.decision_function(x)[0])
    p = float(_logistic(np.array([raw]), model.calibration_a, model.calibration_b)[0])
    # Guard against NaN / inf if calibration drifts in pathological inputs.
    if not math.isfinite(p):
        return 0.5
    return max(0.0, min(1.0, p))


# ─── heuristic safety net ──────────────────────────────────────────────────


def heuristic_score(features: dict[str, float]) -> float:
    """Cheap, hand-tuned scorer. Used as a guarantee floor when the model is
    missing, and as a max-merge alongside the ML score so the system can't be
    fooled by a single under-confident model.
    """
    score = 0.0
    score += 0.7 * features.get("suspicious_keyword_score", 0.0)
    score += 0.3 * features.get("user_agent_risk", 0.0)
    score += min(features.get("request_count_burst", 0.0) / 30.0, 1.0) * 0.4
    score += min(features.get("request_count_1min", 0.0) / 100.0, 1.0) * 0.3
    score += min(features.get("distinct_paths", 0.0) / 20.0, 1.0) * 0.3
    score += min(features.get("path_length", 0.0) / 200.0, 1.0) * 0.1
    score += min(features.get("content_length", 0.0) / 80_000.0, 1.0) * 0.1
    score += features.get("method_risk", 0.0) * 0.1
    return min(score, 1.0)


def combine(ml: float, heuristic: float) -> float:
    """Maximum-of-two combination — never let either signal silence the other."""
    return max(ml, heuristic)


def metadata(model: CalibratedModel | None) -> dict[str, Any]:
    """Diagnostic blob exposed by /info."""
    if model is None:
        return {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "loaded": False,
            "feature_order": FEATURE_ORDER,
        }
    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "loaded": True,
        "trained_at": model.trained_at,
        "train_samples": model.train_samples,
        "train_anomaly_fraction": model.train_anomaly_fraction,
        "feature_order": model.feature_order,
        "calibration": {"a": model.calibration_a, "b": model.calibration_b},
    }
