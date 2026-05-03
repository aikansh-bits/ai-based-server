# ai-based-server

ML-driven API abuse detection service. Built as the **slow-but-accurate**
counterpart to the Node `rule-based-server` for the dissertation
**"Experimental Analysis of Latency–Accuracy Trade-offs in Real-Time API
Abuse Detection Systems."**

The service exposes a single inference endpoint, `POST /detect`, that takes a
small feature payload describing one inbound API request and returns a
calibrated anomaly score plus a human-readable explanation.

---

## Highlights

- **Hybrid scorer**: an Isolation Forest trained on synthetic legitimate
  traffic, **maxed** with a small heuristic safety net. The heuristic
  guarantees that obvious attacks (SQLi keywords, missing UAs, burst rates)
  are caught even if the model under-confidently scores them, and it lets
  the service operate gracefully when the model artefact is missing.
- **Calibrated [0, 1] score**: the IF's raw decision function is mapped
  through a fitted logistic squash so a single threshold (`ANOMALY_THRESHOLD`)
  works across runs.
- **Interpretable**: every response includes the numeric features that fed
  the model and a short list of human-readable explanations. The analyser
  dashboard renders these, so reviewers don't see a black-box score.
- **Deterministic, reproducible training**: synthetic dataset and seed are
  fixed; the trained `.joblib` artefact is byte-stable for a given seed.
- **Optional simulated inference latency** (`SIMULATED_LATENCY_MS`) so the
  experiments can sweep latency budgets without retraining a heavier model.
- **Production-shaped FastAPI**: app factory, CORS, structured JSON logging,
  correlation IDs, error handler, `/health`, `/ready`, `/info`, OpenAPI docs.

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# train the model (creates ./models/isolation_forest.joblib)
python -m app.ml.train

cp .env.example .env
python -m app.main
```

Server defaults to `http://localhost:8000`.

```bash
curl -s localhost:8000/health | jq
curl -s -X POST localhost:8000/detect -H 'content-type: application/json' -d '{
  "request_id": "demo",
  "method": "GET",
  "path": "/api/search",
  "endpoint": "/api/search?q=1%27%20OR%20%271%27%3D%271",
  "user_agent": "Mozilla/5.0",
  "content_length": 0,
  "has_body": false,
  "query_keys": ["q"],
  "body_keys": [],
  "history": {"requests_1min": 5, "requests_burst": 1, "distinct_paths": 2}
}' | jq
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/ready`  | Readiness — flags `degraded` if the model artefact is missing |
| `GET`  | `/info`   | Service + model metadata (training info, calibration, feature order) |
| `GET`  | `/docs`   | Swagger UI |
| `POST` | `/detect` | Score a single API request |

### `POST /detect` request shape

```jsonc
{
  "request_id": "uuid",
  "method": "POST",
  "path": "/api/login",
  "endpoint": "/api/login",
  "ip": "127.0.0.1",
  "user_agent": "curl/8.4.0",
  "content_length": 84,
  "has_body": true,
  "query_keys": [],
  "body_keys": ["username", "password"],
  "history": {
    "requests_1min": 120,
    "requests_burst": 25,
    "distinct_paths": 1
  }
}
```

### `POST /detect` response shape

```jsonc
{
  "is_anomaly": true,
  "score": 0.91,
  "label": "anomaly",
  "threshold": 0.55,
  "model": "isolation_forest",
  "model_version": "1.0.0",
  "features": {
    "request_count_1min": 120, "request_count_burst": 25,
    "distinct_paths": 1, "content_length": 84, "path_length": 11,
    "query_key_count": 0, "body_key_count": 2,
    "suspicious_keyword_score": 0.0, "user_agent_risk": 1.0,
    "method_risk": 0.0
  },
  "explain": ["suspicious or missing user-agent", "high sustained rate from this IP", "high burst rate from this IP"],
  "inference_ms": 1.832,
  "request_id": "uuid"
}
```

---

## Training

```bash
python -m app.ml.train
```

This script:

1. Generates 5 000 synthetic *legitimate* and 1 000 *malicious* request feature
   vectors. The malicious mixture covers payload-based attacks (SQLi, XSS,
   path traversal, command injection), brute-force/burst behaviour, and
   endpoint-scanning.
2. Fits an Isolation Forest (`n_estimators=200`, `max_samples=2048`) on the
   pooled data with `contamination` set to the empirical anomaly fraction.
3. Fits a one-parameter logistic calibration so the runtime scorer returns
   probabilities in [0, 1].
4. Reports a hold-out ROC-AUC and persists the bundled artefact to
   `MODEL_PATH` (default: `./models/isolation_forest.joblib`).

Re-run after changing the feature definitions in `app/ml/features.py`.

---

## Configuration

All settings come from environment variables (`.env` is auto-loaded). See
[`.env.example`](./.env.example) for the full list. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `./models/isolation_forest.joblib` | Where the trained artefact lives |
| `ANOMALY_THRESHOLD` | `0.55` | Cut-off for `is_anomaly=true` |
| `SIMULATED_LATENCY_MS` | `0` | Add fake inference latency (for experiments) |
| `SIMULATED_LATENCY_JITTER_MS` | `0` | +/- jitter on the above |
| `CORS_ORIGINS` | `*` | Restrict in production |
| `LOG_LEVEL` | `info` | `debug` for dev, `info`/`warn` in prod |

---

## Project layout

```
ai-based-server/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI factory + uvicorn entrypoint
│   ├── config.py          # pydantic-settings
│   ├── logging_config.py  # JSON / coloured logger
│   ├── schemas.py         # request/response Pydantic models
│   ├── routers/
│   │   ├── detect.py      # POST /detect
│   │   └── system.py      # /health /ready /info
│   ├── services/
│   │   └── detector.py    # orchestrates features + model + heuristic
│   └── ml/
│       ├── features.py    # feature engineering
│       ├── model.py       # IF wrapper + calibration + heuristic
│       └── train.py       # synthetic data + training
├── models/                # generated by `python -m app.ml.train`
│   └── isolation_forest.joblib
├── requirements.txt
├── .env.example
└── README.md
```

---

## Docker

```bash
docker build -t ai-based-server .
docker run --rm -p 8000:8000 \
  -e CORS_ORIGINS=http://localhost:3001 \
  ai-based-server
```

The image runs `python -m app.ml.train` as part of the build so the artefact
is shipped inside the image — no external storage required.

---

## License

MIT — academic use.
