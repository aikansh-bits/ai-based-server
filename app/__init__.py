"""AI-based API abuse detection server.

A FastAPI service that scores incoming API requests for anomalous behaviour
using a hybrid of an Isolation Forest model (learned on synthetic legitimate
traffic) and a small set of high-signal heuristic features. Designed as the
slow-but-accurate counterpart to the Node rule-based-server in the
dissertation experiments.
"""

__version__ = "1.0.0"
