"""Prometheus metrics for the backend.

These cover the four things worth watching in a prediction service: traffic and errors,
latency (ours and the model server's), what the model is predicting, and how far the
incoming images have drifted from the training data.

Label values are always from a small fixed set. Request ids, file names and confidence
values are never used as labels, because each new value would create a new time series.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

requests_total = Counter(
    "leafguard_requests_total", "HTTP requests handled", ["endpoint", "method", "status"]
)
request_seconds = Histogram(
    "leafguard_request_duration_seconds", "End-to-end request duration", ["endpoint"]
)
model_seconds = Histogram(
    "leafguard_model_latency_seconds", "Time spent waiting for the model server"
)
predictions_total = Counter(
    "leafguard_predictions_total", "Predictions returned", ["predicted_class", "status"]
)
confidence = Histogram(
    "leafguard_prediction_confidence",
    "Confidence of the top class",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
rejected_total = Counter(
    "leafguard_rejected_total", "Uploads rejected before reaching the model", ["reason"]
)
model_errors_total = Counter(
    "leafguard_model_errors_total", "Failed calls to the model server", ["kind"]
)
feedback_total = Counter("leafguard_feedback_total", "User feedback received", ["agreed"])
drift_score = Gauge(
    "leafguard_drift_score", "PSI of recent uploads against the training baseline", ["feature"]
)
model_info = Gauge(
    "leafguard_model_info",
    "Always 1; the labels carry the serving model version",
    ["version", "alias"],
)
