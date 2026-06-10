import json
import time
import uuid
import threading
from typing import Any, Dict, Optional
from loguru import logger

# Thread-local storage to propagate trace IDs across nested function calls
_trace_context = threading.local()

def get_current_trace_id() -> str:
    """Retrieve the active trace ID from the thread context, generating one if absent."""
    if not hasattr(_trace_context, "trace_id") or _trace_context.trace_id is None:
        _trace_context.trace_id = str(uuid.uuid4())
    return _trace_context.trace_id

def set_current_trace_id(trace_id: Optional[str]) -> None:
    """Set the active trace ID in the thread context."""
    _trace_context.trace_id = trace_id


class TelemetryTracker:
    """
    Structured Telemetry and Distributed Tracing tracker.
    Generates unified JSON logs for CloudWatch, Datadog, or Grafana Loki.
    """
    def __init__(self, span_name: str, parent_trace_id: Optional[str] = None):
        self.span_name = span_name
        self.parent_trace_id = parent_trace_id
        self.trace_id = parent_trace_id or get_current_trace_id()
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.attributes: Dict[str, Any] = {}

    def __enter__(self):
        self.start_time = time.perf_counter()
        set_current_trace_id(self.trace_id)
        logger.debug(f"[Trace: {self.trace_id}] Starting span '{self.span_name}'")
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        """Add metadata attributes to the telemetry trace payload."""
        self.attributes[key] = value

    def record_exception(self, exc: Exception) -> None:
        """Record exception metadata in the span attributes."""
        self.attributes["error"] = True
        self.attributes["error_type"] = type(exc).__name__
        self.attributes["error_message"] = str(exc)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        latency_ms = (self.end_time - self.start_time) * 1000.0
        
        if exc_val is not None:
            self.record_exception(exc_val)

        log_payload = {
            "telemetry_type": "span",
            "trace_id": self.trace_id,
            "span_name": self.span_name,
            "parent_trace_id": self.parent_trace_id,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
            "attributes": self.attributes
        }

        # Print structured JSON directly to standard output.
        # Log-forwarders (e.g. Datadog Agent, AWS CloudWatch Agent) read stdout
        # and parse JSON strings automatically.
        logger.info(f"TELEMETRY_JSON: {json.dumps(log_payload)}")
        
        # Reset trace context if it was newly created
        if self.parent_trace_id is None:
            set_current_trace_id(None)


def log_business_metric(metric_name: str, value: float, unit: str, tags: Dict[str, str]) -> None:
    """Emit a business or performance metric log in standard telemetry schema."""
    payload = {
        "telemetry_type": "metric",
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "tags": tags,
        "timestamp": time.time()
    }
    logger.info(f"METRIC_JSON: {json.dumps(payload)}")
