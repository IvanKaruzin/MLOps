"""Per-process, bounded-label metrics; request/session IDs belong in logs."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


BUCKETS = (.01, .025, .05, .1, .2, .4, .8, 1.5, 3, 6, 12, 24, 48, 90)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        options = {"registry": self.registry}
        self.running = Gauge("hw11_running", "Admitted requests retaining a compute slot", **options)
        self.waiting = Gauge("hw11_waiting", "Requests waiting for a compute slot", **options)
        self.requests = Counter("hw11_requests_total", "Terminal request outcomes", ["outcome"], **options)
        self.cache = Counter("hw11_cache_total", "Simulated local prefix cache lookups", ["result"], **options)
        self.ttfc = Histogram("hw11_ttfc_seconds", "TTFC of successfully completed responses", buckets=BUCKETS, **options)
        self.duration = Histogram("hw11_request_duration_seconds", "Full duration of successfully completed responses", buckets=BUCKETS, **options)
        self.queue = Histogram("hw11_queue_seconds", "Queue duration of admitted requests", buckets=BUCKETS, **options)
        for outcome in ("completed", "rejected", "interrupted", "error"):
            self.requests.labels(outcome)
        for result in ("hit", "miss"):
            self.cache.labels(result)

    def render(self):
        return generate_latest(self.registry)
