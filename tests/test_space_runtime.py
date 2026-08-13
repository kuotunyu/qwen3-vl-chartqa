from __future__ import annotations

import base64
import io
import sys
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))

from runtime import (  # noqa: E402
    SHORT,
    AdmissionController,
    InferenceService,
    InputRejected,
    LlamaServerManager,
    MetricsRegistry,
    QueueSaturated,
    RateLimitExceeded,
    Settings,
    SlidingWindowRateLimiter,
    decode_data_uri,
)


def settings(**overrides) -> Settings:
    base = Settings(
        model_repo="example/model",
        model_revision="abc123",
        model_file="model.gguf",
        mmproj_file="mmproj.gguf",
        model_alias="test",
        llama_port=18080,
        startup_timeout_s=5,
        inference_timeout_s=5,
        queue_wait_timeout_s=1,
        max_queue_size=1,
        max_question_chars=100,
        max_image_pixels=1_000_000,
        max_image_bytes=1_000_000,
        max_output_tokens=128,
        rate_limit_requests=10,
        rate_limit_window_s=60,
        max_restarts_per_hour=2,
        dry_run=True,
        admin_token="",
    )
    return replace(base, **overrides)


class RateLimiterTests(unittest.TestCase):
    def test_sliding_window_returns_retry_after(self):
        limiter = SlidingWindowRateLimiter(limit=2, window_s=10)
        self.assertEqual(limiter.check("ip", now=0), (True, 0))
        self.assertEqual(limiter.check("ip", now=1), (True, 0))
        allowed, retry_after = limiter.check("ip", now=2)
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 8)
        self.assertEqual(limiter.check("ip", now=11), (True, 0))

    def test_rate_limit_is_per_client(self):
        metrics = MetricsRegistry()
        gate = AdmissionController(
            settings(rate_limit_requests=1, max_queue_size=2), metrics
        )
        with gate.admit("a"):
            pass
        with self.assertRaises(RateLimitExceeded):
            with gate.admit("a"):
                pass
        with gate.admit("b"):
            pass


class AdmissionTests(unittest.TestCase):
    def test_bounded_queue_rejects_third_request(self):
        metrics = MetricsRegistry()
        gate = AdmissionController(settings(max_queue_size=1), metrics)
        active = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def first():
            with gate.admit("first"):
                active.set()
                release.wait(timeout=2)

        def second():
            active.wait(timeout=2)
            second_started.set()
            with gate.admit("second"):
                pass

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        self.assertTrue(active.wait(timeout=2))
        second_thread.start()
        self.assertTrue(second_started.wait(timeout=2))
        time.sleep(0.05)
        with self.assertRaises(QueueSaturated):
            with gate.admit("third"):
                pass
        release.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())


class InferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings()
        self.metrics = MetricsRegistry()
        self.manager = LlamaServerManager(self.settings, self.metrics)
        self.manager.start()
        self.admission = AdmissionController(self.settings, self.metrics)
        self.service = InferenceService(
            self.settings, self.manager, self.admission, self.metrics
        )

    def tearDown(self):
        self.manager.stop()

    def test_dry_run_inference_updates_metrics(self):
        image = Image.new("RGB", (320, 240), "white")
        result = self.service.infer(
            image, "Which bar is highest?", SHORT, 32, client_id="test"
        )
        self.assertIn("dry-run", result.answer)
        snap = self.metrics.snapshot()
        self.assertEqual(snap["counters"]["requests_total"], 1)
        self.assertEqual(snap["counters"]["requests_succeeded"], 1)
        self.assertEqual(snap["in_flight"], 0)

    def test_rejects_long_question_and_large_image(self):
        image = Image.new("RGB", (320, 240), "white")
        with self.assertRaises(InputRejected):
            self.service.infer(image, "x" * 101, SHORT, 32, client_id="test")
        huge = Image.new("RGB", (1001, 1001), "white")
        with self.assertRaises(InputRejected):
            self.service.infer(huge, "question", SHORT, 32, client_id="test2")

    def test_data_uri_round_trip(self):
        image = Image.new("RGB", (16, 12), "red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        decoded = decode_data_uri(uri, 100_000)
        self.assertEqual(decoded.size, (16, 12))
        with self.assertRaises(InputRejected):
            decode_data_uri("https://example.com/image.png", 100_000)


if __name__ == "__main__":
    unittest.main()
