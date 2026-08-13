"""Production runtime primitives for the ChartQA GGUF Space.

This module deliberately has no Gradio or FastAPI dependency.  The admission
controller, rate limiter, metrics registry, llama.cpp supervisor, and inference
service can therefore be tested independently from the web UI.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import httpx
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

SHORT = "ChartQA 短答"
EXPLAIN = "分析說明"
ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} 必須 >= {minimum}，目前為 {value}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    if raw not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} 必須是 0/1/false/true/no/yes")
    return raw in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    model_repo: str
    model_revision: str
    model_file: str
    mmproj_file: str
    model_alias: str
    llama_port: int
    startup_timeout_s: int
    inference_timeout_s: int
    queue_wait_timeout_s: int
    max_queue_size: int
    max_question_chars: int
    max_image_pixels: int
    max_image_bytes: int
    max_output_tokens: int
    rate_limit_requests: int
    rate_limit_window_s: int
    max_restarts_per_hour: int
    dry_run: bool
    admin_token: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_repo=os.getenv("MODEL_REPO", "steven0226/qwen3vl-8b-chartqa-gguf"),
            model_revision=os.getenv("MODEL_REVISION", "main"),
            model_file=os.getenv("MODEL_FILE", "Qwen3VL-8B-ChartQA-Q4_K_M.gguf"),
            mmproj_file=os.getenv("MMPROJ_FILE", "mmproj-Qwen3VL-8B-ChartQA-Q8_0.gguf"),
            model_alias=os.getenv("MODEL_ALIAS", "qwen3vl-chartqa-gguf"),
            llama_port=_env_int("LLAMA_PORT", 8080),
            startup_timeout_s=_env_int("STARTUP_TIMEOUT_S", 900),
            inference_timeout_s=_env_int("INFERENCE_TIMEOUT_S", 360),
            queue_wait_timeout_s=_env_int("QUEUE_WAIT_TIMEOUT_S", 180),
            max_queue_size=_env_int("MAX_QUEUE_SIZE", 6),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", 800),
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 16_000_000),
            max_image_bytes=_env_int("MAX_IMAGE_BYTES", 8_000_000),
            max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 256),
            rate_limit_requests=_env_int("RATE_LIMIT_REQUESTS", 6),
            rate_limit_window_s=_env_int("RATE_LIMIT_WINDOW_S", 600),
            max_restarts_per_hour=_env_int("MAX_RESTARTS_PER_HOUR", 3),
            dry_run=_env_bool("SPACE_DRY_RUN", False),
            admin_token=os.getenv("ADMIN_TOKEN", ""),
        )

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("admin_token", None)
        return data


class InputRejected(ValueError):
    """The request is malformed or exceeds a documented safety limit."""


class ServiceUnavailable(RuntimeError):
    """The model server is loading, stopped, or unhealthy."""


class RateLimitExceeded(RuntimeError):
    def __init__(self, retry_after_s: int):
        super().__init__(f"請求過於頻繁，請在 {retry_after_s} 秒後重試。")
        self.retry_after_s = retry_after_s


class QueueSaturated(RuntimeError):
    """The bounded admission queue has no remaining capacity."""


class QueueWaitTimeout(RuntimeError):
    """The request waited too long for the single inference slot."""


def log_event(event: str, **fields: object) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


class MetricsRegistry:
    def __init__(self, latency_window: int = 512):
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._in_flight = 0
        self._queued = 0
        self._latencies: deque[float] = deque(maxlen=latency_window)
        self._last_success_at: float | None = None
        self._last_error_at: float | None = None
        self._last_error = ""

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def set_queue(self, value: int) -> None:
        with self._lock:
            self._queued = max(0, value)

    def in_flight_delta(self, value: int) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight + value)

    def observe_success(self, latency_s: float) -> None:
        with self._lock:
            self._counters["requests_succeeded"] += 1
            self._latencies.append(float(latency_s))
            self._last_success_at = time.time()

    def observe_error(self, message: str) -> None:
        with self._lock:
            self._counters["requests_failed"] += 1
            self._last_error_at = time.time()
            self._last_error = message[:400]

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
        return values[index]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latencies = list(self._latencies)
            counters = dict(self._counters)
            return {
                "uptime_s": round(time.time() - self.started_at, 3),
                "counters": counters,
                "in_flight": self._in_flight,
                "queued": self._queued,
                "latency_s": {
                    "samples": len(latencies),
                    "p50": round(median(latencies), 3) if latencies else None,
                    "p95": (
                        round(self._percentile(latencies, 0.95) or 0.0, 3)
                        if latencies
                        else None
                    ),
                    "max": round(max(latencies), 3) if latencies else None,
                },
                "last_success_at": self._last_success_at,
                "last_error_at": self._last_error_at,
                "last_error": self._last_error,
            }

    def prometheus(self, manager_state: str) -> str:
        snap = self.snapshot()
        counters = snap["counters"]
        latency = snap["latency_s"]
        state_value = 1 if manager_state == "ready" else 0
        lines = [
            "# HELP chartops_ready Whether the llama.cpp backend is ready.",
            "# TYPE chartops_ready gauge",
            f"chartops_ready {state_value}",
            "# TYPE chartops_in_flight gauge",
            f"chartops_in_flight {snap['in_flight']}",
            "# TYPE chartops_queued gauge",
            f"chartops_queued {snap['queued']}",
            "# TYPE chartops_uptime_seconds gauge",
            f"chartops_uptime_seconds {snap['uptime_s']}",
        ]
        for name in sorted(counters):
            base_name = name.removesuffix("_total").replace("-", "_")
            metric = "chartops_" + base_name + "_total"
            lines.extend([f"# TYPE {metric} counter", f"{metric} {counters[name]}"])
        if latency["samples"]:
            lines.extend(
                [
                    "# TYPE chartops_inference_latency_seconds gauge",
                    f'chartops_inference_latency_seconds{{quantile="0.50"}} {latency["p50"]}',
                    f'chartops_inference_latency_seconds{{quantile="0.95"}} {latency["p95"]}',
                    f'chartops_inference_latency_seconds{{quantile="1.00"}} {latency["max"]}',
                ]
            )
        return "\n".join(lines) + "\n"


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_s: int):
        self.limit = limit
        self.window_s = window_s
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        now = time.monotonic() if now is None else now
        cutoff = now - self.window_s
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(events[0] + self.window_s - now) + 1)
                return False, retry_after
            events.append(now)
            if len(self._events) > 10_000:
                empty = [name for name, queue in self._events.items() if not queue]
                for name in empty[:1_000]:
                    self._events.pop(name, None)
            return True, 0


class AdmissionController:
    def __init__(self, settings: Settings, metrics: MetricsRegistry):
        self.settings = settings
        self.metrics = metrics
        self.rate_limiter = SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_s
        )
        self._slot = threading.BoundedSemaphore(value=1)
        self._queue_lock = threading.Lock()
        self._queued = 0

    @contextmanager
    def admit(
        self, client_id: str, *, bypass_rate_limit: bool = False
    ) -> Iterator[None]:
        if not bypass_rate_limit:
            allowed, retry_after = self.rate_limiter.check(client_id or "anonymous")
            if not allowed:
                self.metrics.increment("requests_rejected_rate_limit")
                raise RateLimitExceeded(retry_after)

        with self._queue_lock:
            if self._queued >= self.settings.max_queue_size:
                self.metrics.increment("requests_rejected_queue_full")
                raise QueueSaturated("服務目前滿載，等待佇列已滿，請稍後再試。")
            self._queued += 1
            self.metrics.set_queue(self._queued)

        try:
            acquired = self._slot.acquire(timeout=self.settings.queue_wait_timeout_s)
        finally:
            with self._queue_lock:
                self._queued -= 1
                self.metrics.set_queue(self._queued)

        if not acquired:
            self.metrics.increment("requests_rejected_queue_timeout")
            raise QueueWaitTimeout("等待推論資源逾時，請稍後重新送出。")

        self.metrics.in_flight_delta(1)
        try:
            yield
        finally:
            self.metrics.in_flight_delta(-1)
            self._slot.release()


class LlamaServerManager:
    def __init__(self, settings: Settings, metrics: MetricsRegistry):
        self.settings = settings
        self.metrics = metrics
        self.base_url = f"http://127.0.0.1:{settings.llama_port}"
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.log_path = Path("/tmp/llama_server.log")
        self.resolved_revision = (
            "dry-run" if settings.dry_run else settings.model_revision
        )
        self._state = "created"
        self._last_error = ""
        self._state_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._restart_times: deque[float] = deque()

    def _set_state(self, state: str, error: str = "") -> None:
        with self._state_lock:
            self._state = state
            self._last_error = error[:800]

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def last_error(self) -> str:
        with self._state_lock:
            return self._last_error

    def is_ready(self) -> bool:
        if self.settings.dry_run:
            return self.state == "ready"
        return (
            self.state == "ready"
            and self.process is not None
            and self.process.poll() is None
        )

    def status(self) -> dict[str, object]:
        return {
            "state": self.state,
            "ready": self.is_ready(),
            "resolved_revision": self.resolved_revision,
            "last_error": self.last_error,
            "pid": self.process.pid
            if self.process and self.process.poll() is None
            else None,
        }

    def _tail_log(self, lines: int = 60) -> str:
        if not self.log_path.exists():
            return "(llama.cpp log 尚未建立)"
        return "\n".join(
            self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -lines:
            ]
        )

    def _command(self, model_path: str, mmproj_path: str) -> list[str]:
        threads = min(max(1, int(os.getenv("CPU_CORES", os.cpu_count() or 2))), 8)
        return [
            shutil.which("llama-server") or "llama-server",
            "--model",
            model_path,
            "--mmproj",
            mmproj_path,
            "--alias",
            self.settings.model_alias,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.settings.llama_port),
            "--ctx-size",
            "4096",
            "--threads",
            str(threads),
            "--threads-batch",
            str(threads),
            "--batch-size",
            "512",
            "--ubatch-size",
            "128",
            "--parallel",
            "1",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--image-min-tokens",
            "256",
            "--image-max-tokens",
            "1024",
            "--jinja",
            "--no-webui",
        ]

    def start_background(self) -> None:
        if self.state in {"starting", "ready"}:
            return
        self._set_state("starting")
        thread = threading.Thread(target=self._start_guarded, daemon=True)
        thread.start()

    def _start_guarded(self) -> None:
        try:
            self.start()
        except Exception as exc:  # pragma: no cover - exercised by integration tests
            self._stop_process()
            message = f"{type(exc).__name__}: {exc}"
            self._set_state("error", message)
            self.metrics.increment("server_start_failures")
            log_event("server_start_failed", error=message)

    def start(self) -> None:
        with self._start_lock:
            if self.is_ready():
                return
            self._set_state("starting")
            if self.settings.dry_run:
                self.resolved_revision = "dry-run"
                self._set_state("ready")
                log_event("server_ready", mode="dry-run")
                return

            binary = shutil.which("llama-server")
            if not binary:
                raise RuntimeError("找不到 llama-server；Docker image 建置不完整。")

            self._stop_process()
            api = HfApi()
            info = api.model_info(
                self.settings.model_repo, revision=self.settings.model_revision
            )
            self.resolved_revision = info.sha
            log_event(
                "model_resolving",
                repo=self.settings.model_repo,
                revision=self.resolved_revision,
            )
            model_path = hf_hub_download(
                self.settings.model_repo,
                self.settings.model_file,
                revision=self.resolved_revision,
            )
            mmproj_path = hf_hub_download(
                self.settings.model_repo,
                self.settings.mmproj_file,
                revision=self.resolved_revision,
            )

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = self.log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                self._command(model_path, mmproj_path),
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + self.settings.startup_timeout_s
            with httpx.Client(timeout=5) as client:
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            "llama-server 提前退出 "
                            f"(exit={self.process.returncode})\n{self._tail_log()}"
                        )
                    try:
                        if client.get(f"{self.base_url}/health").status_code == 200:
                            self._set_state("ready")
                            log_event(
                                "server_ready",
                                pid=self.process.pid,
                                revision=self.resolved_revision,
                            )
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(2)
            raise TimeoutError(f"llama-server 啟動逾時。\n{self._tail_log()}")

    def _restart_allowed(self) -> bool:
        now = time.monotonic()
        cutoff = now - 3600
        while self._restart_times and self._restart_times[0] < cutoff:
            self._restart_times.popleft()
        if len(self._restart_times) >= self.settings.max_restarts_per_hour:
            return False
        self._restart_times.append(now)
        return True

    def ensure_ready(self) -> None:
        if self.is_ready():
            return
        state = self.state
        if state == "starting":
            raise ServiceUnavailable("模型仍在載入，請稍後再試。")
        if not self._restart_allowed():
            raise ServiceUnavailable("模型服務暫停自動重啟，請稍後再試。")
        self.metrics.increment("server_restarts")
        log_event("server_restart_requested", previous_state=state)
        try:
            self.start()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._set_state("error", message)
            raise ServiceUnavailable("模型服務目前不可用，已記錄錯誤。") from exc

    def chat(self, messages: list[dict], max_tokens: int, request_id: str) -> str:
        self.ensure_ready()
        if self.settings.dry_run:
            return "[dry-run] UI、限流、佇列與 API 請求格式正常；未載入 GGUF。"
        payload = {
            "model": self.settings.model_alias,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.settings.inference_timeout_s, connect=10)
            ) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            if self.process is None or self.process.poll() is not None:
                self._set_state("error", f"llama.cpp exited during {request_id}")
            raise ServiceUnavailable(
                f"llama.cpp 推論失敗：{type(exc).__name__}"
            ) from exc
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        return str(content).strip()

    def _stop_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def stop(self) -> None:
        with self._start_lock:
            self._stop_process()
            self._set_state("stopped")
            log_event("server_stopped")


@dataclass(frozen=True)
class InferenceResult:
    answer: str
    trace: str
    request_id: str
    latency_s: float


class InferenceService:
    def __init__(
        self,
        settings: Settings,
        manager: LlamaServerManager,
        admission: AdmissionController,
        metrics: MetricsRegistry,
    ):
        self.settings = settings
        self.manager = manager
        self.admission = admission
        self.metrics = metrics

    def _prepare_image(self, image: Image.Image) -> str:
        if not isinstance(image, Image.Image):
            raise InputRejected("圖片格式無法辨識。")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise InputRejected("圖片尺寸無效。")
        if width * height > self.settings.max_image_pixels:
            raise InputRejected(
                f"圖片像素過大；上限為 {self.settings.max_image_pixels:,} pixels。"
            )
        prepared = image.convert("RGB")
        prepared.thumbnail((1280, 1280))
        buf = io.BytesIO()
        prepared.save(buf, format="JPEG", quality=92, optimize=True)
        payload = buf.getvalue()
        if len(payload) > self.settings.max_image_bytes:
            raise InputRejected(
                f"圖片編碼後超過 {self.settings.max_image_bytes // 1_000_000} MB。"
            )
        return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")

    def _prompt(self, question: str, response_mode: str) -> tuple[str, int]:
        question = (question or "").strip()
        if not question:
            raise InputRejected("請輸入你想詢問圖表的問題。")
        if len(question) > self.settings.max_question_chars:
            raise InputRejected(
                f"問題過長；上限為 {self.settings.max_question_chars} 個字元。"
            )
        if response_mode == SHORT:
            return f"{question}\n{ANSWER_INSTRUCTION}", 64
        if response_mode == EXPLAIN:
            return (
                question
                + "\nRead the exact values from the chart. Explain the evidence and "
                "calculation briefly. End with 'Final answer: ...'.",
                128,
            )
        raise InputRejected("回答模式無效。")

    def infer(
        self,
        image: Image.Image,
        question: str,
        response_mode: str,
        max_tokens: int,
        *,
        client_id: str,
        bypass_rate_limit: bool = False,
    ) -> InferenceResult:
        request_id = uuid.uuid4().hex[:12]
        self.metrics.increment("requests_total")
        try:
            prompt, mode_min_tokens = self._prompt(question, response_mode)
            effective_max_tokens = min(
                self.settings.max_output_tokens,
                max(mode_min_tokens, int(max_tokens)),
            )
            image_uri = self._prepare_image(image)
        except InputRejected:
            self.metrics.increment("requests_rejected_input")
            log_event(
                "request_rejected",
                request_id=request_id,
                reason="InputRejected",
            )
            raise
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        log_event(
            "request_received",
            request_id=request_id,
            client=client_id,
            response_mode=response_mode,
            max_tokens=effective_max_tokens,
        )
        started = time.perf_counter()
        try:
            with self.admission.admit(client_id, bypass_rate_limit=bypass_rate_limit):
                answer = self.manager.chat(messages, effective_max_tokens, request_id)
        except (RateLimitExceeded, QueueSaturated, QueueWaitTimeout) as exc:
            log_event(
                "request_rejected",
                request_id=request_id,
                reason=type(exc).__name__,
            )
            raise
        except Exception as exc:
            self.metrics.observe_error(f"{type(exc).__name__}: {exc}")
            log_event(
                "request_failed",
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            raise
        latency_s = time.perf_counter() - started
        self.metrics.observe_success(latency_s)
        log_event(
            "request_succeeded",
            request_id=request_id,
            latency_s=round(latency_s, 3),
        )
        trace = (
            f"request={request_id} · {latency_s:.2f}s · deterministic · "
            f"max_tokens={effective_max_tokens} · llama.cpp CPU · "
            f"{self.settings.model_repo}@{self.manager.resolved_revision[:12]}"
        )
        return InferenceResult(answer, trace, request_id, latency_s)


def decode_data_uri(data_uri: str, max_bytes: int) -> Image.Image:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:image/"):
        raise InputRejected("image_data_uri 必須是 data:image/... 格式。")
    try:
        header, encoded = data_uri.split(",", 1)
    except ValueError as exc:
        raise InputRejected("image_data_uri 格式不完整。") from exc
    if ";base64" not in header:
        raise InputRejected("image_data_uri 必須使用 base64。")
    estimated_size = len(encoded) * 3 // 4
    if estimated_size > max_bytes:
        raise InputRejected(f"圖片 payload 超過 {max_bytes // 1_000_000} MB。")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > max_bytes:
            raise InputRejected(f"圖片 payload 超過 {max_bytes // 1_000_000} MB。")
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except InputRejected:
        raise
    except Exception as exc:
        raise InputRejected("圖片 payload 無法解碼。") from exc
