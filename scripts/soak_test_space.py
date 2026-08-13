"""Run a bounded long-duration soak against the live Space inference API."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import statistics
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "ood_charts" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--duration-minutes", type=float, default=120)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=420)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "production" / "soak_report.json",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def main() -> None:
    args = parse_args()
    if args.duration_minutes <= 0 or args.concurrency <= 0:
        raise SystemExit("duration-minutes and concurrency must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prepared = []
    for case in manifest:
        raw = (args.manifest.parent / case["image"]).read_bytes()
        uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        prepared.append((case, uri))
    base_url = args.base_url.rstrip("/")
    headers = {}
    if os.getenv("SPACE_ADMIN_TOKEN"):
        headers["X-Admin-Token"] = os.environ["SPACE_ADMIN_TOKEN"]

    deadline = time.monotonic() + args.duration_minutes * 60
    counter_lock = threading.Lock()
    next_index = 0
    results: list[dict] = []

    def allocate():
        nonlocal next_index
        with counter_lock:
            if time.monotonic() >= deadline:
                return None
            if args.max_requests and next_index >= args.max_requests:
                return None
            index = next_index
            next_index += 1
            return index, prepared[index % len(prepared)]

    def worker(worker_id: int):
        with httpx.Client(
            timeout=args.timeout_seconds, follow_redirects=True
        ) as client:
            while True:
                item = allocate()
                if item is None:
                    return
                index, (case, uri) = item
                payload = {
                    "image_data_uri": uri,
                    "question": case["question"],
                    "response_mode": "ChartQA 短答",
                    "max_tokens": 64,
                }
                started = time.perf_counter()
                try:
                    response = client.post(
                        f"{base_url}/api/v1/infer", json=payload, headers=headers
                    )
                    latency = time.perf_counter() - started
                    result = {
                        "index": index,
                        "worker": worker_id,
                        "case_id": case["id"],
                        "status_code": response.status_code,
                        "latency_s": round(latency, 3),
                        "request_id": response.json().get("request_id")
                        if response.status_code == 200
                        else None,
                        "error": None
                        if response.status_code == 200
                        else response.text[:300],
                    }
                except Exception as exc:
                    result = {
                        "index": index,
                        "worker": worker_id,
                        "case_id": case["id"],
                        "status_code": 0,
                        "latency_s": round(time.perf_counter() - started, 3),
                        "request_id": None,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                with counter_lock:
                    results.append(result)
                    print(
                        f"[{len(results)}] worker={worker_id} case={case['id']} status={result['status_code']} latency={result['latency_s']}s"
                    )
                if args.interval_seconds:
                    time.sleep(args.interval_seconds)

    started_at = datetime.now(UTC)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        ready = client.get(f"{base_url}/readyz")
        if ready.status_code != 200:
            raise RuntimeError(
                f"Space 尚未 ready: {ready.status_code} {ready.text[:500]}"
            )
        status_before = client.get(f"{base_url}/api/status").json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, idx) for idx in range(args.concurrency)]
        for future in futures:
            future.result()

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        ready_after = client.get(f"{base_url}/readyz")
        status_after = client.get(f"{base_url}/api/status").json()
    codes = Counter(str(item["status_code"]) for item in results)
    latencies = [item["latency_s"] for item in results if item["status_code"] == 200]
    report = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "config": {
            "duration_minutes": args.duration_minutes,
            "concurrency": args.concurrency,
            "interval_seconds": args.interval_seconds,
            "max_requests": args.max_requests,
        },
        "summary": {
            "attempted": len(results),
            "succeeded": len(latencies),
            "status_codes": dict(codes),
            "success_rate": len(latencies) / len(results) if results else 0.0,
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_p95_s": percentile(latencies, 0.95),
            "latency_max_s": max(latencies) if latencies else None,
            "ready_after": ready_after.status_code == 200,
        },
        "status_before": status_before,
        "status_after": status_after,
        "requests": sorted(results, key=lambda item: item["index"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("saved:", args.output)
    if not report["summary"]["ready_after"]:
        raise SystemExit("Space lost readiness during soak")
    if codes.get("0", 0):
        raise SystemExit("Transport failures occurred during soak")


if __name__ == "__main__":
    main()
