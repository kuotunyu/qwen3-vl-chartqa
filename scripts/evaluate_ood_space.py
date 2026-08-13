"""Evaluate the live Space API on the deterministic non-ChartQA chart suite."""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.relaxed_accuracy import normalize_prediction, relaxed_correctness  # noqa: E402

DEFAULT_MANIFEST = ROOT / "assets" / "ood_charts" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout-seconds", type=float, default=420)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "production" / "ood_report.json",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    return parser.parse_args()


def image_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def main() -> None:
    args = parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_url = args.base_url.rstrip("/")
    headers = {}
    if os.getenv("SPACE_ADMIN_TOKEN"):
        headers["X-Admin-Token"] = os.environ["SPACE_ADMIN_TOKEN"]
    results = []
    latencies = []
    with httpx.Client(timeout=args.timeout_seconds, follow_redirects=True) as client:
        ready = client.get(f"{base_url}/readyz")
        if ready.status_code != 200:
            raise RuntimeError(
                f"Space 尚未 ready: {ready.status_code} {ready.text[:500]}"
            )
        for index, case in enumerate(cases, start=1):
            payload = {
                "image_data_uri": image_uri(args.manifest.parent / case["image"]),
                "question": case["question"],
                "response_mode": "ChartQA 短答",
                "max_tokens": 64,
            }
            started = time.perf_counter()
            response = client.post(
                f"{base_url}/api/v1/infer", json=payload, headers=headers
            )
            latency = time.perf_counter() - started
            if response.status_code != 200:
                result = {
                    **case,
                    "status_code": response.status_code,
                    "error": response.text[:500],
                    "latency_s": round(latency, 3),
                    "correct": False,
                }
            else:
                body = response.json()
                answer = normalize_prediction(str(body["answer"]))
                correct = relaxed_correctness(answer, str(case["expected"]))
                latencies.append(latency)
                result = {
                    **case,
                    "status_code": 200,
                    "answer": answer,
                    "correct": correct,
                    "latency_s": round(latency, 3),
                    "request_id": body.get("request_id"),
                }
            results.append(result)
            print(
                f"[{index}/{len(cases)}] {case['id']}: {result.get('answer', result.get('error'))} | correct={result['correct']}"
            )
    success = [item for item in results if item["status_code"] == 200]
    accuracy = sum(bool(item["correct"]) for item in results) / len(results)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "manifest": str(args.manifest),
        "summary": {
            "total": len(results),
            "http_success": len(success),
            "accuracy": accuracy,
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_p95_s": percentile(latencies, 0.95),
            "latency_max_s": max(latencies) if latencies else None,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("saved:", args.output)
    if len(success) != len(results):
        raise SystemExit("OOD evaluation has HTTP failures")
    if accuracy < args.min_accuracy:
        raise SystemExit(
            f"OOD accuracy {accuracy:.2%} < required {args.min_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
