"""Poll production endpoints and write an availability/queue/latency report."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--duration-minutes", type=float, default=60)
    parser.add_argument("--interval-seconds", type=float, default=30)
    parser.add_argument("--max-p95-seconds", type=float, default=360)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "production" / "monitor_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.duration_minutes * 60
    base_url = args.base_url.rstrip("/")
    samples = []
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        while time.monotonic() < deadline:
            sample = {"ts": datetime.now(UTC).isoformat()}
            try:
                ready = client.get(f"{base_url}/readyz")
                status = client.get(f"{base_url}/api/status")
                sample.update(
                    {
                        "ready_status": ready.status_code,
                        "status_status": status.status_code,
                        "status": status.json() if status.status_code == 200 else None,
                    }
                )
            except Exception as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"[:300]
            samples.append(sample)
            print(
                sample["ts"],
                "ready=",
                sample.get("ready_status"),
                "error=",
                sample.get("error"),
            )
            time.sleep(args.interval_seconds)
    ready_samples = [item for item in samples if item.get("ready_status") == 200]
    p95_values = [
        item["status"]["metrics"]["latency_s"]["p95"]
        for item in samples
        if item.get("status")
        and item["status"]["metrics"]["latency_s"]["p95"] is not None
    ]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "summary": {
            "samples": len(samples),
            "ready_samples": len(ready_samples),
            "availability": len(ready_samples) / len(samples) if samples else 0.0,
            "max_observed_p95_s": max(p95_values) if p95_values else None,
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["availability"] < 1.0:
        raise SystemExit("Monitoring detected readiness loss")
    if p95_values and max(p95_values) > args.max_p95_seconds:
        raise SystemExit("Monitoring detected p95 latency above threshold")


if __name__ == "__main__":
    main()
