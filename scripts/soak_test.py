from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_json(
    method: str,
    url: str,
    timeout: int = 60,
    **kwargs: Any,
) -> tuple[int, Any, float]:
    started = time.perf_counter()
    response = requests.request(method, url, timeout=timeout, **kwargs)
    elapsed = round(time.perf_counter() - started, 3)
    response.raise_for_status()
    return response.status_code, response.json(), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AquaBio API/MCP long-running stability test."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-seconds", type=int, default=21600)
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--mcp-every", type=int, default=20)
    parser.add_argument(
        "--output",
        default="data/mrag/logs/soak_test.jsonl",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1, args.duration_seconds)
    iteration = 0
    failures = 0
    started_at = now()

    with output.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "start",
                    "started_at": started_at,
                    "duration_seconds": args.duration_seconds,
                    "interval_seconds": args.interval_seconds,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()

        while time.monotonic() < deadline:
            iteration += 1
            record: dict[str, Any] = {
                "event": "probe",
                "time": now(),
                "iteration": iteration,
                "checks": {},
            }
            try:
                status, payload, elapsed = request_json(
                    "GET", f"{args.base_url}/api/health", timeout=15
                )
                record["checks"]["health"] = {
                    "status_code": status,
                    "status": payload.get("status"),
                    "elapsed_seconds": elapsed,
                }
                _, system, elapsed = request_json(
                    "GET",
                    f"{args.base_url}/api/system/status",
                    timeout=30,
                )
                record["checks"]["system"] = {
                    "status": system.get("status"),
                    "warmup": system.get("warmup", {}).get("status"),
                    "graph_storage_valid": system.get("graph", {}).get(
                        "storage_valid"
                    ),
                    "elapsed_seconds": elapsed,
                }
                if iteration == 1 or iteration % args.mcp_every == 0:
                    _, architecture, elapsed = request_json(
                        "GET",
                        f"{args.base_url}/api/system/architecture",
                        timeout=30,
                    )
                    record["checks"]["architecture"] = {
                        "layers": len(architecture.get("layers", [])),
                        "mcp_servers": architecture.get("mcp", {}).get(
                            "servers", []
                        ),
                        "elapsed_seconds": elapsed,
                    }
            except Exception as error:
                failures += 1
                record["error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.interval_seconds, remaining))

        summary = {
            "event": "summary",
            "started_at": started_at,
            "finished_at": now(),
            "iterations": iteration,
            "failures": failures,
            "passed": failures == 0,
        }
        stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        stream.flush()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
