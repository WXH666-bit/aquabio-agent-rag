from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the AquaBio soak-test process and log status."
    )
    parser.add_argument(
        "--log",
        default="data/mrag/logs/soak_test_6h.jsonl",
    )
    parser.add_argument(
        "--pid-file",
        default="data/mrag/logs/soak_test_6h.pid",
    )
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    pid_path = Path(args.pid_file).resolve()
    pid = 0
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = 0

    records = []
    if log_path.exists():
        with log_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    probes = [row for row in records if row.get("event") == "probe"]
    summaries = [row for row in records if row.get("event") == "summary"]
    failures = sum(1 for row in probes if row.get("error"))
    process_exists = False
    if pid:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        process_exists = f'"{pid}"' in completed.stdout
    running = bool(process_exists and not summaries)
    last_probe = probes[-1] if probes else {}
    result = {
        "pid": pid or None,
        "running": running,
        "log": str(log_path),
        "iterations": len(probes),
        "failures": failures,
        "last_probe_time": last_probe.get("time"),
        "last_probe_error": last_probe.get("error"),
        "summary": summaries[-1] if summaries else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
