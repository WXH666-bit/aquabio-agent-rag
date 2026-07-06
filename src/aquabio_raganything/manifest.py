from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SegmentManifest:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def get(self, segment_id: str) -> dict | None:
        return next(
            (
                row
                for row in reversed(self.rows())
                if row.get("segment_id") == segment_id
            ),
            None,
        )

    def update(self, segment: dict, status: str, **values) -> dict:
        previous = self.get(segment["segment_id"]) or {}
        if status in {"indexing", "fully_processed"}:
            previous = {**previous, "traceback": ""}
        row = {
            **previous,
            "segment_id": segment["segment_id"],
            "doc_id": segment["doc_id"],
            "scope": segment["scope"],
            "source_file": segment["source_file"],
            "start_page": segment["start_page"],
            "end_page": segment["end_page"],
            "source_sha256": segment["source_sha256"],
            "index_status": status,
            "attempts": previous.get("attempts", 0)
            + (1 if status == "indexing" else 0),
            "updated_at": _now(),
            **values,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
