from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    """Durable session storage for CLI processes."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_session_id(session_id: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())
        if not value:
            raise ValueError("session_id cannot be empty")
        return value[:80]

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{self.normalize_session_id(session_id)}.json"

    def load(self, session_id: str) -> dict[str, Any]:
        normalized = self.normalize_session_id(session_id)
        path = self.path_for(normalized)
        if not path.is_file():
            return {
                "session_id": normalized,
                "created_at": _now(),
                "updated_at": _now(),
                "summary": {
                    "last_species_ids": [],
                    "last_species_names": [],
                    "last_image_path": "",
                    "last_image_caption": "",
                    "last_answer_summary": "",
                    "last_evidence_ids": [],
                },
                "turns": [],
            }
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Invalid conversation file: {path}")
        value.setdefault("summary", {})
        value.setdefault("turns", [])
        return value

    def save(self, session: dict[str, Any]) -> Path:
        session_id = self.normalize_session_id(session["session_id"])
        path = self.path_for(session_id)
        session["session_id"] = session_id
        session["updated_at"] = _now()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def append_turn(
        self,
        session_id: str,
        turn: dict[str, Any],
        summary: dict[str, Any],
    ) -> Path:
        session = self.load(session_id)
        turns = session.setdefault("turns", [])
        turn["turn_index"] = len(turns) + 1
        turn["created_at"] = _now()
        turns.append(turn)
        session["turns"] = turns[-100:]
        session["summary"] = summary
        return self.save(session)

    def clear(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(
            self.root.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "session_id": value.get("session_id", path.stem),
                    "turns": len(value.get("turns", [])),
                    "updated_at": value.get("updated_at", ""),
                    "last_species_names": value.get("summary", {}).get(
                        "last_species_names", []
                    ),
                }
            )
        return rows
