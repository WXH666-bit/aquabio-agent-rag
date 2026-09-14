from __future__ import annotations

import json
import re
import uuid
import threading
from weakref import WeakValueDictionary
from contextlib import contextmanager, ExitStack
from functools import wraps
from filelock import FileLock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_file_locks = WeakValueDictionary()
_file_locks_guard = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    """Durable session storage for CLI processes."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_session_id(session_id: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", session_id):
            raise ValueError("session_id must be 1-80 lowercase ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit")
        if session_id.split(".")[0] in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
            raise ValueError("Reserved session_id")
        return session_id

    @contextmanager
    def transaction(self, session_id: str, purpose: str = "store"):
        normalized = self.normalize_session_id(session_id)
        locks = self.root / ".locks"
        locks.mkdir(exist_ok=True)
        key = str((locks / f"{normalized}.{purpose}.lock").resolve())
        with _file_locks_guard:
            lock = _file_locks.get(key)
            if lock is None:
                lock = FileLock(key, timeout=0.2 if purpose == "workflow" else -1)
                _file_locks[key] = lock
        with lock:
            yield

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
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
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
        with self.transaction(session_id):
            session = self.load(session_id)
            turns = session.setdefault("turns", [])
            turn["turn_index"] = (turns[-1].get("turn_index", len(turns)) if turns else 0) + 1
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


def serialized_session(function):
    """Serialize an entire workflow turn across instances and processes."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        import inspect
        from .cancellation import check_cancelled
        bound = inspect.signature(function).bind(self, *args, **kwargs)
        bound.apply_defaults()
        session_id = bound.arguments.get("session_id") or bound.arguments["request"].session_id
        # Use timed acquisition so queued requests can respond to cancellation.
        from filelock import Timeout
        while True:
            check_cancelled()
            stack = ExitStack()
            try:
                stack.enter_context(self.conversations.transaction(session_id, "workflow"))
            except Timeout:
                stack.close()
                continue
            with stack:
                check_cancelled()
                return function(self, *args, **kwargs)
    return wrapped
