from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WebStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            is_favorite INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            attachments_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attachments (
            file_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evidence (
            row_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            source_system TEXT NOT NULL,
            source_type TEXT NOT NULL,
            content TEXT NOT NULL,
            score REAL NOT NULL,
            image_path TEXT NOT NULL DEFAULT '',
            page INTEGER,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trace_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            node TEXT NOT NULL,
            event TEXT NOT NULL,
            detail TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(session_id, turn_id)
        );
        """
        with self.connection() as connection:
            connection.executescript(schema)
            connection.execute(
                """
                UPDATE sessions
                SET title = COALESCE(
                    (
                        SELECT substr(m.content, 1, 36)
                        FROM messages m
                        WHERE m.session_id = sessions.session_id
                          AND m.role = 'user'
                        ORDER BY m.created_at
                        LIMIT 1
                    ),
                    title
                )
                WHERE title = '新会话'
                  AND EXISTS (
                    SELECT 1 FROM messages m
                    WHERE m.session_id = sessions.session_id
                      AND m.role = 'user'
                )
                """
            )

    def create_session(
        self, title: str = "新会话", session_id: str | None = None
    ) -> dict[str, Any]:
        value = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (value, title.strip() or "新会话", now, now),
            )
        return self.get_session(value)

    def ensure_session(self, session_id: str, title: str = "新会话") -> None:
        self.create_session(title, session_id)

    def list_sessions(self, search: str = "") -> list[dict[str, Any]]:
        query = """
            SELECT s.*,
              (SELECT COUNT(*) FROM messages m
               WHERE m.session_id=s.session_id) AS message_count,
              (SELECT COUNT(DISTINCT m.turn_id) FROM messages m
               WHERE m.session_id=s.session_id) AS turn_count,
              (SELECT content FROM messages m
               WHERE m.session_id=s.session_id
               ORDER BY created_at DESC LIMIT 1) AS last_message
            FROM sessions s
        """
        values: tuple[Any, ...] = ()
        if search:
            query += " WHERE title LIKE ? OR session_id LIKE ?"
            token = f"%{search}%"
            values = (token, token)
        query += " ORDER BY is_favorite DESC, updated_at DESC"
        with self.connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._session_row(row) for row in rows]

    def _session_row(self, row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["is_favorite"] = bool(value.get("is_favorite"))
        value["tags"] = json.loads(value.pop("tags_json", "[]"))
        return value

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            messages = connection.execute(
                """
                SELECT * FROM messages WHERE session_id=?
                ORDER BY created_at
                """,
                (session_id,),
            ).fetchall()
        result = self._session_row(row)
        result["messages"] = [
            {
                **dict(message),
                "attachments": json.loads(
                    dict(message).get("attachments_json", "[]")
                ),
            }
            for message in messages
        ]
        return result

    def update_session(
        self, session_id: str, **updates: Any
    ) -> dict[str, Any]:
        allowed = {}
        if updates.get("title") is not None:
            allowed["title"] = str(updates["title"]).strip() or "新会话"
        if updates.get("is_favorite") is not None:
            allowed["is_favorite"] = int(bool(updates["is_favorite"]))
        if updates.get("tags") is not None:
            allowed["tags_json"] = json.dumps(
                updates["tags"], ensure_ascii=False
            )
        allowed["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in allowed)
        with self.connection() as connection:
            connection.execute(
                f"UPDATE sessions SET {assignments} WHERE session_id=?",
                (*allowed.values(), session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        deleted_rows = 0
        with self.connection() as connection:
            for table in (
                "messages",
                "attachments",
                "evidence",
                "trace_events",
                "feedback",
            ):
                cursor = connection.execute(
                    f"DELETE FROM {table} WHERE session_id=?", (session_id,)
                )
                deleted_rows += cursor.rowcount
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id=?", (session_id,)
            )
            deleted_rows += cursor.rowcount
        return deleted_rows > 0

    def save_attachment(
        self,
        session_id: str,
        file_id: str,
        file_type: str,
        file_name: str,
        file_path: str,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_session(session_id)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    session_id,
                    file_type,
                    file_name,
                    file_path,
                    size_bytes,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now(),
                ),
            )
        return self.get_attachment(file_id)

    def get_attachment(self, file_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM attachments WHERE file_id=?", (file_id,)
            ).fetchone()
        if row is None:
            raise KeyError(file_id)
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json", "{}"))
        return value

    def list_attachments(
        self, session_id: str | None = None, file_type: str | None = None
    ) -> list[dict[str, Any]]:
        clauses = []
        values: list[Any] = []
        if session_id:
            clauses.append("session_id=?")
            values.append(session_id)
        if file_type:
            clauses.append("file_type=?")
            values.append(file_type)
        query = "SELECT * FROM attachments"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self.connection() as connection:
            rows = connection.execute(query, values).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["metadata"] = json.loads(
                value.pop("metadata_json", "{}")
            )
            result.append(value)
        return result

    def save_turn(
        self,
        session_id: str,
        turn_id: str,
        query: str,
        answer: str,
        attachments: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        assistant_attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ensure_session(session_id)
        now = _now()
        title_candidate = query.strip()[:36]
        if not title_candidate.strip("?？ "):
            title_candidate = ""
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"msg_{uuid.uuid4().hex}",
                        session_id,
                        turn_id,
                        "user",
                        query,
                        json.dumps(attachments, ensure_ascii=False),
                        now,
                    ),
                    (
                        f"msg_{uuid.uuid4().hex}",
                        session_id,
                        turn_id,
                        "assistant",
                        answer,
                        json.dumps(
                            assistant_attachments or [],
                            ensure_ascii=False,
                        ),
                        now,
                    ),
                ],
            )
            for item in evidence:
                connection.execute(
                    """
                    INSERT INTO evidence VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"evrow_{uuid.uuid4().hex}",
                        session_id,
                        turn_id,
                        item["evidence_id"],
                        item["source_system"],
                        item["source_type"],
                        item["content"],
                        float(item.get("score", 0.0)),
                        item.get("image_path", ""),
                        item.get("page"),
                        json.dumps(
                            item.get("metadata", {}), ensure_ascii=False
                        ),
                        now,
                    ),
                )
            for index, item in enumerate(trace, start=1):
                connection.execute(
                    """
                    INSERT INTO trace_events VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"trace_{uuid.uuid4().hex}",
                        session_id,
                        turn_id,
                        index,
                        item.get("node", ""),
                        item.get("event", ""),
                        item.get("detail", ""),
                        now,
                    ),
                )
            title_row = connection.execute(
                "SELECT title FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            current_title = (
                str(title_row["title"]).strip()
                if title_row is not None
                else ""
            )
            is_default_title = (
                not current_title
                or current_title == "新会话"
                or current_title.startswith("鏂")
                or current_title.startswith("閺")
            )
            next_title = (
                title_candidate
                if is_default_title and title_candidate
                else current_title or "新会话"
            )
            connection.execute(
                """
                UPDATE sessions
                SET updated_at=?, title=?
                WHERE session_id=?
                """,
                (now, next_title, session_id),
            )

    def evidence_for_turn(
        self, session_id: str, turn_id: str
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence
                WHERE session_id=? AND turn_id=?
                ORDER BY rowid
                """,
                (session_id, turn_id),
            ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["metadata"] = json.loads(
                value.pop("metadata_json", "{}")
            )
            result.append(value)
        return result

    def save_feedback(
        self,
        session_id: str,
        turn_id: str,
        rating: int,
        comment: str = "",
    ) -> dict[str, Any]:
        if rating not in {-1, 0, 1}:
            raise ValueError("rating must be -1, 0 or 1")
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO feedback
                (feedback_id, session_id, turn_id, rating, comment,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, turn_id) DO UPDATE SET
                    rating=excluded.rating,
                    comment=excluded.comment,
                    updated_at=excluded.updated_at
                """,
                (
                    f"feedback_{uuid.uuid4().hex}",
                    session_id,
                    turn_id,
                    rating,
                    comment.strip(),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM feedback
                WHERE session_id=? AND turn_id=?
                """,
                (session_id, turn_id),
            ).fetchone()
        return dict(row) if row is not None else {}

    def feedback_stats(self) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT rating, COUNT(*) AS count
                FROM feedback GROUP BY rating
                """
            ).fetchall()
        counts = {int(row["rating"]): int(row["count"]) for row in rows}
        total = sum(counts.values())
        return {
            "total": total,
            "positive": counts.get(1, 0),
            "neutral": counts.get(0, 0),
            "negative": counts.get(-1, 0),
            "positive_rate": (
                round(counts.get(1, 0) / total, 4) if total else 0.0
            ),
        }
