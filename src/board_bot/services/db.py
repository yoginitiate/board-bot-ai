from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text


class PostgresRepo:
    def __init__(self, dsn: str) -> None:
        self.engine = create_engine(dsn, future=True)

    def upsert_case(self, case_id: str, thread_id: str, state: dict[str, Any]) -> None:
        query = text(
            """
            INSERT INTO cases(case_id, thread_id, created_at, updated_at, state_jsonb)
            VALUES (:case_id, :thread_id, now(), now(), CAST(:state AS jsonb))
            ON CONFLICT (case_id) DO UPDATE SET updated_at = now(), state_jsonb = CAST(:state AS jsonb)
            """
        )
        with self.engine.begin() as conn:
            conn.execute(query, {"case_id": case_id, "thread_id": thread_id, "state": __import__("json").dumps(state)})

    def log_event(self, case_id: str, node: str, event: str, payload: dict[str, Any]) -> None:
        query = text(
            "INSERT INTO execution_logs(case_id, node, event, payload_jsonb, created_at) VALUES (:case_id,:node,:event,CAST(:payload AS jsonb),:created_at)"
        )
        with self.engine.begin() as conn:
            conn.execute(query, {"case_id": case_id, "node": node, "event": event, "payload": __import__("json").dumps(payload), "created_at": datetime.utcnow()})

    def metric(self, case_id: str, name: str, value: float, tags: dict[str, Any]) -> None:
        query = text(
            "INSERT INTO metrics_events(case_id, metric_name, value, tags_jsonb, created_at) VALUES (:case_id,:name,:value,CAST(:tags AS jsonb),now())"
        )
        with self.engine.begin() as conn:
            conn.execute(query, {"case_id": case_id, "name": name, "value": value, "tags": __import__("json").dumps(tags)})

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT case_id, state_jsonb FROM cases WHERE case_id=:id"), {"id": case_id}).first()
        if not row:
            return None
        return {"case_id": row[0], "state": row[1]}
