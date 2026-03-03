"""Postgres 접근 레이어.

케이스 상태, 실행 로그, 메트릭 이벤트를 저장/조회한다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text


class PostgresRepo:
    """게시판 Agent용 Postgres 저장소 클래스."""

    def __init__(self, dsn: str) -> None:
        """DB 엔진을 초기화한다.

        Args:
            dsn: SQLAlchemy 연결 문자열.
        """

        self.engine = create_engine(dsn, future=True)

    def upsert_case(self, case_id: str, thread_id: str, state: dict[str, Any]) -> None:
        """케이스 상태를 upsert한다.

        Side Effects:
            `board_bot.t_cases` 테이블에 INSERT/UPDATE 수행.
        """

        query = text(
            """
            INSERT INTO board_bot.t_cases(case_id, thread_id, created_at, updated_at, state_jsonb)
            VALUES (:case_id, :thread_id, now(), now(), CAST(:state AS jsonb))
            ON CONFLICT (case_id) DO UPDATE SET updated_at = now(), state_jsonb = CAST(:state AS jsonb)
            """
        )
        with self.engine.begin() as conn:
            conn.execute(query, {"case_id": case_id, "thread_id": thread_id, "state": json.dumps(state)})

    def log_event(self, case_id: str, node: str, event: str, payload: dict[str, Any]) -> None:
        """실행 로그를 저장한다."""

        query = text(
            "INSERT INTO board_bot.t_execution_logs(case_id, node, event, payload_jsonb, created_at) VALUES (:case_id,:node,:event,CAST(:payload AS jsonb),:created_at)"
        )
        with self.engine.begin() as conn:
            conn.execute(query, {"case_id": case_id, "node": node, "event": event, "payload": json.dumps(payload), "created_at": datetime.now(UTC)})

    def metric(self, case_id: str, name: str, value: float, tags: dict[str, Any]) -> None:
        """단일 메트릭 이벤트를 저장한다."""

        query = text(
            "INSERT INTO board_bot.t_metrics_events(case_id, metric_name, value, tags_jsonb, created_at) VALUES (:case_id,:name,:value,CAST(:tags AS jsonb),now())"
        )
        with self.engine.begin() as conn:
            conn.execute(query, {"case_id": case_id, "name": name, "value": value, "tags": json.dumps(tags)})

    def insert_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """버퍼링된 메트릭 이벤트를 배치 저장한다.

        Why:
            노드마다 DB I/O를 수행하지 않고 마지막에 일괄 적재해 지연을 줄인다.
        """

        query = text(
            "INSERT INTO board_bot.t_metrics_events(case_id, metric_name, value, tags_jsonb, created_at) VALUES (:case_id,:name,:value,CAST(:tags AS jsonb),now())"
        )
        with self.engine.begin() as conn:
            for item in metrics:
                tags = item.get("tags", {})
                conn.execute(
                    query,
                    {
                        "case_id": tags.get("case_id"),
                        "name": item.get("metric_name"),
                        "value": float(item.get("value", 1.0)),
                        "tags": json.dumps(tags),
                    },
                )


    def is_handoff_notified(self, case_id: str) -> bool:
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT 1 FROM board_bot.t_handoff_notifications WHERE case_id=:id LIMIT 1"), {"id": case_id}).first()
        return row is not None

    def mark_handoff_notified(self, case_id: str, payload: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO board_bot.t_handoff_notifications(case_id, payload_jsonb, created_at) VALUES (:id, CAST(:payload AS jsonb), now()) ON CONFLICT (case_id) DO NOTHING"),
                {"id": case_id, "payload": json.dumps(payload)},
            )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """단일 케이스 상태를 조회한다."""

        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT case_id, state_jsonb FROM board_bot.t_cases WHERE case_id=:id"), {"id": case_id}).first()
        if not row:
            return None
        return {"case_id": row[0], "state": row[1]}
