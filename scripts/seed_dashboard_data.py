from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, text

DSN = os.getenv("POSTGRES_DSN", "postgresql+psycopg://postgres:postgres@localhost:5432/boardbot")
engine = create_engine(DSN, future=True)

DECISIONS = ["AUTO_POST", "DRAFT", "ASK_MORE", "HANDOFF"]
TYPES = [("배송", "지연"), ("클레임", "파손"), ("반품", "오배송")]


def ins_metric(conn, case_id: str, name: str, tags: dict, value: float = 1.0, created_at: datetime | None = None):
    conn.execute(
        text("INSERT INTO metrics_events(case_id, metric_name, value, tags_jsonb, created_at) VALUES (:c,:n,:v,CAST(:t AS jsonb),:d)"),
        {"c": case_id, "n": name, "v": value, "t": json.dumps(tags), "d": created_at or datetime.now(UTC)},
    )


with engine.begin() as conn:
    for i in range(1, 201):
        case_id = f"seed-{i:04d}"
        tenant_id = f"tenant-{(i % 3) + 1}"
        t, st = random.choice(TYPES)
        decision = random.choices(DECISIONS, weights=[55, 20, 20, 5])[0]
        risk_level = random.choices(["LOW", "MEDIUM", "HIGH"], weights=[70, 25, 5])[0]
        created = datetime.now(UTC) - timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))
        base = {
            "tenant_id": tenant_id,
            "case_id": case_id,
            "type": t,
            "subtype": st,
            "response_mode": "AGENT" if st == "파손" else "SCENARIO",
            "decision": decision,
            "prompt_version": "v1",
            "config_version": "v1",
            "model": "gemini-1.5-flash",
            "success": True,
        }
        ins_metric(conn, case_id, "case_received", base, created_at=created)
        ins_metric(conn, case_id, "classification_completed", {**base, "is_mismatch": random.random() < 0.07, "mismatch_reason": "", "risk_level": risk_level, "risk_tags": "", "handoff_required": decision == "HANDOFF", "is_multi_issue": random.random() < 0.25, "issues_count": random.randint(1, 3), "composition_hint": "SECTIONED_REPLY"}, created_at=created)
        ins_metric(conn, case_id, "llm_usage", {**base, "prompt_tokens": random.randint(100, 600), "completion_tokens": random.randint(80, 400), "total_tokens": random.randint(200, 900), "estimated_cost_usd": round(random.uniform(0.0002, 0.004), 6)}, created_at=created)
        for phase in ["PII", "POLICY", "GROUNDING"]:
            passed = random.random() > (0.03 if phase != "GROUNDING" else 0.12)
            ins_metric(conn, case_id, "validator_result", {**base, "phase": phase, "passed": passed, "fail_reason_code": "" if passed else f"{phase}_FAIL"}, created_at=created)
        for tool in ["rag_search", "post_reply"]:
            succ = random.random() > 0.05
            ins_metric(conn, case_id, "tool_call", {**base, "tool_name": tool, "latency_ms": random.randint(50, 900), "success": succ, "retry_count": 0, "circuit_breaker_open": False, "error_code": "" if succ else "TOOL_TIMEOUT"}, created_at=created)
        ins_metric(conn, case_id, "decision_made", {**base, "decision": decision, "latency_ms": random.randint(200, 1500)}, created_at=created)
        if decision in {"AUTO_POST", "ASK_MORE"}:
            ins_metric(conn, case_id, "post_reply_result", {**base, "success": True, "error_code": ""}, created_at=created)
        conn.execute(
            text("INSERT INTO cases(case_id, thread_id, created_at, updated_at, state_jsonb) VALUES (:c,:t,:d,:d,CAST(:s AS jsonb)) ON CONFLICT (case_id) DO NOTHING"),
            {
                "c": case_id,
                "t": case_id,
                "d": created,
                "s": json.dumps({"input": {"tenant_id": tenant_id, "case_id": case_id, "type": t, "subtype": st, "title": "seed", "body": "masked"}, "classification": {"issues": [{"issue_id": "I1", "summary": "seed"}]}, "plan": {"mode": base["response_mode"]}, "tooling": {"trace": []}, "decision": {"type": decision}, "telemetry": {"metrics": []}}),
            },
        )

print("seed completed")
