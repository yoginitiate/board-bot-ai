"""metrics_events 버퍼링 유틸리티.

이 모듈은 노드 내부에서 공통 태그를 자동으로 채운 뒤, 이벤트를
`state.telemetry.metrics`에 누적한다. 실제 DB INSERT는 API 로거 단계에서 수행한다.

Security/Privacy:
    - PII 원문 저장 금지 원칙을 따른다.
    - 이벤트에는 코드/카운트/요약 수치만 저장한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def emit_metric(state: dict[str, Any], metric_name: str, value: float = 1.0, tags: dict[str, Any] | None = None) -> None:
    """메트릭 이벤트를 telemetry 버퍼에 추가한다.

    Args:
        state: LangGraph 상태 딕셔너리.
        metric_name: 이벤트 이름(case_received, tool_call 등).
        value: 수치 값(기본 1.0).
        tags: 이벤트별 상세 태그.

    Returns:
        None.

    Side Effects:
        `state.telemetry.metrics` 리스트가 변경된다.

    Observability:
        공통 태그(tenant/type/subtype/decision/model/version)를 자동 부여한다.
    """

    telemetry = state.setdefault("telemetry", {})
    items = telemetry.setdefault("metrics", [])
    base = {
        "tenant_id": state.get("input", {}).get("tenant_id"),
        "case_id": state.get("input", {}).get("case_id"),
        "thread_id": state.get("input", {}).get("case_id"),
        "type": state.get("input", {}).get("type"),
        "subtype": state.get("input", {}).get("subtype"),
        "response_mode": state.get("plan", {}).get("mode", "SCENARIO"),
        "decision": state.get("decision", {}).get("type"),
        "prompt_version": state.get("telemetry", {}).get("prompt_version", "v1"),
        "config_version": state.get("telemetry", {}).get("config_version", "v1"),
        "model": state.get("telemetry", {}).get("model", "gemini-1.5-flash"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if tags:
        base.update(tags)
    items.append({"metric_name": metric_name, "value": value, "tags": base})
