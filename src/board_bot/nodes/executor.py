"""Executor 영역 노드를 정의한다.

Executor는 Master가 만든 Plan DSL을 "어떻게 실행할지"에 집중한다.
도구 호출, 응답 생성, Validator 게이트, 최종 결정/출력, 텔레메트리 기록을 담당한다.
"""

from __future__ import annotations

import time

from board_bot.models.state import AgentState
from board_bot.services.metrics import emit_metric
from board_bot.utils.safety import mask_pii


def plan_engine(state: AgentState, tool_call) -> AgentState:
    """Plan DSL의 steps를 순차 실행한다.

    Args:
        state: `plan.steps`와 `plan.budget`가 포함된 상태.
        tool_call: MCP 도구 호출 함수.

    Returns:
        tool 실행 trace가 반영된 상태.

    Side Effects:
        - 외부 MCP 네트워크 I/O
        - `tool_call` 메트릭 적재

    Observability:
        tool_name/latency/success/retry/circuit_breaker/error_code 태그를 기록한다.

    Why:
        budget 초과 시 강제 DRAFT 전환해 runaway 호출을 차단한다.
    """

    results = []
    budget = state["plan"]["budget"]
    for idx, step in enumerate(state["plan"]["steps"]):
        if idx >= budget["max_tools"]:
            state["decision"] = {"type": "DRAFT", "reason": "budget_exceeded"}
            break

        started = time.perf_counter()
        ok = True
        err = ""
        try:
            res = tool_call(step["tool"], step.get("args", {}))
        except Exception as exc:  # pragma: no cover
            ok = False
            err = str(exc)
            res = {"error": err}
        latency_ms = (time.perf_counter() - started) * 1000

        emit_metric(
            state,
            "tool_call",
            tags={
                "tool_name": step["tool"],
                "latency_ms": latency_ms,
                "success": ok,
                "retry_count": 0,
                "circuit_breaker_open": False,
                "error_code": err[:64],
            },
        )
        results.append({"tool": step["tool"], "result": res})

    state.setdefault("tooling", {})["trace"] = results
    return state


def response_mode_gate(state: AgentState) -> AgentState:
    """SCENARIO/AGENT 모드를 draft에 반영한다.

    Why:
        하위 서브그래프의 렌더링 전략과 검증 강도를 모드별로 다르게 가져가기 위함.
    """

    state.setdefault("draft", {})["mode"] = state["plan"].get("mode", "SCENARIO")
    return state


def scenario_subgraph(state: AgentState) -> AgentState:
    """SCENARIO 모드 응답을 issue 섹션 단위로 합성한다.

    SECTIONED_REPLY 규칙을 따라, 멀티 이슈를 하나의 답변 안에서 분리 안내한다.
    """

    sections = []
    for issue in state["classification"]["issues"]:
        sections.append(f"[{issue['issue_id']}] {issue['summary']} 관련 안내드립니다.")
    state["draft"]["text"] = "\n".join(sections)
    return state


def agent_generate_subgraph(state: AgentState) -> AgentState:
    """AGENT 모드 응답 초안을 생성한다.

    Args:
        state: tool trace가 포함된 상태.

    Returns:
        `draft.text`, `draft.citations`가 채워진 상태.

    Security/Privacy:
        입력은 마스킹 텍스트를 사용한다.
    """

    trace = state.get("tooling", {}).get("trace", [])
    citations = [t["result"].get("sources", []) for t in trace if isinstance(t.get("result"), dict)]
    state["draft"]["text"] = f"문의 검토 결과입니다.\n{state['safety']['masked_text'][:120]}"
    state["draft"]["citations"] = citations
    return state


def validate_pii(state: AgentState) -> AgentState:
    """출력 초안에 2차 PII 마스킹을 적용한다.

    Observability:
        `validator_result(phase=PII)`를 기록한다.

    Why:
        입력 단계 마스킹 이후에도 템플릿/도구 응답 조합 과정에서 재노출이 생길 수 있어
        출력 직전 재검증이 필요하다.
    """

    before = state["draft"].get("text", "")
    masked = mask_pii(before)
    state["draft"]["text"] = masked
    emit_metric(state, "validator_result", tags={"phase": "PII", "passed": before == masked, "fail_reason_code": "" if before == masked else "PII_MASKED", "success": True})
    return state


def validate_policy(state: AgentState) -> AgentState:
    """정책 위반 표현을 규칙 기반으로 점검/수정한다.

    Observability:
        `validator_result(phase=POLICY)`를 기록한다.
    """

    passed = "금지" not in state["draft"].get("text", "")
    if not passed:
        state["draft"]["text"] = state["draft"]["text"].replace("금지", "")
    emit_metric(state, "validator_result", tags={"phase": "POLICY", "passed": passed, "fail_reason_code": "" if passed else "POLICY_TERM", "success": True})
    return state


def validate_grounding(state: AgentState) -> AgentState:
    """근거(인용) 일치성을 검증한다.

    Rules:
        - AGENT 모드는 인용이 없으면 실패로 간주한다.
        - 실패 시 ASK_MORE로 안전 전환한다.

    Observability:
        `validator_result(phase=GROUNDING)`를 기록한다.
    """

    has_citation = bool(state["draft"].get("citations"))
    passed = state["plan"]["mode"] != "AGENT" or has_citation
    if not passed:
        state["decision"] = {"type": "ASK_MORE", "reason": "grounding_missing"}
    emit_metric(state, "validator_result", tags={"phase": "GROUNDING", "passed": passed, "fail_reason_code": "" if passed else "GROUNDING_NO_CITATION", "success": True})
    return state


def decision_node(state: AgentState) -> AgentState:
    """최종 Decision을 확정한다.

    우선순위:
        1) 기존 결정(HANDOFF/ASK_MORE/DRAFT) 유지
        2) 정책 허용 시 AUTO_POST
        3) 비허용 시 DRAFT

    Observability:
        `decision_made` 이벤트를 기록한다.
    """

    if state.get("decision", {}).get("type") in {"HANDOFF", "ASK_MORE", "DRAFT"}:
        emit_metric(state, "decision_made", tags={"decision": state["decision"]["type"], "success": True})
        return state
    if state["policy"].get("allow_auto_post", True):
        state["decision"] = {"type": "AUTO_POST"}
    else:
        state["decision"] = {"type": "DRAFT", "reason": "policy_restrict"}
    emit_metric(state, "decision_made", tags={"decision": state["decision"]["type"], "success": True})
    return state


def output_mask(state: AgentState) -> AgentState:
    """게시 전 최종 마스킹을 적용한다."""

    state["draft"]["text"] = mask_pii(state["draft"].get("text", ""))
    return state


def output_node(state: AgentState, tool_call) -> AgentState:
    """Decision에 따라 외부 출력을 수행한다.

    Side Effects:
        AUTO_POST/ASK_MORE일 때 `post_reply` MCP 도구 호출.

    Observability:
        `post_reply_result` 이벤트를 기록한다.
    """

    dtype = state["decision"]["type"]
    if dtype in {"AUTO_POST", "ASK_MORE"}:
        posted = tool_call("post_reply", {"case_id": state["input"]["case_id"], "content": state["draft"]["text"], "mode": dtype})
        state["decision"]["post_result"] = posted
        emit_metric(state, "post_reply_result", tags={"success": posted.get("posted", False), "error_code": ""})
    return state


def telemetry_logger(state: AgentState, logger) -> AgentState:
    """상태/메트릭을 영속화하는 로거 콜백을 호출한다."""

    logger("telemetry", state)
    return state
