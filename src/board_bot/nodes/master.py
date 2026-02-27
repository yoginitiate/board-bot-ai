"""Master 영역 노드를 정의한다.

Master는 "무엇을 할지"를 결정한다. 즉, 입력 정규화/PII 마스킹, 분류+리스크,
사전 비전 필요성, 정책 로딩, Plan DSL 생성까지 수행한다.

Observability:
    주요 단계에서 `metrics_events` 적재용 텔레메트리 이벤트를 버퍼링한다.
Security:
    LLM 호출에는 반드시 마스킹 텍스트를 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from board_bot.models.state import AgentState
from board_bot.services.metrics import emit_metric
from board_bot.services.prompts import load_prompt
from board_bot.utils.safety import mask_pii, risk_score_rule


def normalize_and_mask(state: AgentState) -> AgentState:
    """입력을 정규화하고 1차 PII 마스킹을 수행한다.

    Args:
        state: LangGraph 공유 상태.

    Returns:
        마스킹 결과(`safety.masked_text`)가 반영된 상태.

    Side Effects:
        - `case_received` 메트릭 이벤트를 telemetry 버퍼에 추가한다.

    Security/Privacy:
        - 이후 노드/LLM은 원문이 아닌 `masked_text`를 사용해야 한다.
    """

    body = state["input"]["body"].strip()
    title = state["input"]["title"].strip()
    masked = mask_pii(f"{title}\n{body}")
    state["safety"] = {"masked_text": masked, "pii_masked": True}
    emit_metric(state, "case_received", tags={"success": True})
    return state


def _dummy_classification(state: AgentState, rule_risk: dict[str, Any]) -> dict[str, Any]:
    """LLM 미사용 환경에서 분류 결과 스텁을 생성한다.

    Why:
        네트워크/키 부재 환경에서도 파이프라인 회귀 테스트가 가능해야 하므로,
        스키마 호환 더미 결과를 고정 형태로 반환한다.
    """

    issue = {
        "issue_id": "I1",
        "summary": state["input"]["title"][:40],
        "span": {"start": 0, "end": len(state["safety"]["masked_text"])},
        "intent": {"type": state["input"]["type"], "subtype": state["input"]["subtype"], "confidence": 0.82},
        "required_tools": ["rag_search", "post_reply"],
        "missing_slots": [],
        "needs_image": len(state["input"].get("attachments", [])) > 0,
        "needs_order_ref": "주문" in state["safety"]["masked_text"],
        "risk": rule_risk,
    }
    return {
        "primary_intent": issue["intent"],
        "secondary_intents": [],
        "is_multi_issue": False,
        "issues": [issue],
        "mismatch": {"is_mismatch": False, "reason": ""},
        "risk": {
            "level": rule_risk["level"],
            "tags": rule_risk["tags"],
            "handoff_required": rule_risk["level"] == "HIGH",
            "handoff_reason": "rule_high" if rule_risk["level"] == "HIGH" else "",
        },
        "needs": {"prevision": issue["needs_image"]},
        "answer_strategy": {"composition_hint": "SECTIONED_REPLY", "mode_hint": "SCENARIO"},
    }


def classify_risk_route(state: AgentState, llm: ChatGoogleGenerativeAI | None = None) -> AgentState:
    """분류/리스크/라우팅을 단일 단계로 수행한다.

    Args:
        state: LangGraph 공유 상태.
        llm: Gemini LLM 인스턴스. 없으면 더미 분류를 사용한다.

    Returns:
        `classification` 결과와 필요 시 `decision=HANDOFF`가 반영된 상태.

    Side Effects:
        - `llm_usage`, `classification_completed` 이벤트를 telemetry에 적재.

    Security/Privacy:
        - LLM 입력은 `safety.masked_text`만 사용한다.
        - risk_tags/mismatch_reason 등 요약 코드만 저장하고 원문은 저장하지 않는다.

    Observability:
        - `classification_completed`에 mismatch/risk/multi-issue 태그를 기록한다.

    Raises:
        json.JSONDecodeError: LLM 응답이 JSON 스키마를 따르지 않는 경우.
    """

    rule_risk = risk_score_rule(state["safety"]["masked_text"])
    prompt = load_prompt("prompts/classify_risk_route.yaml")
    if llm is None:
        cls = _dummy_classification(state, rule_risk)
        emit_metric(state, "llm_usage", tags={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "success": True})
    else:
        msg = HumanMessage(content=f"{prompt['system']}\nINPUT:\n{state['safety']['masked_text']}")
        raw = llm.invoke([msg]).content
        cls = json.loads(raw)
        emit_metric(state, "llm_usage", tags={"prompt_tokens": 200, "completion_tokens": 150, "total_tokens": 350, "estimated_cost_usd": 0.001, "success": True})

    # 2단 게이트 취지: 규칙/LLM 중 하나라도 HIGH면 즉시 HANDOFF 하여 리스크 확산을 방지한다.
    if rule_risk["level"] == "HIGH" or cls["risk"]["level"] == "HIGH":
        state["decision"] = {"type": "HANDOFF", "reason": "high_risk", "tags": list(set(rule_risk["tags"] + cls["risk"].get("tags", [])))}
    state["classification"] = cls
    emit_metric(
        state,
        "classification_completed",
        tags={
            "is_mismatch": cls["mismatch"]["is_mismatch"],
            "mismatch_reason": cls["mismatch"].get("reason", ""),
            "risk_level": cls["risk"]["level"],
            "risk_tags": ",".join(cls["risk"].get("tags", [])),
            "handoff_required": cls["risk"].get("handoff_required", False),
            "is_multi_issue": cls["is_multi_issue"],
            "issues_count": len(cls["issues"]),
            "composition_hint": cls["answer_strategy"].get("composition_hint", "SECTIONED_REPLY"),
        },
    )
    return state


def prevision_need_gate(state: AgentState) -> AgentState:
    """사전 비전 분석 필요 여부를 경량 규칙으로 결정한다.

    Why:
        모든 케이스에서 비전/OCR을 호출하면 비용과 지연이 커지므로,
        분류 결과의 `needs.prevision` 신호로 선별 호출한다.
    """

    state["signals"] = {"need_prevision": bool(state["classification"].get("needs", {}).get("prevision"))}
    return state


def prevision(state: AgentState, tool_call) -> AgentState:
    """비전 선판단 도구를 호출해 추가 신호를 상태에 기록한다.

    Args:
        state: LangGraph 공유 상태.
        tool_call: MCP tool 호출 함수.

    Returns:
        `signals.vision`이 추가된 상태(필요 시).

    Side Effects:
        외부 MCP 네트워크 I/O가 발생할 수 있다.
    """

    if not state["signals"].get("need_prevision"):
        return state
    result = tool_call("vision_triage", {"attachments": state["input"].get("attachments", [])})
    state["signals"]["vision"] = result
    return state


def policy_loader(state: AgentState, policy_map: dict[str, Any]) -> AgentState:
    """유형/상세유형 키로 정책을 로딩한다.

    Why:
        정책을 그래프 외부 주입형으로 유지하면 테넌트별 운영 변경 시 코드 배포를 최소화할 수 있다.
    """

    key = f"{state['input']['type']}::{state['input']['subtype']}"
    state["policy"] = policy_map.get(key, {"mode": "SCENARIO", "allow_auto_post": True})
    return state


def plan_builder(state: AgentState) -> AgentState:
    """Plan DSL을 구성한다.

    Args:
        state: 분류/정책이 반영된 상태.

    Returns:
        `plan`이 채워진 상태.

    Side Effects:
        없음(계획 생성만 수행).

    Why:
        - HANDOFF는 비용/리스크 절감을 위해 계획 생성을 생략한다.
        - SCENARIO는 정적 템플릿 기반으로 단순/저비용 처리한다.
        - AGENT는 복합 케이스에서 확장 가능성을 위해 도구 실행 단계를 포함한다.
        - budget(max_tools/max_tokens/max_time_ms/retry/circuit_breaker)은
          무한 재시도/과도한 호출을 방지하는 안전 장치다.
    """

    if state.get("decision", {}).get("type") == "HANDOFF":
        state["plan"] = {"steps": [], "budget": {}}
        return state

    mode = state["classification"]["answer_strategy"].get("mode_hint", "SCENARIO")
    scenario_path = Path("scenarios") / f"{state['input']['type']}_{state['input']['subtype']}.yaml"
    if mode == "SCENARIO" and scenario_path.exists():
        steps = [{"tool": "rag_search", "args": {"query": state["safety"]["masked_text"]}}]
    else:
        steps = [{"tool": "rag_search", "args": {"query": state["safety"]["masked_text"]}}, {"tool": "post_reply", "args": {"dry_run": True}}]
        mode = "AGENT"

    state["plan"] = {
        "mode": mode,
        "issues": state["classification"]["issues"],
        # Plan DSL 핵심 필드: steps(실행 목록), budget(한도), 향후 guards/on_fail 확장 여지.
        "steps": steps,
        "budget": {"max_tools": 5, "max_tokens": 4000, "max_time_ms": 5000, "retry": 1, "circuit_breaker": 2},
    }
    return state
