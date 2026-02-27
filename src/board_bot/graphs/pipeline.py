"""Master/Executor 통합 LangGraph 파이프라인을 구성한다.

Why:
    노드 책임을 분리하고, 고위험 조기 종료/모드 분기/검증 게이트를
    선언적으로 관리하기 위해 StateGraph를 사용한다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from board_bot.models.state import AgentState
from board_bot.nodes.executor import (
    agent_generate_subgraph,
    decision_node,
    output_mask,
    output_node,
    plan_engine,
    response_mode_gate,
    scenario_subgraph,
    telemetry_logger,
    validate_grounding,
    validate_pii,
    validate_policy,
)
from board_bot.nodes.master import (
    classify_risk_route,
    normalize_and_mask,
    plan_builder,
    policy_loader,
    prevision,
    prevision_need_gate,
)


def compile_graph(llm, tool_call, policy_map, logger):
    """에이전트 실행 그래프를 컴파일한다.

    Args:
        llm: 분류/생성에 사용할 LLM 인스턴스(선택).
        tool_call: MCP 도구 호출 함수.
        policy_map: 유형별 정책 맵.
        logger: 최종 상태/메트릭 영속화 함수.

    Returns:
        컴파일된 LangGraph runnable 객체.

    Side Effects:
        없음(그래프 정의/컴파일만 수행).
    """

    graph = StateGraph(AgentState)

    graph.add_node("NormalizeAndMask", normalize_and_mask)
    graph.add_node("ClassifyRiskRoute", lambda s: classify_risk_route(s, llm))
    graph.add_node("PreVisionNeedGate", prevision_need_gate)
    graph.add_node("PreVision", lambda s: prevision(s, tool_call))
    graph.add_node("PolicyLoader", lambda s: policy_loader(s, policy_map))
    graph.add_node("PlanBuilder", plan_builder)
    graph.add_node("PlanEngine", lambda s: plan_engine(s, tool_call))
    graph.add_node("ResponseModeGate", response_mode_gate)
    graph.add_node("ScenarioSubgraph", scenario_subgraph)
    graph.add_node("AgentGenerateSubgraph", agent_generate_subgraph)
    graph.add_node("ValidatePII", validate_pii)
    graph.add_node("ValidatePolicy", validate_policy)
    graph.add_node("ValidateGrounding", validate_grounding)
    graph.add_node("Decision", decision_node)
    graph.add_node("OutputMask", output_mask)
    graph.add_node("Output", lambda s: output_node(s, tool_call))
    graph.add_node("TelemetryLogger", lambda s: telemetry_logger(s, logger))

    graph.add_edge(START, "NormalizeAndMask")
    graph.add_edge("NormalizeAndMask", "ClassifyRiskRoute")

    def risk_router(state: AgentState):
        """고위험(HANDOFF) 조기 종료 라우팅을 결정한다.

        Why:
            HANDOFF 케이스는 도구 호출/생성을 생략해 비용과 법적 리스크를 줄인다.
        """

        return "handoff" if state.get("decision", {}).get("type") == "HANDOFF" else "continue"

    graph.add_conditional_edges("ClassifyRiskRoute", risk_router, {"handoff": "TelemetryLogger", "continue": "PreVisionNeedGate"})
    graph.add_edge("PreVisionNeedGate", "PreVision")
    graph.add_edge("PreVision", "PolicyLoader")
    graph.add_edge("PolicyLoader", "PlanBuilder")
    graph.add_edge("PlanBuilder", "PlanEngine")
    graph.add_edge("PlanEngine", "ResponseModeGate")

    def mode_router(state: AgentState):
        """응답 모드 분기를 결정한다."""

        return "agent" if state["plan"]["mode"] == "AGENT" else "scenario"

    # 모드 분기 이후 동일한 Validator 체인을 통과시켜 안전성과 일관성을 확보한다.
    graph.add_conditional_edges("ResponseModeGate", mode_router, {"scenario": "ScenarioSubgraph", "agent": "AgentGenerateSubgraph"})
    graph.add_edge("ScenarioSubgraph", "ValidatePII")
    graph.add_edge("AgentGenerateSubgraph", "ValidatePII")
    graph.add_edge("ValidatePII", "ValidatePolicy")
    graph.add_edge("ValidatePolicy", "ValidateGrounding")
    graph.add_edge("ValidateGrounding", "Decision")
    graph.add_edge("Decision", "OutputMask")
    graph.add_edge("OutputMask", "Output")
    graph.add_edge("Output", "TelemetryLogger")
    graph.add_edge("TelemetryLogger", END)

    return graph.compile()
