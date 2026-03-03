"""LangGraph 파이프라인 정의.

그래프는 Master(무엇을 할지)와 Executor(어떻게 할지)를 분리해 구성한다.
- Master: Normalize -> Classify/Risk -> (HANDOFF?) -> PreVision -> Policy -> Plan
- Executor: Plan 실행 -> Scenario/Agent 생성 -> Validator 3단계 -> Decision -> Output
"""

from __future__ import annotations

import time
from collections.abc import Callable

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
    handoff_notify,
    normalize_and_mask,
    term_extract,
    plan_builder,
    policy_loader,
    prevision,
    prevision_need_gate,
)
from board_bot.services.metrics import emit_metric


def _with_node_logging(node_name: str, fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    """모든 노드에 공통 실행 메트릭을 주입하는 wrapper.

    Observability:
        node_execution(metric_name)에 latency/success/error를 공통 포맷으로 기록한다.

    Security:
        PII 원문 대신 masked text 길이만 제한적으로 기록한다.
    """

    def _wrapped(state: AgentState) -> AgentState:
        started = time.perf_counter()
        ok = True
        err = ""
        try:
            return fn(state)
        except Exception as exc:  # pragma: no cover
            ok = False
            err = str(exc)
            raise
        finally:
            emit_metric(
                state,
                "node_execution",
                tags={
                    "node": node_name,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "success": ok,
                    "error_code": err[:64],
                    "masked_len": len(state.get("safety", {}).get("masked_text", "")[:256]),
                },
            )

    return _wrapped


def compile_graph(llm, tool_call, policy_map, logger, repo, glossary=None):
    """실행 가능한 LangGraph를 컴파일한다.

    Args:
        llm: provider별 Chat LLM 객체.
        tool_call: MCP 호출 함수.
        policy_map: 유형/상세유형 정책 맵.
        logger: 최종 상태 영속화 함수.
        repo: handoff idempotency 확인용 저장소 객체.

    Returns:
        컴파일된 graph runnable.

    Why:
        HANDOFF를 조기에 분기해 비용/리스크를 최소화하고,
        Validator를 3단계(PII->Policy->Grounding)로 분리해 안전성과 추적성을 높인다.
    """

    graph = StateGraph(AgentState)

    # Master nodes
    graph.add_node("NormalizeAndMask", _with_node_logging("NormalizeAndMask", normalize_and_mask))
    graph.add_node("TermExtract", _with_node_logging("TermExtract", lambda s: term_extract(s, glossary)))
    graph.add_node("ClassifyRiskRoute", _with_node_logging("ClassifyRiskRoute", lambda s: classify_risk_route(s, llm)))
    graph.add_node("HandoffNotify", _with_node_logging("HandoffNotify", lambda s: handoff_notify(s, tool_call, repo)))
    graph.add_node("PreVisionNeedGate", _with_node_logging("PreVisionNeedGate", prevision_need_gate))
    graph.add_node("PreVision", _with_node_logging("PreVision", lambda s: prevision(s, tool_call)))
    graph.add_node("PolicyLoader", _with_node_logging("PolicyLoader", lambda s: policy_loader(s, policy_map)))
    graph.add_node("PlanBuilder", _with_node_logging("PlanBuilder", lambda s: plan_builder(s, llm)))

    # Executor nodes
    graph.add_node("PlanEngine", _with_node_logging("PlanEngine", lambda s: plan_engine(s, tool_call)))
    graph.add_node("ResponseModeGate", _with_node_logging("ResponseModeGate", response_mode_gate))
    graph.add_node("ScenarioSubgraph", _with_node_logging("ScenarioSubgraph", scenario_subgraph))
    graph.add_node("AgentGenerateSubgraph", _with_node_logging("AgentGenerateSubgraph", lambda s: agent_generate_subgraph(s, llm, tool_call)))
    graph.add_node("ValidatePII", _with_node_logging("ValidatePII", validate_pii))
    graph.add_node("ValidatePolicy", _with_node_logging("ValidatePolicy", lambda s: validate_policy(s, llm)))
    graph.add_node("ValidateGrounding", _with_node_logging("ValidateGrounding", validate_grounding))
    graph.add_node("Decision", _with_node_logging("Decision", decision_node))
    graph.add_node("OutputMask", _with_node_logging("OutputMask", output_mask))
    graph.add_node("Output", _with_node_logging("Output", lambda s: output_node(s, tool_call)))
    graph.add_node("TelemetryLogger", _with_node_logging("TelemetryLogger", lambda s: telemetry_logger(s, logger)))

    graph.add_edge(START, "NormalizeAndMask")
    graph.add_edge("NormalizeAndMask", "TermExtract")
    graph.add_edge("TermExtract", "ClassifyRiskRoute")

    def risk_router(state: AgentState):
        """고위험 케이스 라우팅.

        HANDOFF는 후속 생성/검증을 생략하고 즉시 알림 노드로 보내 비용/리스크를 줄인다.
        """

        return "handoff" if state.get("decision", {}).get("type") == "HANDOFF" else "continue"

    graph.add_conditional_edges("ClassifyRiskRoute", risk_router, {"handoff": "HandoffNotify", "continue": "PreVisionNeedGate"})
    graph.add_edge("HandoffNotify", "TelemetryLogger")

    graph.add_edge("PreVisionNeedGate", "PreVision")
    graph.add_edge("PreVision", "PolicyLoader")
    graph.add_edge("PolicyLoader", "PlanBuilder")
    graph.add_edge("PlanBuilder", "PlanEngine")
    graph.add_edge("PlanEngine", "ResponseModeGate")

    def mode_router(state: AgentState):
        return "agent" if state["plan"]["mode"] == "AGENT" else "scenario"

    graph.add_conditional_edges("ResponseModeGate", mode_router, {"scenario": "ScenarioSubgraph", "agent": "AgentGenerateSubgraph"})

    # Validator를 3단계로 분리해 실패 지점을 명확히 추적/완화한다.
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
