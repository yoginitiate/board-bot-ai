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
        return "handoff" if state.get("decision", {}).get("type") == "HANDOFF" else "continue"

    graph.add_conditional_edges("ClassifyRiskRoute", risk_router, {"handoff": "TelemetryLogger", "continue": "PreVisionNeedGate"})
    graph.add_edge("PreVisionNeedGate", "PreVision")
    graph.add_edge("PreVision", "PolicyLoader")
    graph.add_edge("PolicyLoader", "PlanBuilder")
    graph.add_edge("PlanBuilder", "PlanEngine")
    graph.add_edge("PlanEngine", "ResponseModeGate")

    def mode_router(state: AgentState):
        return "agent" if state["plan"]["mode"] == "AGENT" else "scenario"

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
