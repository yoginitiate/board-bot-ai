from __future__ import annotations

from typing import Any

from board_bot.models.state import AgentState
from board_bot.utils.safety import mask_pii


def plan_engine(state: AgentState, tool_call) -> AgentState:
    results = []
    budget = state["plan"]["budget"]
    for idx, step in enumerate(state["plan"]["steps"]):
        if idx >= budget["max_tools"]:
            state["decision"] = {"type": "DRAFT", "reason": "budget_exceeded"}
            break
        results.append({"tool": step["tool"], "result": tool_call(step["tool"], step.get("args", {}))})
    state.setdefault("tooling", {})["trace"] = results
    return state


def response_mode_gate(state: AgentState) -> AgentState:
    state.setdefault("draft", {})["mode"] = state["plan"].get("mode", "SCENARIO")
    return state


def scenario_subgraph(state: AgentState) -> AgentState:
    sections = []
    for issue in state["classification"]["issues"]:
        sections.append(f"[{issue['issue_id']}] {issue['summary']} 관련 안내드립니다.")
    state["draft"]["text"] = "\n".join(sections)
    return state


def agent_generate_subgraph(state: AgentState) -> AgentState:
    trace = state.get("tooling", {}).get("trace", [])
    citations = [t["result"].get("sources", []) for t in trace if isinstance(t.get("result"), dict)]
    state["draft"]["text"] = f"문의 검토 결과입니다.\n{state['safety']['masked_text'][:120]}"
    state["draft"]["citations"] = citations
    return state


def validate_pii(state: AgentState) -> AgentState:
    masked = mask_pii(state["draft"].get("text", ""))
    state["draft"]["text"] = masked
    return state


def validate_policy(state: AgentState) -> AgentState:
    if "금지" in state["draft"].get("text", ""):
        state["draft"]["text"] = state["draft"]["text"].replace("금지", "")
    return state


def validate_grounding(state: AgentState) -> AgentState:
    has_citation = bool(state["draft"].get("citations"))
    if state["plan"]["mode"] == "AGENT" and not has_citation:
        state["decision"] = {"type": "ASK_MORE", "reason": "grounding_missing"}
    return state


def decision_node(state: AgentState) -> AgentState:
    if state.get("decision", {}).get("type") in {"HANDOFF", "ASK_MORE", "DRAFT"}:
        return state
    if state["policy"].get("allow_auto_post", True):
        state["decision"] = {"type": "AUTO_POST"}
    else:
        state["decision"] = {"type": "DRAFT", "reason": "policy_restrict"}
    return state


def output_mask(state: AgentState) -> AgentState:
    state["draft"]["text"] = mask_pii(state["draft"].get("text", ""))
    return state


def output_node(state: AgentState, tool_call) -> AgentState:
    dtype = state["decision"]["type"]
    if dtype in {"AUTO_POST", "ASK_MORE"}:
        posted = tool_call("post_reply", {"case_id": state["input"]["case_id"], "content": state["draft"]["text"], "mode": dtype})
        state["decision"]["post_result"] = posted
    return state


def telemetry_logger(state: AgentState, logger) -> AgentState:
    logger("telemetry", state)
    return state
