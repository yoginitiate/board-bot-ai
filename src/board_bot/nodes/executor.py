"""Executor 영역 노드를 정의한다."""

from __future__ import annotations

import time
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from board_bot.models.state import AgentState
from board_bot.services.llm_chains import agent_generate_chain, policy_validation_chain
from board_bot.services.metrics import emit_metric
from board_bot.utils.safety import mask_pii


def _build_langchain_tools(tool_call):
    def _wrap(name: str):
        def _fn(**kwargs):
            return tool_call(name, kwargs)

        return StructuredTool.from_function(
            name=name,
            description=f"MCP tool {name}",
            func=_fn,
        )

    return [_wrap(n) for n in ["get_order", "get_shipping", "get_claims", "get_customer", "get_inventory", "rag_search"]]


def plan_engine(state: AgentState, tool_call) -> AgentState:
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
    state.setdefault("draft", {})["mode"] = state["plan"].get("mode", "SCENARIO")
    return state


def scenario_subgraph(state: AgentState) -> AgentState:
    sections = []
    for issue in state["classification"]["issues"]:
        sections.append(f"[{issue['issue_id']}] {issue['summary']} 관련 안내드립니다.")
    state["draft"]["text"] = "\n".join(sections)
    return state


def agent_generate_subgraph(state: AgentState, llm, tool_call) -> AgentState:
    if llm is None:
        trace = state.get("tooling", {}).get("trace", [])
        tool_summary = str(trace)[:500]
        out = {"answer": f"문의 검토 결과입니다.\n{state['safety']['masked_text'][:120]}\n도구요약:{tool_summary}", "citations": ["fallback"]}
    else:
        tools = _build_langchain_tools(tool_call)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 게시판 상담 에이전트다. 필요한 도구를 호출해 답변을 만든다."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=False)
        result = executor.invoke({"input": state["safety"]["masked_text"]})
        steps = result.get("intermediate_steps", [])
        for step in steps:
            emit_metric(state, "tool_call", tags={"tool_name": getattr(step[0], 'tool', 'agent_tool'), "latency_ms": 0, "success": True, "retry_count": 0, "circuit_breaker_open": False, "error_code": ""})
        out = agent_generate_chain(llm, state["safety"]["masked_text"], str(result.get("output", "")))
        state.setdefault("tooling", {}).setdefault("agent_steps", []).extend([str(s[0]) for s in steps])

    state["draft"]["text"] = out["answer"]
    state["draft"]["citations"] = out.get("citations", [])
    return state


def validate_pii(state: AgentState) -> AgentState:
    before = state["draft"].get("text", "")
    masked = mask_pii(before)
    state["draft"]["text"] = masked
    emit_metric(state, "validator_result", tags={"phase": "PII", "passed": before == masked, "fail_reason_code": "" if before == masked else "PII_MASKED", "success": True})
    return state


def validate_policy(state: AgentState, llm=None) -> AgentState:
    text = state["draft"].get("text", "")
    if llm is None:
        passed = "금지" not in text
        revised = text.replace("금지", "") if not passed else text
        fail_reason = "" if passed else "POLICY_TERM"
    else:
        parsed = policy_validation_chain(llm, text)
        passed = bool(parsed["passed"])
        revised = parsed.get("revised_text") or text
        fail_reason = parsed.get("fail_reason_code", "")

    state["draft"]["text"] = revised
    emit_metric(state, "validator_result", tags={"phase": "POLICY", "passed": passed, "fail_reason_code": fail_reason, "success": True})
    return state


def validate_grounding(state: AgentState) -> AgentState:
    has_citation = bool(state["draft"].get("citations"))
    passed = state["plan"]["mode"] != "AGENT" or has_citation
    if not passed:
        state["decision"] = {"type": "ASK_MORE", "reason": "grounding_missing"}
    emit_metric(state, "validator_result", tags={"phase": "GROUNDING", "passed": passed, "fail_reason_code": "" if passed else "GROUNDING_NO_CITATION", "success": True})
    return state


def decision_node(state: AgentState) -> AgentState:
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
    state["draft"]["text"] = mask_pii(state["draft"].get("text", ""))
    return state


def output_node(state: AgentState, tool_call) -> AgentState:
    dtype = state["decision"]["type"]
    if dtype in {"AUTO_POST", "ASK_MORE"}:
        posted = tool_call("post_reply", {"case_id": state["input"]["case_id"], "content": state["draft"]["text"], "mode": dtype})
        state["decision"]["post_result"] = posted
        emit_metric(state, "post_reply_result", tags={"success": posted.get("posted", False), "error_code": ""})
    return state


def telemetry_logger(state: AgentState, logger) -> AgentState:
    logger("telemetry", state)
    return state
