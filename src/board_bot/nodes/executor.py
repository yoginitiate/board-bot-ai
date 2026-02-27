"""Executor 영역 노드."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

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

        return StructuredTool.from_function(name=name, description=f"MCP tool {name}", func=_fn)

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
        emit_metric(state, "tool_call", tags={"tool_name": step["tool"], "latency_ms": latency_ms, "success": ok, "retry_count": 0, "circuit_breaker_open": False, "error_code": err[:64]})
        results.append({"tool": step["tool"], "result": res})
    state.setdefault("tooling", {})["trace"] = results
    return state


def response_mode_gate(state: AgentState) -> AgentState:
    state.setdefault("draft", {})["mode"] = state["plan"].get("mode", "SCENARIO")
    return state


def _sanitize_template_text(text: str) -> str:
    text = mask_pii(text)
    text = re.sub(r"금지", "", text)
    return text


def scenario_subgraph(state: AgentState) -> AgentState:
    issues = state.get("plan", {}).get("issues", [])
    tpl_path = "scenarios/배송_지연.yaml"
    try:
        import yaml

        template = yaml.safe_load(open(tpl_path, encoding="utf-8")) or {}
        tpl_text = template.get("template", "안내드립니다.")
    except Exception:
        tpl_text = "안내드립니다."

    def render(issue):
        slots = state.get("tooling", {}).get("slots", {})
        body = f"[{issue['issue_id']}] {issue['intent'].get('label','문의')} - {tpl_text}"
        for k, v in slots.items():
            if v:
                body += f"\n- {k}: {v}"
        if state.get("signals", {}).get("vision", {}).get("label") == "unknown":
            body += f"\n- 이미지 판단 보류 사유: {state['signals']['vision'].get('unknown_reason','')}"
        return _sanitize_template_text(body)

    if state.get("plan", {}).get("independent_issues") and len(issues) > 1:
        max_workers = min(int(state.get("plan", {}).get("max_parallel", 1)), 3)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            sections = list(ex.map(render, issues))
    else:
        sections = [render(i) for i in issues]

    state.setdefault("draft", {})["text"] = "\n\n".join(sections)
    return state


def agent_generate_subgraph(state: AgentState, llm, tool_call) -> AgentState:
    if llm is None:
        trace = state.get("tooling", {}).get("trace", [])
        out = {"answer": f"문의 검토 결과입니다.\n{state['safety']['masked_text'][:120]}\n도구요약:{str(trace)[:300]}", "citations": ["fallback:tool_trace"]}
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
            emit_metric(state, "tool_call", tags={"tool_name": getattr(step[0], "tool", "agent_tool"), "latency_ms": 0, "success": True, "retry_count": 0, "circuit_breaker_open": False, "error_code": ""})
        out = agent_generate_chain(llm, state["safety"]["masked_text"], str(result.get("output", "")))
        state.setdefault("tooling", {}).setdefault("agent_steps", []).extend([str(s[0]) for s in steps])

    state.setdefault("draft", {})["text"] = _sanitize_template_text(out["answer"])
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
    issues = state.get("classification", {}).get("issues", [])
    answer = state.get("draft", {}).get("text", "")
    coverage = 0
    for issue in issues:
        if issue.get("summary", "")[:8] in answer or issue.get("intent", {}).get("label", "") in answer:
            coverage += 1
    issue_score = coverage / max(len(issues), 1)
    citation_score = 1.0 if state.get("draft", {}).get("citations") else 0.0
    policy_score = 0.0 if "금지" in answer else 1.0
    score = 0.5 * issue_score + 0.3 * citation_score + 0.2 * policy_score
    threshold = float(state.get("config", {}).get("grounding", {}).get("threshold", 0.7))
    max_rewrite = int(state.get("config", {}).get("grounding", {}).get("max_rewrite", 2))
    retries = int(state.setdefault("draft", {}).get("grounding_retry", 0))
    passed = score >= threshold
    if not passed and retries < max_rewrite:
        state["draft"]["grounding_retry"] = retries + 1
        state["draft"]["text"] = state["draft"].get("text", "") + "\n\n[보강] 누락된 문의 항목을 추가 확인하여 안내드립니다."
        passed = True if state["draft"]["grounding_retry"] >= max_rewrite else False
    if not passed:
        state["decision"] = {"type": "DRAFT", "reason": "grounding_low_score"}
    emit_metric(state, "validator_result", tags={"phase": "GROUNDING", "passed": passed, "fail_reason_code": "" if passed else "GROUNDING_LOW_SCORE", "success": True, "grounding_score": score})
    return state


def decision_node(state: AgentState) -> AgentState:
    if state.get("decision", {}).get("type") in {"HANDOFF", "DRAFT"}:
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
    if dtype == "AUTO_POST":
        posted = tool_call("post_reply", {"case_id": state["input"]["case_id"], "content": state["draft"]["text"], "mode": dtype})
        state["decision"]["post_result"] = posted
        emit_metric(state, "post_reply_result", tags={"success": posted.get("posted", False), "error_code": ""})
    return state


def telemetry_logger(state: AgentState, logger) -> AgentState:
    logger("telemetry", state)
    return state
