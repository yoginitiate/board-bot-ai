from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from board_bot.models.state import AgentState
from board_bot.services.prompts import load_prompt
from board_bot.utils.safety import mask_pii, risk_score_rule


def normalize_and_mask(state: AgentState) -> AgentState:
    body = state["input"]["body"].strip()
    title = state["input"]["title"].strip()
    masked = mask_pii(f"{title}\n{body}")
    state["safety"] = {"masked_text": masked, "pii_masked": True}
    return state


def _dummy_classification(state: AgentState, rule_risk: dict[str, Any]) -> dict[str, Any]:
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
    rule_risk = risk_score_rule(state["safety"]["masked_text"])
    prompt = load_prompt("prompts/classify_risk_route.yaml")
    if llm is None:
        cls = _dummy_classification(state, rule_risk)
    else:
        msg = HumanMessage(content=f"{prompt['system']}\nINPUT:\n{state['safety']['masked_text']}")
        raw = llm.invoke([msg]).content
        cls = json.loads(raw)
    if rule_risk["level"] == "HIGH" or cls["risk"]["level"] == "HIGH":
        state["decision"] = {"type": "HANDOFF", "reason": "high_risk", "tags": list(set(rule_risk["tags"] + cls["risk"].get("tags", [])))}
    state["classification"] = cls
    return state


def prevision_need_gate(state: AgentState) -> AgentState:
    state["signals"] = {"need_prevision": bool(state["classification"].get("needs", {}).get("prevision"))}
    return state


def prevision(state: AgentState, tool_call) -> AgentState:
    if not state["signals"].get("need_prevision"):
        return state
    result = tool_call("vision_triage", {"attachments": state["input"].get("attachments", [])})
    state["signals"]["vision"] = result
    return state


def policy_loader(state: AgentState, policy_map: dict[str, Any]) -> AgentState:
    key = f"{state['input']['type']}::{state['input']['subtype']}"
    state["policy"] = policy_map.get(key, {"mode": "SCENARIO", "allow_auto_post": True})
    return state


def plan_builder(state: AgentState) -> AgentState:
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
        "steps": steps,
        "budget": {"max_tools": 5, "max_tokens": 4000, "max_time_ms": 5000, "retry": 1, "circuit_breaker": 2},
    }
    return state
