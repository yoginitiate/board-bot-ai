"""Master 그래프 노드 모듈."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from pymilvus import Collection, connections

from board_bot.llm.factory import create_embeddings
from board_bot.models.state import AgentState
from board_bot.services.glossary import GlossaryService
from board_bot.services.llm_chains import ALLOWED_INTENTS, classification_chain, slot_extraction_chain
from board_bot.services.metrics import emit_metric
from board_bot.utils.safety import mask_pii, risk_score_rule

_INTENT_TO_TYPE = {
    "상품 누락": ("주문", "누락"),
    "상품 오배송": ("주문", "오배송"),
    "상품 미배송": ("배송", "미배송"),
    "품절": ("주문", "품절"),
    "상품 파손": ("클레임", "파손"),
    "주문 취소": ("주문", "취소"),
}


def normalize_and_mask(state: AgentState) -> AgentState:
    body = re.sub(r"\s+", " ", state["input"]["body"]).strip()
    title = re.sub(r"\s+", " ", state["input"]["title"]).strip()
    masked = mask_pii(f"{title}\n{body}")
    state["input"]["body"] = body
    state["input"]["title"] = title
    state["safety"] = {"masked_text": masked, "pii_masked": True}
    emit_metric(state, "case_received", tags={"success": True})
    return state


def term_extract(state: AgentState, glossary: GlossaryService | None = None) -> AgentState:
    """마스킹 텍스트에서 용어를 추출해 state.glossary에 저장한다."""

    cfg = state.get("config", {}).get("glossary", {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled or glossary is None:
        state["glossary"] = {"terms": [], "normalized_terms_used": False}
        emit_metric(state, "glossary_extract", tags={"term_count": 0, "success": True})
        return state

    max_terms = int(cfg.get("max_terms", 8))
    max_synonyms = int(cfg.get("max_synonyms_per_term", 3))
    desc_max = int(cfg.get("description_max_chars", 120))
    extracted = glossary.extract_terms(state.get("safety", {}).get("masked_text", ""), max_terms=max_terms, description_max_chars=desc_max)

    terms: list[dict[str, Any]] = []
    for item in extracted:
        syns = glossary.get_synonyms(item["term"], max_synonyms=max_synonyms)
        terms.append({"term": item["term"], "description": item["description"], "synonyms": syns})

    state["glossary"] = {"terms": terms, "normalized_terms_used": bool(terms)}
    emit_metric(state, "glossary_extract", tags={"term_count": len(terms), "success": True})
    return state


def _build_glossary_context(state: AgentState) -> tuple[str, str]:
    cfg = state.get("config", {}).get("glossary", {})
    terms = state.get("glossary", {}).get("terms", [])
    if not terms:
        return "", ""

    desc_max = int(cfg.get("description_max_chars", 120))
    max_syn = int(cfg.get("max_synonyms_per_term", 3))
    context_lines: list[str] = []
    expanded_bits: list[str] = []
    for t in terms:
        syns = (t.get("synonyms") or [])[:max_syn]
        context_lines.append(f"- {t.get('term','')}: {(t.get('description','') or '')[:desc_max]} (동의어: {', '.join(syns)})")
        expanded_bits.append(" ".join([t.get("term", ""), *syns]).strip())
    return "\n".join(context_lines), " ".join([b for b in expanded_bits if b]).strip()


def _retrieve_intent_examples(state: AgentState, top_k: int) -> str:
    cfg = state.get("config", {})
    host = cfg.get("milvus", {}).get("host", "milvus")
    port = cfg.get("milvus", {}).get("port", 19530)
    provider = cfg.get("llm", {}).get("provider", "google")
    api_key = cfg.get("llm", {}).get("api_key") or ""
    emb_model = cfg.get("llm", {}).get("embedding_model", "text-embedding-004")
    try:
        connections.connect(alias="intent", host=host, port=str(port))
        coll = Collection("intent_examples", using="intent")
        coll.load()
        if api_key:
            vec = create_embeddings(provider=provider, model=emb_model, api_key=api_key).embed_query(state["safety"]["masked_text"])
        else:
            vec = [0.1] * 8
        res = coll.search([vec], "embedding", {"metric_type": "COSINE", "params": {"nprobe": 10}}, limit=top_k, output_fields=["intent_label", "example_text"])
        lines = [f"- {h.entity.get('intent_label')}: {h.entity.get('example_text')}" for h in res[0]]
        return "\n".join(lines)
    except Exception:
        return "\n".join([f"- {i}: 샘플 예문" for i in ALLOWED_INTENTS[:top_k]])


def _dummy_classification(state: AgentState, rule_risk: dict[str, Any]) -> dict[str, Any]:
    label = "상품 미배송"
    issue = {
        "issue_id": "I1",
        "summary": state["input"]["title"][:40],
        "span": {"start": 0, "end": len(state["safety"]["masked_text"])},
        "intent": {"type": _INTENT_TO_TYPE[label][0], "subtype": _INTENT_TO_TYPE[label][1], "confidence": 0.7, "label": label},
        "required_tools": ["rag_search", "get_shipping", "post_reply"],
        "missing_slots": [],
        "needs_image": len(state["input"].get("attachments", [])) > 0,
        "needs_order_ref": True,
        "risk": rule_risk,
    }
    return {
        "primary_intent": issue["intent"],
        "secondary_intents": [],
        "is_multi_issue": False,
        "issues": [issue],
        "mismatch": {"is_mismatch": False, "reason": ""},
        "risk": {"level": rule_risk["level"], "tags": rule_risk["tags"], "handoff_required": rule_risk["level"] == "HIGH", "handoff_reason": "rule_high" if rule_risk["level"] == "HIGH" else ""},
        "needs": {"prevision": issue["needs_image"] and label in set(state.get("config", {}).get("prevision", {}).get("enabled_intents", []))},
        "answer_strategy": {"composition_hint": "SECTIONED_REPLY", "mode_hint": "SCENARIO"},
    }


def classify_risk_route(state: AgentState, llm: BaseChatModel | None = None) -> AgentState:
    rule_risk = risk_score_rule(state["safety"]["masked_text"])
    top_k = int(state.get("config", {}).get("classify", {}).get("top_k", 6))
    context = _retrieve_intent_examples(state, top_k)

    glossary_cfg = state.get("config", {}).get("glossary", {})
    inject_mode = glossary_cfg.get("injection", {}).get("mode", "append_context")
    glossary_context, glossary_expanded = _build_glossary_context(state)

    if llm is None:
        cls = _dummy_classification(state, rule_risk)
        emit_metric(state, "llm_usage", tags={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "success": True})
    else:
        expanded_text = f"{state['safety']['masked_text']} {glossary_expanded}".strip() if inject_mode == "query_expand" else ""
        parsed = classification_chain(
            llm,
            state["safety"]["masked_text"],
            context,
            glossary_context=glossary_context if inject_mode == "append_context" else "",
            expanded_query=expanded_text,
        )
        intents = parsed.get("intents", [])[:3]
        issues: list[dict[str, Any]] = []
        enabled_intents = set(state.get("config", {}).get("prevision", {}).get("enabled_intents", []))
        for idx, it in enumerate(intents or [{"label": "상품 미배송", "confidence": 0.5}], start=1):
            t, st = _INTENT_TO_TYPE[it["label"]]
            issues.append(
                {
                    "issue_id": f"I{idx}",
                    "summary": it["label"],
                    "span": {"start": 0, "end": len(state["safety"]["masked_text"])},
                    "intent": {"type": t, "subtype": st, "confidence": float(it.get("confidence", 0.0)), "label": it["label"]},
                    "required_tools": ["rag_search", "post_reply"],
                    "missing_slots": [],
                    "needs_image": len(state["input"].get("attachments", [])) > 0 and it["label"] in enabled_intents,
                    "needs_order_ref": it["label"] in {"상품 미배송", "상품 오배송", "상품 누락"},
                    "risk": parsed.get("risk", {}),
                }
            )
        cls = {
            "primary_intent": issues[0]["intent"],
            "secondary_intents": [i["intent"] for i in issues[1:]],
            "is_multi_issue": len(issues) > 1,
            "issues": issues,
            "mismatch": {"is_mismatch": False, "reason": ""},
            "risk": parsed.get("risk", {"level": rule_risk["level"], "tags": rule_risk["tags"], "handoff_required": False, "handoff_reason": ""}),
            "needs": {"prevision": any(i["needs_image"] for i in issues)},
            "answer_strategy": {"composition_hint": "SECTIONED_REPLY", "mode_hint": "SCENARIO"},
        }
        emit_metric(state, "llm_usage", tags={"prompt_tokens": 250, "completion_tokens": 180, "total_tokens": 430, "estimated_cost_usd": 0.0012, "success": True})

    if rule_risk["level"] == "HIGH" or cls["risk"]["level"] == "HIGH":
        state["decision"] = {"type": "HANDOFF", "reason": "high_risk", "tags": list(set(rule_risk.get("tags", []) + cls["risk"].get("tags", [])))}

    state["classification"] = cls
    emit_metric(state, "classification_completed", tags={"is_mismatch": cls["mismatch"]["is_mismatch"], "mismatch_reason": cls["mismatch"].get("reason", ""), "risk_level": cls["risk"]["level"], "risk_tags": ",".join(cls["risk"].get("tags", [])), "handoff_required": cls["risk"].get("handoff_required", False), "is_multi_issue": cls["is_multi_issue"], "issues_count": len(cls["issues"]), "composition_hint": cls["answer_strategy"].get("composition_hint", "SECTIONED_REPLY")})
    return state


def prevision_need_gate(state: AgentState) -> AgentState:
    issues = state.get("classification", {}).get("issues", [])
    has_attachment = len(state["input"].get("attachments", [])) > 0
    enabled_intents = set(state.get("config", {}).get("prevision", {}).get("enabled_intents", []))
    need = has_attachment and any(i.get("intent", {}).get("label") in enabled_intents for i in issues)
    state.setdefault("signals", {})["need_prevision"] = need
    return state


def prevision(state: AgentState, tool_call) -> AgentState:
    if not state.get("signals", {}).get("need_prevision"):
        state.setdefault("signals", {})["vision"] = {"label": "skipped", "confidence": 0.0}
        return state
    attachments = state["input"].get("attachments", [])[: int(state.get("config", {}).get("prevision", {}).get("max_images", 3))]
    out = tool_call("vision_triage", {"attachments": attachments, "timeout_s": state.get("config", {}).get("prevision", {}).get("timeout_s", 20), "retry": state.get("config", {}).get("prevision", {}).get("retry", {}).get("max_attempts", 1)})
    scores = {"damage": float(out.get("damage", 0)), "misdelivery": float(out.get("misdelivery", 0)), "unclear": float(out.get("unclear", 0))}
    label = max(scores, key=scores.get)
    conf = scores[label]
    thr = float(state.get("config", {}).get("prevision", {}).get("confidence_threshold", 0.65))
    if conf < thr or label == "unclear":
        state.setdefault("signals", {})["vision"] = {"label": "unknown", "confidence": conf, "unknown_reason": "low_confidence_or_unclear"}
    else:
        state.setdefault("signals", {})["vision"] = {"label": label, "confidence": conf}
    return state


def policy_loader(state: AgentState, policy_map: dict[str, Any]) -> AgentState:
    key = f"{state['input']['type']}::{state['input']['subtype']}"
    state["policy"] = policy_map.get(key, {"mode": "SCENARIO", "allow_auto_post": False})
    return state


def plan_builder(state: AgentState, llm: BaseChatModel | None = None) -> AgentState:
    mode = state.get("policy", {}).get("mode", "SCENARIO")
    issues = state.get("classification", {}).get("issues", [])
    masked = state.get("safety", {}).get("masked_text", "")

    slots = slot_extraction_chain(llm, masked) if llm is not None else {"product_name": "", "quantity": "", "date": "", "order_status": "", "shipping_status": ""}
    state.setdefault("tooling", {})["slots"] = slots

    glossary_cfg = state.get("config", {}).get("glossary", {})
    terms = state.get("glossary", {}).get("terms", [])
    dense_query_text = masked
    sparse_query_text = masked

    if glossary_cfg.get("enabled", True) and terms:
        if glossary_cfg.get("expand", {}).get("for_dense", True):
            dense_extra = " ".join([f"{t.get('term','')}:{(t.get('description','') or '')[:40]}" for t in terms])
            dense_query_text = f"{masked} {dense_extra}".strip()
        if glossary_cfg.get("expand", {}).get("for_sparse", True):
            sparse_extra = []
            for t in terms:
                sparse_extra.append(t.get("term", ""))
                sparse_extra.extend((t.get("synonyms") or [])[: int(glossary_cfg.get("max_synonyms_per_term", 3))])
            sparse_query_text = f"{masked} {' '.join([x for x in sparse_extra if x])}".strip()

    emit_metric(
        state,
        "rag_query_expansion",
        tags={
            "query_expanded_term_count": len(terms),
            "dense_query_length": len(dense_query_text),
            "sparse_query_length": len(sparse_query_text),
            "success": True,
        },
    )

    independent = len({i["intent"]["label"] for i in issues}) == len(issues) and len(issues) > 1
    rag_args = {
        "query": masked,
        "dense_query_text": dense_query_text,
        "sparse_query_text": sparse_query_text,
        "sources": ["policy", "manual", "script"],
        "top_k": 5,
    }
    if mode == "SCENARIO":
        steps = [{"tool": "rag_search", "args": rag_args}, {"tool": "get_shipping", "args": {"order_id": ""}}]
    else:
        steps = [{"tool": "rag_search", "args": rag_args}, {"tool": "get_order", "args": {}}, {"tool": "get_shipping", "args": {}}]

    state["plan"] = {
        "mode": mode,
        "issues": issues,
        "steps": steps,
        "subplans": [{"issue_id": i["issue_id"], "steps": steps} for i in issues],
        "max_parallel": 3 if independent else 1,
        "independent_issues": independent,
        "use_prevision": bool(state.get("signals", {}).get("need_prevision")),
        "budget": {"max_tools": 6, "max_tokens": 4000, "max_time_ms": 6000, "retry": 1, "circuit_breaker": 2},
    }
    return state


def handoff_notify(state: AgentState, tool_call, repo) -> AgentState:
    case_id = state["input"]["case_id"]
    if repo.is_handoff_notified(case_id):
        state.setdefault("decision", {}).setdefault("notify", {"status": "skipped", "reason": "already_notified"})
        return state
    payload = {"case_id": case_id, "tenant_id": state["input"].get("tenant_id"), "reason": state.get("decision", {}).get("reason", "high_risk"), "tags": state.get("decision", {}).get("tags", [])}
    result = tool_call("notify_handoff", payload)
    repo.mark_handoff_notified(case_id, payload)
    state.setdefault("decision", {})["notify"] = result
    return state
