"""Master 그래프 노드 모듈.

이 모듈은 상담 입력을 안전하게 정규화하고,
분류/리스크 판정/사전 비전 필요성/정책 로딩/실행계획(Plan) 수립까지를 담당한다.

설계 의도:
- 고비용/고위험 처리를 초기에 분기해 후속 노드 비용을 절감한다.
- 분류 정확도를 위해 Milvus intent 예문 검색 결과를 LLM 컨텍스트로 주입한다.
- PII 원문은 절대 로깅하지 않고 마스킹 텍스트만 후속 체인에 전달한다.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from pymilvus import Collection, connections

from board_bot.llm.factory import create_embeddings
from board_bot.models.state import AgentState
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
    """입력 정규화 + PII 마스킹만 수행한다.

    Args:
        state: LangGraph 공유 상태.

    Returns:
        `input` 공백 정규화와 `safety.masked_text`가 반영된 상태.

    Side Effects:
        telemetry 버퍼에 `case_received` 메트릭을 기록한다.

    Security:
        고위험 판정은 이 단계에서 수행하지 않는다(요구사항 준수).
        이후 체인에는 마스킹 텍스트만 전달한다.

    Observability:
        case 수신 이벤트를 metrics_events로 적재할 수 있도록 버퍼링한다.
    """

    body = re.sub(r"\s+", " ", state["input"]["body"]).strip()
    title = re.sub(r"\s+", " ", state["input"]["title"]).strip()
    masked = mask_pii(f"{title}\n{body}")
    state["input"]["body"] = body
    state["input"]["title"] = title
    state["safety"] = {"masked_text": masked, "pii_masked": True}
    emit_metric(state, "case_received", tags={"success": True})
    return state


def _retrieve_intent_examples(state: AgentState, top_k: int) -> str:
    """Milvus `intent_examples`에서 분류 보조 예문을 검색한다.

    Args:
        state: 실행 상태(설정/마스킹 텍스트 포함).
        top_k: 검색할 예문 개수.

    Returns:
        LLM 프롬프트 컨텍스트에 삽입할 예문 문자열.

    Raises:
        없음. 예외는 내부에서 흡수하고 fallback 예문을 반환한다.

    Side Effects:
        Milvus 네트워크 I/O.

    Security:
        검색 질의는 마스킹 텍스트를 사용한다.
    """

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
    """LLM 미구성 환경 fallback 분류 결과를 생성한다."""

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
    """의도 분류 + 고위험 판정을 수행한다.

    Args:
        state: 마스킹 텍스트와 설정이 포함된 상태.
        llm: provider별 Chat LLM 인스턴스.

    Returns:
        `classification` 결과와 필요 시 `decision=HANDOFF`가 반영된 상태.

    Side Effects:
        - Milvus intent 예문 검색 I/O
        - telemetry에 `classification_completed`, `llm_usage` 기록

    Security:
        위험 신호는 intent 목록이 아닌 `risk` 필드에만 기록한다.

    Observability:
        mismatch/risk/multi-issue 태그를 메트릭으로 남긴다.
    """

    rule_risk = risk_score_rule(state["safety"]["masked_text"])
    top_k = int(state.get("config", {}).get("classify", {}).get("top_k", 6))
    context = _retrieve_intent_examples(state, top_k)

    if llm is None:
        cls = _dummy_classification(state, rule_risk)
        emit_metric(state, "llm_usage", tags={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "success": True})
    else:
        parsed = classification_chain(llm, state["safety"]["masked_text"], context)
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
    """이미지 사전판단 실행 여부를 결정한다."""

    issues = state.get("classification", {}).get("issues", [])
    has_attachment = len(state["input"].get("attachments", [])) > 0
    enabled_intents = set(state.get("config", {}).get("prevision", {}).get("enabled_intents", []))
    need = has_attachment and any(i.get("intent", {}).get("label") in enabled_intents for i in issues)
    state.setdefault("signals", {})["need_prevision"] = need
    return state


def prevision(state: AgentState, tool_call) -> AgentState:
    """MCP vision_triage를 조건부 호출해 이미지 신호를 저장한다."""

    cfg = state.get("config", {}).get("prevision", {})
    max_images = int(cfg.get("max_images", 3))
    threshold = float(cfg.get("confidence_threshold", 0.65))
    if not state.get("signals", {}).get("need_prevision"):
        state.setdefault("signals", {})["vision"] = {"label": "unknown", "confidence": 0.0, "unknown_reason": "not_required"}
        return state

    attachments = state["input"].get("attachments", [])[:max_images]
    res = tool_call("vision_triage", {"attachments": attachments, "timeout_s": int(cfg.get("timeout_s", 20)), "retry": int(cfg.get("retry", {}).get("max_attempts", 1))})
    confidence = float(max(res.get("damage", 0.0), res.get("misdelivery", 0.0), 1.0 - float(res.get("unclear", 0.0))))
    label = "unknown"
    if confidence >= threshold:
        if res.get("damage", 0.0) >= max(res.get("misdelivery", 0.0), res.get("unclear", 0.0)):
            label = "파손"
        elif res.get("misdelivery", 0.0) >= max(res.get("damage", 0.0), res.get("unclear", 0.0)):
            label = "오배송"
        else:
            label = "불명확"

    state.setdefault("signals", {})["vision"] = {"label": label, "confidence": confidence, "unknown_reason": "low_confidence" if label == "unknown" else "", "raw": res}
    return state


def policy_loader(state: AgentState, policy_map: dict[str, Any]) -> AgentState:
    """유형/상세유형 기반 운영 정책을 로드한다."""

    primary = state.get("classification", {}).get("primary_intent", {})
    key = f"{primary.get('type', state['input']['type'])}::{primary.get('subtype', state['input']['subtype'])}"
    state["policy"] = policy_map.get(key, {"mode": "SCENARIO", "allow_auto_post": True})
    return state


def plan_builder(state: AgentState, llm: BaseChatModel | None = None) -> AgentState:
    """SCENARIO/AGENT 실행 계획(Plan DSL)을 생성한다.

    Why:
        - SCENARIO: 정형 문의를 LLM 없이 빠르게 처리해 비용을 절감한다.
        - AGENT: 복합 케이스에서 도구 연계를 허용하되 예산으로 제어한다.
        - 독립 이슈는 병렬 실행 후보로 표시해 응답 지연을 줄인다.
    """

    if state.get("decision", {}).get("type") == "HANDOFF":
        state["plan"] = {"steps": [], "budget": {}, "use_prevision": False}
        return state

    mode = state["policy"].get("mode", "SCENARIO")
    issues = state.get("classification", {}).get("issues", [])[:3]
    masked = state["safety"]["masked_text"]
    slots = slot_extraction_chain(llm, masked) if llm else {"product_name": "", "quantity": "", "date": "", "order_status": "", "shipping_status": ""}
    state.setdefault("tooling", {})["slots"] = slots

    independent = len({i["intent"]["label"] for i in issues}) == len(issues) and len(issues) > 1
    if mode == "SCENARIO":
        steps = [{"tool": "rag_search", "args": {"query": masked, "sources": ["policy", "manual", "script"], "top_k": 5}}, {"tool": "get_shipping", "args": {"order_id": ""}}]
    else:
        steps = [{"tool": "rag_search", "args": {"query": masked}}, {"tool": "get_order", "args": {}}, {"tool": "get_shipping", "args": {}}]

    state["plan"] = {
        "mode": mode,
        "issues": issues,
        "steps": steps,
        "subplans": [{"issue_id": i["issue_id"], "steps": steps} for i in issues],
        "max_parallel": 3 if independent else 1,
        "independent_issues": independent,
        "use_prevision": bool(state.get("signals", {}).get("need_prevision")),
        # 운영 안정성: 과도한 도구 호출/장시간 체류를 방지하는 budget 한도
        "budget": {"max_tools": 6, "max_tokens": 4000, "max_time_ms": 6000, "retry": 1, "circuit_breaker": 2},
    }
    return state


def handoff_notify(state: AgentState, tool_call, repo) -> AgentState:
    """고위험 HANDOFF 알림을 idempotent 하게 발송한다.

    Side Effects:
        - MCP `notify_handoff` 호출
        - Postgres `handoff_notifications` 기록
    """

    case_id = state["input"]["case_id"]
    if repo.is_handoff_notified(case_id):
        state.setdefault("decision", {}).setdefault("notify", {"status": "skipped", "reason": "already_notified"})
        return state
    payload = {"case_id": case_id, "tenant_id": state["input"].get("tenant_id"), "reason": state.get("decision", {}).get("reason", "high_risk"), "tags": state.get("decision", {}).get("tags", [])}
    result = tool_call("notify_handoff", payload)
    repo.mark_handoff_notified(case_id, payload)
    state.setdefault("decision", {})["notify"] = result
    return state
