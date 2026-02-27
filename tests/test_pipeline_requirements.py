from __future__ import annotations

import pytest

pytest.importorskip("langchain")

from board_bot.nodes.executor import decision_node
from board_bot.nodes.master import classify_risk_route, handoff_notify, prevision


class FakeRepo:
    def __init__(self):
        self.notified = set()

    def is_handoff_notified(self, case_id: str) -> bool:
        return case_id in self.notified

    def mark_handoff_notified(self, case_id: str, payload: dict):
        self.notified.add(case_id)


def _base_state(body: str = "배송이 안왔어요"):
    return {
        "input": {"tenant_id": "t1", "case_id": "c1", "type": "배송", "subtype": "지연", "title": "문의", "body": body, "attachments": []},
        "safety": {"masked_text": body, "pii_masked": True},
        "signals": {},
        "policy": {"allow_auto_post": True, "mode": "SCENARIO"},
        "tooling": {},
        "draft": {"text": "안내"},
        "decision": {},
        "telemetry": {"metrics": []},
        "config": {"prevision": {"max_images": 1, "confidence_threshold": 0.9}, "classify": {"top_k": 3}},
    }


def test_risk_not_in_intent_and_separated():
    st = _base_state("법적 조치 하겠습니다")
    out = classify_risk_route(st, llm=None)
    labels = [i["intent"]["label"] for i in out["classification"]["issues"]]
    assert "법적" not in " ".join(labels)
    assert out["classification"]["risk"]["level"] in {"LOW", "MEDIUM", "HIGH"}


def test_prevision_unknown_when_low_confidence():
    st = _base_state()
    st["classification"] = {"issues": [{"intent": {"label": "상품 파손"}}]}
    st["input"]["attachments"] = [{"path": "/tmp/not_exist.jpg"}]
    st["signals"] = {"need_prevision": True}

    def tool_call(name, args):
        return {"damage": 0.2, "misdelivery": 0.1, "unclear": 0.8}

    out = prevision(st, tool_call)
    assert out["signals"]["vision"]["label"] == "unknown"
    assert out["signals"]["vision"]["unknown_reason"]


def test_decision_node_has_no_ask_more_branch():
    st = _base_state()
    st["policy"]["allow_auto_post"] = False
    out = decision_node(st)
    assert out["decision"]["type"] in {"AUTO_POST", "DRAFT", "HANDOFF"}


def test_handoff_notify_idempotent():
    repo = FakeRepo()
    called = []

    def tool_call(name, args):
        called.append(name)
        return {"delivered": True}

    st = _base_state("고소하겠습니다")
    st["decision"] = {"type": "HANDOFF", "reason": "high_risk", "tags": ["법적"]}
    handoff_notify(st, tool_call, repo)
    handoff_notify(st, tool_call, repo)
    assert called.count("notify_handoff") == 1
