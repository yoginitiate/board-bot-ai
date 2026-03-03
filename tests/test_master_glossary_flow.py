from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("pymilvus")

from board_bot.nodes.master import plan_builder, term_extract


class DummyGlossary:
    def extract_terms(self, question: str, max_terms: int = 8, description_max_chars: int = 120):
        return [{"term": "역배송", "description": "반품 회수 절차"}]

    def get_synonyms(self, term: str, max_synonyms: int = 3):
        return ["회수배송"]


def _state():
    return {
        "input": {"tenant_id": "t1", "case_id": "c1", "type": "배송", "subtype": "지연", "title": "문의", "body": "역배송", "attachments": []},
        "safety": {"masked_text": "역배송 요청", "pii_masked": True},
        "signals": {},
        "policy": {"allow_auto_post": True, "mode": "SCENARIO"},
        "tooling": {},
        "draft": {"text": "안내"},
        "decision": {},
        "telemetry": {"metrics": []},
        "config": {"glossary": {"enabled": True, "max_terms": 8, "max_synonyms_per_term": 3, "expand": {"for_dense": True, "for_sparse": True}}, "classify": {"top_k": 3}},
        "classification": {"issues": [{"issue_id": "I1", "intent": {"label": "상품 미배송"}}]},
    }


def test_term_extract_and_plan_builder_expand_queries():
    st = _state()
    st = term_extract(st, DummyGlossary())
    assert st["glossary"]["terms"][0]["term"] == "역배송"

    out = plan_builder(st, llm=None)
    rag_step = out["plan"]["steps"][0]
    assert rag_step["tool"] == "rag_search"
    assert "dense_query_text" in rag_step["args"]
    assert "sparse_query_text" in rag_step["args"]
    assert "회수배송" in rag_step["args"]["sparse_query_text"]
