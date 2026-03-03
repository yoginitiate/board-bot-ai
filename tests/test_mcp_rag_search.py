from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastmcp")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mcp_server.app import main as mcp_main


class StubRetriever:
    sparse_weight = 0.6
    dense_weight = 0.4
    search_limit_per_field = 30
    usage = type("Usage", (), {"query_calls": 1})()

    def hybrid_search(self, query: str, sources, tenant_id, top_k):
        assert query
        return [
            {
                "text": "환불 규정 안내",
                "source_type": "policy",
                "doc_id": "POL-001",
                "score": 0.95,
                "metadata": {"seed": True},
            }
        ]


def test_rag_search_returns_hits(monkeypatch):
    monkeypatch.setattr(mcp_main, "_get_hybrid_retriever", lambda: StubRetriever())
    out = mcp_main.rag_search(query="환불", sources=["policy"], top_k=3, tenant_id="tenant-1")
    assert out["hits"][0]["doc_id"] == "POL-001"
    assert out["debug"]["path"] == "milvus_native_hybrid_search"
