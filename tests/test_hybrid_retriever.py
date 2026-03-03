from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pymilvus")

from board_bot.retrieval.hybrid import MilvusHybridRetriever, SparseVectorBuilder, Tokenizer


class DummyEmbedding:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.5, 0.0]


class FakeCollection:
    def __init__(self) -> None:
        self.called = False
        self.last_expr = ""

    def hybrid_search(self, requests, rerank, limit, output_fields, expr):
        self.called = True
        self.last_expr = expr
        w_sparse, w_dense = rerank._weights  # type: ignore[attr-defined]
        if w_dense > w_sparse:
            score = 0.91
            text = "dense 우세"
        elif w_sparse > w_dense:
            score = 0.89
            text = "sparse 우세"
        else:
            score = 0.90
            text = "균형"

        hit = SimpleNamespace(score=score, entity={"page_content": text, "source_type": "policy", "doc_id": "POL-1", "metadata": {"k": "v"}})
        return [[hit]]


def _build_retriever(dense_weight: float, sparse_weight: float) -> MilvusHybridRetriever:
    tok = Tokenizer(use_kiwi=False)
    builder = SparseVectorBuilder(tokenizer=tok)
    builder.build_vocab(["파손 환불 정책", "오배송 처리 매뉴얼"])
    return MilvusHybridRetriever(
        collection=FakeCollection(),
        embedding=DummyEmbedding(),
        sparse_builder=builder,
        final_top_k=3,
        search_limit_per_field=3,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )


def test_hybrid_search_path_and_hits():
    retriever = _build_retriever(dense_weight=0.4, sparse_weight=0.6)
    hits = retriever.hybrid_search("파손 환불", sources=["policy"], tenant_id="tenant-1", top_k=2)
    assert retriever.collection.called is True
    assert 'source_type == "policy"' in retriever.collection.last_expr
    assert 'tenant_id == "tenant-1"' in retriever.collection.last_expr
    assert hits and hits[0]["text"] == "sparse 우세"


def test_dense_sparse_hybrid_result_differs():
    dense = _build_retriever(dense_weight=1.0, sparse_weight=0.0).hybrid_search("오배송")
    sparse = _build_retriever(dense_weight=0.0, sparse_weight=1.0).hybrid_search("오배송")
    hybrid = _build_retriever(dense_weight=0.5, sparse_weight=0.5).hybrid_search("오배송")

    assert dense[0]["text"] != sparse[0]["text"]
    assert hybrid[0]["text"] == "균형"
