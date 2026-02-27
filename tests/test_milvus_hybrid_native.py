from __future__ import annotations

import os

import pytest

pytest.importorskip("pymilvus")
from pymilvus import Collection, connections


@pytest.fixture(scope="module")
def coll():
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    try:
        connections.connect(alias="test", host=host, port=port)
        c = Collection(os.getenv("RAG_HYBRID_COLLECTION", "rag_hybrid_v2"), using="test")
        c.load()
        return c
    except Exception:
        pytest.skip("milvus not available for integration test")


def test_schema_has_dense_and_sparse(coll):
    fields = {f.name: f for f in coll.schema.fields}
    assert "dense_vector" in fields
    assert "sparse_vector" in fields


def test_index_types_exist(coll):
    idx = {i.field_name: i.to_dict() for i in coll.indexes}
    assert "dense_vector" in idx
    assert "sparse_vector" in idx


def test_hybrid_collection_non_empty(coll):
    assert coll.num_entities >= 1
