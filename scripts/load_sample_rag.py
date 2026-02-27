"""샘플 RAG 데이터를 Milvus native hybrid_search 스키마로 적재한다.

- 단일 컬렉션 `rag_hybrid_v2` 사용
- dense_vector + sparse_vector 동시 저장
- sparse vocab/idf는 Postgres(rag_vocab, rag_corpus_stats)에 저장
"""

from __future__ import annotations

import os
from collections import Counter

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sqlalchemy import create_engine, text

from board_bot.llm.factory import create_embeddings
from board_bot.retrieval.bm25_sparse import build_vocab_and_idf, make_doc_sparse_vector, tokenize

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql+psycopg://postgres:postgres@localhost:5432/boardbot")
PROVIDER = os.getenv("LLM_PROVIDER", "google")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
COLLECTION = os.getenv("RAG_HYBRID_COLLECTION", "rag_hybrid_v2")

DOCS = [
    ("P-001", "policy", "환불 정책: 수령 후 7일 이내 미사용 상품은 환불 가능합니다."),
    ("P-002", "policy", "파손 상품은 사진 확인 후 재배송 또는 환불을 안내합니다."),
    ("M-001", "manual", "배송 지연 문의는 운송장 조회와 ETA 안내를 우선합니다."),
    ("S-001", "script", "안녕하세요 고객님, 불편을 드려 죄송합니다. 확인 후 안내드리겠습니다."),
]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if API_KEY:
        return create_embeddings(provider=PROVIDER, model=EMBED_MODEL, api_key=API_KEY).embed_documents(texts)
    return [[0.1] * 8 for _ in texts]


def persist_vocab(engine, texts: list[str]):
    vocab, idf, stats = build_vocab_and_idf(texts)
    doc_tokens = [tokenize(t) for t in texts]
    df = Counter()
    for dt in doc_tokens:
        df.update(set(dt))

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM rag_vocab"))
        for term, idx in vocab.items():
            conn.execute(
                text("INSERT INTO rag_vocab(term, term_index, df, idf, updated_at) VALUES (:t,:i,:df,:idf,now())"),
                {"t": term, "i": idx, "df": int(df[term]), "idf": float(idf[term])},
            )
        conn.execute(
            text("INSERT INTO rag_corpus_stats(id, doc_count, avgdl, updated_at) VALUES (1,:dc,:avg,now()) ON CONFLICT (id) DO UPDATE SET doc_count=:dc, avgdl=:avg, updated_at=now()"),
            {"dc": stats.doc_count, "avg": stats.avgdl},
        )
    return vocab, idf, stats


def upsert_postgres_docs(engine):
    with engine.begin() as conn:
        for doc_id, source_type, content in DOCS:
            conn.execute(
                text(
                    "INSERT INTO rag_documents(doc_id, tenant_id, source_type, content, metadata_jsonb) VALUES (:doc_id, :tenant_id, :source_type, :content, '{\"seed\":true}'::jsonb) ON CONFLICT (doc_id) DO UPDATE SET content=EXCLUDED.content, source_type=EXCLUDED.source_type"
                ),
                {"doc_id": doc_id, "tenant_id": "tenant-1", "source_type": source_type, "content": content},
            )


def recreate_hybrid_collection(dim: int):
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    coll = Collection(COLLECTION, schema=CollectionSchema(fields=fields, description="hybrid rag"))
    coll.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    coll.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    return coll


def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    engine = create_engine(POSTGRES_DSN, future=True)

    upsert_postgres_docs(engine)
    texts = [d[2] for d in DOCS]
    dense = embed_texts(texts)
    vocab, idf, stats = persist_vocab(engine, texts)
    sparse = [make_doc_sparse_vector(t, vocab=vocab, idf=idf, avgdl=stats.avgdl) for t in texts]

    coll = recreate_hybrid_collection(len(dense[0]))
    coll.insert(
        [
            [d[0] for d in DOCS],
            ["tenant-1" for _ in DOCS],
            [d[1] for d in DOCS],
            texts,
            [{"seed": True} for _ in DOCS],
            dense,
            sparse,
        ]
    )
    coll.flush()
    print("sample rag loaded to milvus native hybrid collection")


if __name__ == "__main__":
    main()
