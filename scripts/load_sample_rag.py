"""샘플 RAG 데이터를 Milvus + Postgres(스파스용)에 적재한다."""

from __future__ import annotations

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sqlalchemy import create_engine, text

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql+psycopg://postgres:postgres@localhost:5432/boardbot")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")

DOCS = {
    "policy": [
        ("P-001", "환불 정책: 수령 후 7일 이내 미사용 상품은 환불 가능합니다."),
        ("P-002", "파손 상품은 사진 확인 후 재배송 또는 환불을 안내합니다."),
    ],
    "manual": [
        ("M-001", "배송 지연 문의는 운송장 조회와 ETA 안내를 우선합니다."),
    ],
    "script": [
        ("S-001", "안녕하세요 고객님, 불편을 드려 죄송합니다. 확인 후 안내드리겠습니다."),
    ],
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    if API_KEY:
        emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=API_KEY)
        return emb.embed_documents(texts)
    return [[0.1] * 8 for _ in texts]


def upsert_milvus(source_type: str, docs: list[tuple[str, str]]):
    name = f"rag_{source_type}"
    dim = 768 if API_KEY else 8
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    if utility.has_collection(name):
        coll = Collection(name)
    else:
        coll = Collection(name=name, schema=CollectionSchema(fields=fields, description=f"{name} sample"))
        coll.create_index("embedding", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
    texts = [d[1] for d in docs]
    vectors = embed_texts(texts)
    coll.insert([
        [d[0] for d in docs],
        ["tenant-1" for _ in docs],
        [source_type for _ in docs],
        texts,
        [{"seed": True} for _ in docs],
        vectors,
    ])
    coll.flush()


def upsert_postgres(source_type: str, docs: list[tuple[str, str]]):
    engine = create_engine(POSTGRES_DSN, future=True)
    with engine.begin() as conn:
        for doc_id, content in docs:
            conn.execute(
                text(
                    """
                    INSERT INTO rag_documents(doc_id, tenant_id, source_type, content, metadata_jsonb)
                    VALUES (:doc_id, :tenant_id, :source_type, :content, '{"seed":true}'::jsonb)
                    ON CONFLICT (doc_id) DO UPDATE SET content=EXCLUDED.content, source_type=EXCLUDED.source_type
                    """
                ),
                {"doc_id": doc_id, "tenant_id": "tenant-1", "source_type": source_type, "content": content},
            )


def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    for st, docs in DOCS.items():
        upsert_milvus(st, docs)
        upsert_postgres(st, docs)
    print("sample rag loaded to milvus + postgres")


if __name__ == "__main__":
    main()
