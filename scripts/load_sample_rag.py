"""샘플 RAG 데이터를 Milvus native hybrid_search 스키마로 적재한다.

- dense_vector(FLOAT_VECTOR)
- sparse_vector(SPARSE_FLOAT_VECTOR, BM25)
- vocab.mode=fixed: 기존 vocab 유지 후 미존재 토큰은 무시
- vocab.mode=rebuild: 코퍼스 전체로 vocab/idf 재생성
"""

from __future__ import annotations

import os

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sqlalchemy import create_engine, text

from board_bot.llm.factory import create_embeddings
from board_bot.retrieval.hybrid import SparseVectorBuilder, Tokenizer
from board_bot.services.config import AppConfig

cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")
MILVUS_HOST = cfg.get("milvus", "host", default="localhost")
MILVUS_PORT = cfg.get("milvus", "port", default=19530)
POSTGRES_DSN = cfg.get("postgres", "dsn", default="postgresql+psycopg://user:password@db-host:5432/boardbot")
PROVIDER = cfg.get("llm", "provider", default="google")
API_KEY = cfg.get("llm", "api_key", default="")
EMBED_MODEL = cfg.get("llm", "embedding_model", default="text-embedding-004")
COLLECTION = cfg.get("milvus", "collection_name", default="board_bot_rag_v2")

DOCS = [
    ("POL-001", "policy", "상품 파손 접수 시 사진 증빙이 있으면 환불/재배송을 우선 검토한다."),
    ("MAN-001", "manual", "오배송은 주문정보와 수령상품을 대조하여 책임소재를 판단한다."),
    ("SCR-001", "script", "고객 불편을 사과하고 현재 상황을 확인 중임을 안내한다."),
]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if API_KEY:
        emb = create_embeddings(provider=PROVIDER, model=EMBED_MODEL, api_key=API_KEY)
        return emb.embed_documents(texts)
    out: list[list[float]] = []
    for t in texts:
        v = [0.0] * 8
        for i, b in enumerate(t.encode("utf-8")):
            v[i % len(v)] += b / 255
        out.append(v)
    return out


def upsert_postgres_docs(engine) -> None:
    with engine.begin() as conn:
        for doc_id, source_type, content in DOCS:
            conn.execute(
                text(
                    "INSERT INTO board_bot.t_rag_documents(doc_id, tenant_id, source_type, content, metadata_jsonb) "
                    "VALUES (:doc_id, :tenant_id, :source_type, :content, '{\"seed\":true}'::jsonb) "
                    "ON CONFLICT (doc_id) DO UPDATE SET content=EXCLUDED.content, source_type=EXCLUDED.source_type"
                ),
                {"doc_id": doc_id, "tenant_id": "tenant-1", "source_type": source_type, "content": content},
            )


def load_existing_vocab(engine) -> tuple[dict[str, int], dict[str, float], float]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT term, term_index, idf FROM board_bot.t_rag_vocab ORDER BY term_index")).mappings().all()
        st = conn.execute(text("SELECT avgdl FROM board_bot.t_rag_corpus_stats WHERE id=1")).mappings().first()
    vocab = {r["term"]: int(r["term_index"]) for r in rows}
    idf = {r["term"]: float(r["idf"]) for r in rows}
    avgdl = float(st["avgdl"]) if st else 0.0
    return vocab, idf, avgdl


def persist_vocab(engine, builder: SparseVectorBuilder, texts: list[str], mode: str) -> tuple[dict[str, int], dict[str, float], float]:
    if mode == "fixed":
        vocab, idf, avgdl = load_existing_vocab(engine)
        if vocab:
            builder.vocab, builder.idf, builder.avgdl = vocab, idf, avgdl
            return vocab, idf, avgdl

    vocab, idf, stats = builder.build_vocab(texts)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM board_bot.t_rag_vocab"))
        for term, idx in vocab.items():
            conn.execute(
                text("INSERT INTO board_bot.t_rag_vocab(term, term_index, df, idf, updated_at) VALUES (:t,:i,1,:idf,now())"),
                {"t": term, "i": idx, "idf": float(idf[term])},
            )
        conn.execute(
            text(
                "INSERT INTO board_bot.t_rag_corpus_stats(id, doc_count, avgdl, updated_at) VALUES (1,:dc,:avg,now()) "
                "ON CONFLICT (id) DO UPDATE SET doc_count=:dc, avgdl=:avg, updated_at=now()"
            ),
            {"dc": stats.doc_count, "avg": stats.avgdl},
        )
    return vocab, idf, stats.avgdl


def recreate_hybrid_collection(dim: int) -> Collection:
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    coll = Collection(COLLECTION, schema=CollectionSchema(fields=fields, description="hybrid rag"))
    coll.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    coll.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    return coll


def main() -> None:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    engine = create_engine(POSTGRES_DSN, future=True)

    upsert_postgres_docs(engine)
    texts = [d[2] for d in DOCS]
    dense = embed_texts(texts)

    vocab_cfg = cfg.get("rag", "hybrid", "vocab", default={})
    builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=bool(cfg.get("retriever", "use_kiwi", default=True))))
    _, _, avgdl = persist_vocab(engine, builder=builder, texts=texts, mode=str(vocab_cfg.get("mode", "fixed")))
    builder.avgdl = avgdl

    sparse = [builder.build_doc_vector(t) for t in texts]

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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
