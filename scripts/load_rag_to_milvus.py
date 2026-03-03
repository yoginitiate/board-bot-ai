"""외부 Postgres 코퍼스를 외부 Milvus 하이브리드 컬렉션으로 적재한다.

튜닝 포인트:
- dense metric/nprobe/limit은 config.hybrid_search.dense에서 조정
- sparse BM25 생성은 Postgres vocab(idf) 기준으로 생성

운영 주의:
- 대량 적재 시 배치 크기/flush 주기 조정 필요
- 운영 컬렉션 잠금/재색인 비용을 고려해 비업무 시간대 수행 권장
"""

from __future__ import annotations

import json

from pymilvus import Collection, connections
from sqlalchemy import create_engine, text

from board_bot.llm.factory import create_embeddings
from board_bot.retrieval.hybrid import SparseVectorBuilder, Tokenizer
from board_bot.services.config import AppConfig


def _load_vocab(engine):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT term, term_index, idf FROM board_bot.t_rag_vocab ORDER BY term_index")).mappings().all()
        st = conn.execute(text("SELECT avgdl FROM board_bot.t_rag_corpus_stats WHERE id=1")).mappings().first()
    if not rows:
        raise RuntimeError("board_bot.t_rag_vocab이 비어 있습니다. scripts/build_bm25_vocab.py를 먼저 실행하세요.")
    vocab = {r["term"]: int(r["term_index"]) for r in rows}
    idf = {r["term"]: float(r["idf"]) for r in rows}
    return vocab, idf, float(st["avgdl"]) if st else 0.0


def main() -> None:
    cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")

    dsn = cfg.get("postgres", "dsn", default="postgresql+psycopg://user:password@db-host:5432/boardbot")
    milvus_host = cfg.get("milvus", "host", default="127.0.0.1")
    milvus_port = cfg.get("milvus", "port", default=19530)
    collection_name = cfg.get("milvus", "collection_name", default="semas_v2")
    provider = cfg.get("llm", "provider", default="openai")
    embedding_model = cfg.get("hybrid_search", "embedding_model", default=cfg.get("llm", "embedding_model", default="text-embedding-3-large"))
    api_key = cfg.get("llm", "api_key", default="")
    use_kiwi = bool(cfg.get("retriever", "use_kiwi", default=True))

    engine = create_engine(dsn, future=True)
    with engine.begin() as conn:
        docs = conn.execute(
            text(
                """
                SELECT doc_id, tenant_id, source_type, content, metadata_jsonb
                FROM board_bot.t_rag_documents
                ORDER BY created_at
                """
            )
        ).mappings().all()
    if not docs:
        raise RuntimeError("board_bot.t_rag_documents가 비어 있습니다.")

    vocab, idf, avgdl = _load_vocab(engine)
    sparse_builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=use_kiwi), vocab=vocab, idf=idf, avgdl=avgdl)

    emb = create_embeddings(provider=provider, model=embedding_model, api_key=api_key) if api_key else None
    dense_vectors = emb.embed_documents([d["content"] for d in docs]) if emb else [[0.0] * 8 for _ in docs]
    sparse_vectors = [sparse_builder.build_doc_vector(d["content"]) for d in docs]

    connections.connect(alias="default", host=str(milvus_host), port=str(milvus_port))
    coll = Collection(collection_name)
    coll.load()

    ids = [f"{d['tenant_id'] or 'default'}:{d['doc_id']}" for d in docs]
    entities = [
        ids,
        [d.get("tenant_id") or "default" for d in docs],
        [d["source_type"] for d in docs],
        [d["doc_id"] for d in docs],
        [d["content"] for d in docs],
        [d["metadata_jsonb"] if isinstance(d["metadata_jsonb"], dict) else json.loads(d["metadata_jsonb"] or "{}") for d in docs],
        dense_vectors,
        sparse_vectors,
    ]
    coll.upsert(entities)
    coll.flush()
    print(f"loaded docs={len(docs)} into {collection_name}")


if __name__ == "__main__":
    main()
