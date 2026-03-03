"""외부 운영 PostgreSQL 코퍼스 기반 BM25 vocab/idf 생성 스크립트.

왜 필요한가?
- Milvus hybrid_search의 sparse_vector는 질의/문서 모두 동일 vocab/idf 기준이 필요하다.
- vocab/idf를 Postgres(board_bot.t_rag_vocab, board_bot.t_rag_corpus_stats)에 저장하면
  운영에서 fixed/rebuild 전략을 제어할 수 있다.

운영 주의:
- rebuild 모드는 기존 vocab를 교체하므로 검색 점수가 바뀔 수 있다.
- 운영 반영 전 샘플 질의로 회귀 검증을 수행해야 한다.
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import create_engine, text

from board_bot.retrieval.hybrid import SparseVectorBuilder, Tokenizer
from board_bot.services.config import AppConfig


def main() -> None:
    cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")
    dsn = cfg.get("postgres", "dsn", default="postgresql+psycopg://user:password@db-host:5432/boardbot")
    vocab_mode = str(cfg.get("rag", "hybrid", "vocab", "mode", default="fixed"))
    use_kiwi = bool(cfg.get("retriever", "use_kiwi", default=True))

    engine = create_engine(dsn, future=True)
    with engine.begin() as conn:
        docs = conn.execute(text("SELECT content FROM board_bot.t_rag_documents ORDER BY created_at")).scalars().all()

    if not docs:
        raise RuntimeError("board_bot.t_rag_documents에 문서가 없어 vocab를 생성할 수 없습니다.")

    builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=use_kiwi))
    vocab, idf, stats = builder.build_vocab(list(docs))

    # df는 토큰화 결과로 재계산
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(builder.tokenizer.tokenize(d)))

    with engine.begin() as conn:
        if vocab_mode == "rebuild":
            conn.execute(text("DELETE FROM board_bot.t_rag_vocab"))

        # fixed 모드: 이미 있으면 유지(없을 때만 insert)
        for term, idx in vocab.items():
            conn.execute(
                text(
                    """
                    INSERT INTO board_bot.t_rag_vocab(term, term_index, df, idf, updated_at)
                    VALUES (:t,:i,:df,:idf,now())
                    ON CONFLICT (term) DO UPDATE
                    SET term_index = EXCLUDED.term_index,
                        df = EXCLUDED.df,
                        idf = EXCLUDED.idf,
                        updated_at = now()
                    """
                ),
                {"t": term, "i": idx, "df": int(df.get(term, 1)), "idf": float(idf[term])},
            )

        conn.execute(
            text(
                """
                INSERT INTO board_bot.t_rag_corpus_stats(id, doc_count, avgdl, updated_at)
                VALUES (1,:dc,:avg,now())
                ON CONFLICT (id) DO UPDATE
                SET doc_count=:dc, avgdl=:avg, updated_at=now()
                """
            ),
            {"dc": stats.doc_count, "avg": stats.avgdl},
        )

    print(f"BM25 vocab build completed. terms={len(vocab)} mode={vocab_mode} avgdl={stats.avgdl:.3f}")


if __name__ == "__main__":
    main()
