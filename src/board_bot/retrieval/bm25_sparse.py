"""BM25 sparse vector 호환 함수.

기존 스크립트와의 호환을 위해 함수형 인터페이스를 유지하되,
내부 구현은 `retrieval.hybrid` 모듈을 사용한다.
"""

from __future__ import annotations

from .hybrid import SparseCorpusStats as Bm25Stats
from .hybrid import SparseVectorBuilder, Tokenizer


def tokenize(text: str, use_kiwi: bool = True) -> list[str]:
    return Tokenizer(use_kiwi=use_kiwi).tokenize(text)


def build_vocab_and_idf(texts: list[str], use_kiwi: bool = True) -> tuple[dict[str, int], dict[str, float], Bm25Stats]:
    builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=use_kiwi))
    return builder.build_vocab(texts)


def make_doc_sparse_vector(
    text: str,
    vocab: dict[str, int],
    idf: dict[str, float],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
    use_kiwi: bool = True,
) -> dict[int, float]:
    builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=use_kiwi), vocab=vocab, idf=idf, avgdl=avgdl)
    return builder.build_doc_vector(text, k1=k1, b=b)


def make_query_sparse_vector(query: str, vocab: dict[str, int], idf: dict[str, float], use_kiwi: bool = True) -> dict[int, float]:
    builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=use_kiwi), vocab=vocab, idf=idf)
    return builder.build_query_vector(query)
