"""BM25 기반 sparse vector 생성기.

Milvus `SPARSE_FLOAT_VECTOR` 입력 형식(dict[int,float])을 만들기 위해
고정 vocab(index) + idf를 사용한다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")


@dataclass
class Bm25Stats:
    doc_count: int
    avgdl: float


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_vocab_and_idf(texts: list[str]) -> tuple[dict[str, int], dict[str, float], Bm25Stats]:
    docs = [tokenize(t) for t in texts]
    n = len(docs)
    if n == 0:
        return {}, {}, Bm25Stats(doc_count=0, avgdl=0.0)

    df: Counter[str] = Counter()
    total_len = 0
    for d in docs:
        total_len += len(d)
        df.update(set(d))

    vocab = {term: i for i, term in enumerate(sorted(df.keys()))}
    idf = {term: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for term, f in df.items()}
    return vocab, idf, Bm25Stats(doc_count=n, avgdl=(total_len / n) if n else 0.0)


def make_doc_sparse_vector(text: str, vocab: dict[str, int], idf: dict[str, float], avgdl: float, k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
    tokens = tokenize(text)
    tf = Counter(tokens)
    dl = len(tokens)
    out: dict[int, float] = {}
    for term, freq in tf.items():
        if term not in vocab:
            continue
        denom = freq + k1 * (1 - b + b * (dl / avgdl if avgdl else 1.0))
        score = idf.get(term, 0.0) * (freq * (k1 + 1)) / (denom if denom else 1.0)
        if score > 0:
            out[vocab[term]] = float(score)
    return out


def make_query_sparse_vector(query: str, vocab: dict[str, int], idf: dict[str, float]) -> dict[int, float]:
    tf = Counter(tokenize(query))
    out: dict[int, float] = {}
    for term, freq in tf.items():
        if term in vocab:
            out[vocab[term]] = float(idf.get(term, 0.0) * freq)
    return out
