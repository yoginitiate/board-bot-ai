"""Milvus Native Hybrid Retriever.

요구사항:
- dense_vector + sparse_vector 동시 검색
- AnnSearchRequest + Collection.hybrid_search + WeightedRanker 사용
- 한국어 토큰화(BM25)와 provider 독립 sparse 생성
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pymilvus import AnnSearchRequest, Collection, RRFRanker, WeightedRanker


@dataclass
class Tokenizer:
    """한국어 토큰화기.

    - use_kiwi=True 이고 kiwipiepy가 설치된 경우 Kiwi 형태소 분석 사용
    - 그렇지 않으면 정규식 기반 fallback 사용
    """

    use_kiwi: bool = True
    _token_re: re.Pattern[str] = field(default_factory=lambda: re.compile(r"[0-9A-Za-z가-힣_]+"), init=False)

    def __post_init__(self) -> None:
        self._kiwi = None
        if self.use_kiwi and importlib.util.find_spec("kiwipiepy") is not None:
            kiwi_mod = importlib.import_module("kiwipiepy")
            self._kiwi = kiwi_mod.Kiwi()

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        if self._kiwi is not None:
            return [t.form.lower() for t in self._kiwi.tokenize(text) if getattr(t, "form", "")]
        return [t.lower() for t in self._token_re.findall(text)]


@dataclass
class SparseCorpusStats:
    doc_count: int
    avgdl: float


class SparseVectorBuilder:
    """BM25 기반 sparse vector 생성기.

    vocab 고정/재빌드 전략을 지원한다.
    """

    def __init__(self, tokenizer: Tokenizer, vocab: dict[str, int] | None = None, idf: dict[str, float] | None = None, avgdl: float = 0.0) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab or {}
        self.idf = idf or {}
        self.avgdl = avgdl

    def build_vocab(self, texts: list[str]) -> tuple[dict[str, int], dict[str, float], SparseCorpusStats]:
        tokenized = [self.tokenizer.tokenize(t) for t in texts]
        n = len(tokenized)
        if n == 0:
            self.vocab, self.idf, self.avgdl = {}, {}, 0.0
            return self.vocab, self.idf, SparseCorpusStats(doc_count=0, avgdl=0.0)

        df: Counter[str] = Counter()
        total_len = 0
        for toks in tokenized:
            total_len += len(toks)
            df.update(set(toks))

        vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        idf = {term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}
        avgdl = total_len / n if n else 0.0

        self.vocab, self.idf, self.avgdl = vocab, idf, avgdl
        return vocab, idf, SparseCorpusStats(doc_count=n, avgdl=avgdl)

    def build_doc_vector(self, text: str, k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
        tf = Counter(self.tokenizer.tokenize(text))
        dl = sum(tf.values())
        out: dict[int, float] = {}
        for term, freq in tf.items():
            if term not in self.vocab:
                continue
            denom = freq + k1 * (1 - b + b * (dl / self.avgdl if self.avgdl else 1.0))
            score = self.idf.get(term, 0.0) * (freq * (k1 + 1)) / (denom if denom else 1.0)
            if score > 0:
                out[self.vocab[term]] = float(score)
        return out

    def build_query_vector(self, query: str) -> dict[int, float]:
        tf = Counter(self.tokenizer.tokenize(query))
        out: dict[int, float] = {}
        for term, freq in tf.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            out[idx] = float(self.idf.get(term, 0.0) * freq)
        return out


@dataclass
class EmbeddingUsage:
    query_calls: int = 0
    last_dense_query_length: int = 0
    last_sparse_query_length: int = 0


@dataclass
class MilvusHybridRetriever:
    """Milvus 공식 hybrid_search 방식 retriever."""

    collection: Collection
    embedding: Any
    sparse_builder: SparseVectorBuilder
    final_top_k: int = 10
    search_limit_per_field: int = 30
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    metric_type: str = "IP"
    dense_search_params: dict[str, Any] = field(default_factory=dict)
    sparse_search_params: dict[str, Any] = field(default_factory=dict)
    output_fields: list[str] = field(default_factory=lambda: ["page_content", "source_type", "doc_id", "metadata", "tenant_id"])
    ranker_type: str = "weighted"
    rrf_k: int = 60
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)

    def _build_expr(self, sources: list[str] | None, tenant_id: str | None) -> str:
        exprs: list[str] = []
        if sources:
            src_expr = " or ".join([f'source_type == "{s}"' for s in sources])
            exprs.append(f"({src_expr})")
        if tenant_id:
            exprs.append(f'(tenant_id == "{tenant_id}")')
        return " and ".join(exprs)

    def hybrid_search(self, query: str, sources: list[str] | None = None, tenant_id: str | None = None, top_k: int | None = None, dense_query_text: str | None = None, sparse_query_text: str | None = None) -> list[dict[str, Any]]:
        dense_text = dense_query_text or query
        sparse_text = sparse_query_text or query
        dense_query = self.embedding.embed_query(dense_text)
        self.usage.query_calls += 1
        self.usage.last_dense_query_length = len(dense_text)
        self.usage.last_sparse_query_length = len(sparse_text)
        sparse_query = self.sparse_builder.build_query_vector(sparse_text)

        req_dense = AnnSearchRequest(
            data=[dense_query],
            anns_field="dense_vector",
            param={"metric_type": self.metric_type, "params": self.dense_search_params},
            limit=self.search_limit_per_field,
        )
        req_sparse = AnnSearchRequest(
            data=[sparse_query],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": self.sparse_search_params},
            limit=self.search_limit_per_field,
        )

        reranker = RRFRanker(self.rrf_k) if self.ranker_type.lower() == "rrf" else WeightedRanker(self.sparse_weight, self.dense_weight)
        result = self.collection.hybrid_search(
            [req_sparse, req_dense],
            rerank=reranker,
            limit=top_k or self.final_top_k,
            output_fields=self.output_fields,
            expr=self._build_expr(sources=sources, tenant_id=tenant_id),
        )

        hits: list[dict[str, Any]] = []
        for hit in result[0]:
            entity = hit.entity
            page_content = entity.get("page_content") or entity.get("text") or ""
            metadata = entity.get("metadata") or {}
            hits.append(
                {
                    "text": page_content,
                    "source_type": entity.get("source_type"),
                    "doc_id": entity.get("doc_id"),
                    "score": float(hit.score),
                    "metadata": {**metadata, "hybrid_score": float(hit.score)},
                }
            )
        return hits

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """LangChain 의존성 없이 문서 dict 리스트를 반환한다."""
        return self.hybrid_search(
            query=query,
            sources=kwargs.get("sources"),
            tenant_id=kwargs.get("tenant_id"),
            top_k=kwargs.get("top_k"),
            dense_query_text=kwargs.get("dense_query_text"),
            sparse_query_text=kwargs.get("sparse_query_text"),
        )
