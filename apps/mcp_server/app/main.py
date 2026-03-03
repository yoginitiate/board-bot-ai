"""FastMCP 도구 서버.

Milvus 공식 native hybrid_search(dense_vector + sparse_vector) 방식으로 RAG를 수행한다.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from pymilvus import Collection, connections
from sqlalchemy import create_engine, text

from board_bot.llm.factory import create_embeddings, multimodal_triage
from board_bot.retrieval.hybrid import MilvusHybridRetriever, SparseVectorBuilder, Tokenizer
from board_bot.services.config import AppConfig

mcp = FastMCP("board-bot-mcp")
cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")

MILVUS_HOST = cfg.get("milvus", "host", default="milvus")
MILVUS_PORT = cfg.get("milvus", "port", default=19530)
POSTGRES_DSN = cfg.get("postgres", "dsn", default="postgresql+psycopg://user:password@db-host:5432/boardbot")
PROVIDER = cfg.get("llm", "provider", default="google")
API_KEY = cfg.get("llm", "api_key", default="")
EMBED_MODEL = cfg.get("llm", "embedding_model", default="text-embedding-004")
MM_MODEL = cfg.get("llm", "mm_model", default="gemini-2.0-flash")
COLLECTION = cfg.get("milvus", "collection_name", default=cfg.get("rag", "collection_name", default="board_bot_rag_v2"))

_engine = create_engine(POSTGRES_DSN, future=True)
_embeddings = create_embeddings(provider=PROVIDER, model=EMBED_MODEL, api_key=API_KEY) if API_KEY else None
_connections_ok = False
_retriever: MilvusHybridRetriever | None = None


class _FallbackEmbedding:
    """API Key가 없을 때 테스트/로컬 동작용 임베딩."""

    @staticmethod
    def embed_query(query: str) -> list[float]:
        base = [0.0] * 8
        for i, ch in enumerate(query.encode("utf-8")):
            base[i % len(base)] += ch / 255
        return base


def _ensure_milvus() -> None:
    global _connections_ok
    if not _connections_ok:
        connections.connect(alias="default", host=str(MILVUS_HOST), port=str(MILVUS_PORT))
        _connections_ok = True


def _load_vocab() -> tuple[dict[str, int], dict[str, float], float]:
    with _engine.begin() as conn:
        rows = conn.execute(text("SELECT term, term_index, idf FROM board_bot.t_rag_vocab ORDER BY term_index")).mappings().all()
        stats = conn.execute(text("SELECT avgdl FROM board_bot.t_rag_corpus_stats WHERE id=1")).mappings().first()
    vocab = {r["term"]: int(r["term_index"]) for r in rows}
    idf = {r["term"]: float(r["idf"]) for r in rows}
    avgdl = float(stats["avgdl"]) if stats else 0.0
    return vocab, idf, avgdl


def _get_hybrid_retriever() -> MilvusHybridRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever

    _ensure_milvus()
    coll = Collection(COLLECTION)
    coll.load()

    hcfg = cfg.get("hybrid_search", default={})
    tok_cfg = cfg.get("retriever", default={})
    use_kiwi = bool(tok_cfg.get("use_kiwi", True))

    vocab, idf, avgdl = _load_vocab()
    sparse_builder = SparseVectorBuilder(tokenizer=Tokenizer(use_kiwi=use_kiwi), vocab=vocab, idf=idf, avgdl=avgdl)

    ranker_cfg = hcfg.get("ranker", {}) if isinstance(hcfg.get("ranker", {}), dict) else {}
    weights = ranker_cfg.get("weights", [0.5, 0.5])
    sparse_weight = float(weights[0]) if len(weights) > 0 else 0.5
    dense_weight = float(weights[1]) if len(weights) > 1 else 0.5

    dense_cfg = hcfg.get("dense", {}) if isinstance(hcfg.get("dense", {}), dict) else {}
    sparse_cfg = hcfg.get("sparse", {}) if isinstance(hcfg.get("sparse", {}), dict) else {}
    per_field = int(max(dense_cfg.get("limit", 10), sparse_cfg.get("limit", 10)))
    final_top_k = int(cfg.get("rag", "llm_docs_limit", default=5))

    _retriever = MilvusHybridRetriever(
        collection=coll,
        embedding=_embeddings or _FallbackEmbedding(),
        sparse_builder=sparse_builder,
        final_top_k=final_top_k,
        search_limit_per_field=per_field,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        metric_type=str(dense_cfg.get("metric_type", "COSINE")),
        dense_search_params={"nprobe": int(dense_cfg.get("nprobe", 10))},
        sparse_search_params={"k": int(sparse_cfg.get("k", 10)), "inverted_index_algo": str(sparse_cfg.get("inverted_index_algo", "DAAT_MAXSCORE"))},
        ranker_type=str(ranker_cfg.get("type", "weighted")),
        rrf_k=int(ranker_cfg.get("rrf_k", 60)),
        output_fields=list(cfg.get("milvus", "output_fields", default=["page_content", "source_type", "doc_id", "metadata", "tenant_id"])),
    )
    return _retriever


@mcp.tool()
def rag_search(query: str, sources: list[str] | None = None, top_k: int = 10, tenant_id: str | None = None, dense_query_text: str | None = None, sparse_query_text: str | None = None) -> dict[str, Any]:
    """Milvus native hybrid_search를 수행한다."""

    retriever = _get_hybrid_retriever()
    srcs = sources or cfg.get("rag", "sources_default", default=["policy", "manual", "script"])
    effective_top_k = min(int(top_k), int(cfg.get("rag", "llm_docs_limit", default=5)))
    try:
        hits = retriever.hybrid_search(
            query=query,
            sources=srcs,
            tenant_id=tenant_id,
            top_k=effective_top_k,
            dense_query_text=dense_query_text,
            sparse_query_text=sparse_query_text,
        )
    except Exception as exc:
        return {"ok": False, "reason": f"milvus_hybrid_search_failed:{str(exc)[:180]}", "hits": [], "sources": []}

    return {
        "hits": hits,
        "sources": [f"{h['source_type']}:{h['doc_id']}" for h in hits],
        "debug": {
            "path": "milvus_native_hybrid_search",
            "collection": COLLECTION,
            "sparse_weight": retriever.sparse_weight,
            "dense_weight": retriever.dense_weight,
            "per_field_limit": retriever.search_limit_per_field,
            "llm_docs_limit": int(cfg.get("rag", "llm_docs_limit", default=5)),
            "embedding_query_calls": retriever.usage.query_calls,
            "dense_query_length": retriever.usage.last_dense_query_length,
            "sparse_query_length": retriever.usage.last_sparse_query_length,
        },
    }


def _download_if_needed(url_or_path: str) -> Path:
    p = Path(url_or_path)
    if p.exists():
        return p
    if url_or_path.startswith("http"):
        r = httpx.get(url_or_path, timeout=15)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        Path(tmp).write_bytes(r.content)
        return Path(tmp)
    raise FileNotFoundError(url_or_path)


@mcp.tool()
def vision_triage(attachments: list[dict[str, Any]] | None = None, timeout_s: int = 20, retry: int = 1) -> dict[str, Any]:
    attachments = attachments or []
    if not attachments or not API_KEY:
        return {"damage": 0.0, "misdelivery": 0.0, "unclear": 1.0, "notes": "no_image_or_api_key"}

    paths: list[str] = []
    for att in attachments:
        src = att.get("path") or att.get("url") or att.get("filename")
        if src:
            paths.append(str(_download_if_needed(src)))

    if not paths:
        return {"damage": 0.0, "misdelivery": 0.0, "unclear": 1.0, "notes": "no_valid_paths"}

    prompt = "이 이미지(들)를 보고 파손(damage), 오배송(misdelivery), 불명확(unclear) 확률을 0~1로 JSON만 반환하라. 키: damage, misdelivery, unclear, notes"
    try:
        txt = multimodal_triage(provider=PROVIDER, model=MM_MODEL, api_key=API_KEY, image_paths=paths, prompt=prompt)
        return json.loads(txt)
    except Exception as exc:
        return {"damage": 0.0, "misdelivery": 0.0, "unclear": 1.0, "notes": f"mm_parse_error:{str(exc)[:80]}"}


@mcp.tool()
def image_match_products(image_url: str = "") -> dict[str, Any]:
    return {"candidates": [{"sku": "SKU-001", "score": 0.88, "image_url": image_url}]}


@mcp.tool()
def ocr_extract(image_url: str = "") -> dict[str, str]:
    return {"text": "운송장 1234-5678-9000", "image_url": image_url}


@mcp.tool()
def get_order(order_id: str = "") -> dict[str, Any]:
    return {"order_id": order_id, "status": "SHIPPED"}


@mcp.tool()
def get_shipping(order_id: str = "") -> dict[str, Any]:
    return {"order_id": order_id, "eta": "2026-01-03"}


@mcp.tool()
def get_claims(customer_id: str = "") -> dict[str, Any]:
    return {"customer_id": customer_id, "claim_count": 1}


@mcp.tool()
def get_customer(customer_id: str = "") -> dict[str, Any]:
    return {"customer_id": customer_id, "grade": "silver"}


@mcp.tool()
def get_inventory(sku: str = "") -> dict[str, Any]:
    return {"sku": sku, "available": 14}


@mcp.tool()
def post_reply(case_id: str, content: str, mode: str = "AUTO_POST", dry_run: bool = False) -> dict[str, Any]:
    return {"posted": not dry_run, "reply_id": "R-123", "case_id": case_id, "mode": mode, "preview": content[:120]}


@mcp.tool()
def notify_handoff(case_id: str, tenant_id: str = "", reason: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    payload = {"case_id": case_id, "tenant_id": tenant_id, "reason": reason, "tags": tags or []}
    webhook = os.getenv("HANDOFF_WEBHOOK_URL", "")
    delivered = False
    status = "stub"
    if webhook:
        try:
            resp = httpx.post(webhook, json=payload, timeout=10)
            delivered = resp.status_code < 300
            status = f"http:{resp.status_code}"
        except Exception:
            status = "http_error"
    return {"delivered": delivered, "channel": "webhook", "status": status}


app = mcp.http_app(path="/")
