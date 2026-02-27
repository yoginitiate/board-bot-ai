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
from pymilvus import AnnSearchRequest, Collection, WeightedRanker, connections
from sqlalchemy import create_engine, text

from board_bot.llm.factory import create_embeddings, multimodal_triage
from board_bot.retrieval.bm25_sparse import make_query_sparse_vector
from board_bot.services.config import AppConfig

mcp = FastMCP("board-bot-mcp")
cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")

MILVUS_HOST = cfg.get("milvus", "host", default="milvus")
MILVUS_PORT = cfg.get("milvus", "port", default=19530)
POSTGRES_DSN = cfg.get("postgres", "dsn", default="postgresql+psycopg://postgres:postgres@postgres:5432/boardbot")
PROVIDER = cfg.get("llm", "provider", default="google")
API_KEY = cfg.get("llm", "api_key", default="")
EMBED_MODEL = cfg.get("llm", "embedding_model", default="text-embedding-004")
MM_MODEL = cfg.get("llm", "mm_model", default="gemini-2.0-flash")
COLLECTION = cfg.get("rag", "collection_name", default="rag_hybrid_v2")

_engine = create_engine(POSTGRES_DSN, future=True)
_embeddings = create_embeddings(provider=PROVIDER, model=EMBED_MODEL, api_key=API_KEY) if API_KEY else None
_connections_ok = False
_vocab_cache: tuple[dict[str, int], dict[str, float]] | None = None


def _ensure_milvus() -> None:
    global _connections_ok
    if not _connections_ok:
        connections.connect(alias="default", host=str(MILVUS_HOST), port=str(MILVUS_PORT))
        _connections_ok = True


def _embed_query(query: str) -> list[float]:
    if _embeddings:
        return _embeddings.embed_query(query)
    base = [0.0] * 8
    for i, ch in enumerate(query.encode("utf-8")):
        base[i % len(base)] += ch / 255
    return base


def _load_vocab() -> tuple[dict[str, int], dict[str, float]]:
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache
    with _engine.begin() as conn:
        rows = conn.execute(text("SELECT term, term_index, idf FROM rag_vocab ORDER BY term_index")).mappings().all()
    vocab = {r["term"]: int(r["term_index"]) for r in rows}
    idf = {r["term"]: float(r["idf"]) for r in rows}
    _vocab_cache = (vocab, idf)
    return _vocab_cache


@mcp.tool()
def rag_search(query: str, sources: list[str] | None = None, top_k: int = 10, tenant_id: str | None = None) -> dict[str, Any]:
    """Milvus native hybrid_search를 수행한다.

    구현 방식:
    - dense_vector + sparse_vector가 같은 컬렉션에 저장됨
    - AnnSearchRequest 2개 생성
    - WeightedRanker로 재랭킹
    - col.hybrid_search 호출
    """

    _ensure_milvus()
    coll = Collection(COLLECTION)
    coll.load()

    vocab, idf = _load_vocab()
    q_dense = _embed_query(query)
    q_sparse = make_query_sparse_vector(query, vocab=vocab, idf=idf)

    hcfg = cfg.get("rag", "hybrid", default={})
    per_field_limit = int(hcfg.get("search_limit_per_field", hcfg.get("dense_top_k", 20)))
    final_top_k = int(hcfg.get("final_top_k", top_k))
    sparse_weight = float(hcfg.get("sparse_weight", 0.5))
    dense_weight = float(hcfg.get("dense_weight", 0.5))

    dense_req = AnnSearchRequest(data=[q_dense], anns_field="dense_vector", param={"metric_type": "IP", "params": {}}, limit=per_field_limit)
    sparse_req = AnnSearchRequest(data=[q_sparse], anns_field="sparse_vector", param={"metric_type": "IP", "params": {}}, limit=per_field_limit)

    srcs = sources or cfg.get("rag", "sources_default", default=["policy", "manual", "script"])
    src_expr = " or ".join([f'source_type == "{s}"' for s in srcs])
    tenant_expr = f'tenant_id == "{tenant_id}"' if tenant_id else ""
    expr = f"({src_expr})" + (f" and ({tenant_expr})" if tenant_expr else "")

    res = coll.hybrid_search(
        [sparse_req, dense_req],
        rerank=WeightedRanker(sparse_weight, dense_weight),
        limit=final_top_k,
        output_fields=["text", "source_type", "doc_id", "metadata"],
        expr=expr,
    )

    hits = []
    for h in res[0]:
        ent = h.entity
        hits.append(
            {
                "text": ent.get("text"),
                "source_type": ent.get("source_type"),
                "doc_id": ent.get("doc_id"),
                "score": float(h.score),
                "metadata": {"hybrid_score": float(h.score), **(ent.get("metadata") or {})},
            }
        )

    return {
        "hits": hits,
        "sources": [f"{h['source_type']}:{h['doc_id']}" for h in hits],
        "debug": {
            "path": "milvus_native_hybrid_search",
            "collection": COLLECTION,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "per_field_limit": per_field_limit,
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
