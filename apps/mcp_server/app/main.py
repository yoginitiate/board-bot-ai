"""FastMCP 도구 서버."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import Collection, connections
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None

mcp = FastMCP("board-bot-mcp")

MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql+psycopg://postgres:postgres@postgres:5432/boardbot")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
MM_MODEL = os.getenv("GEMINI_MM_MODEL", "gemini-2.0-flash")

_engine = create_engine(POSTGRES_DSN, future=True)
_embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
_connections_ok = False
_bm25: dict[str, BM25Okapi] = {}
_bm25_docs: dict[str, list[dict[str, Any]]] = {}


def _ensure_milvus():
    global _connections_ok
    if _connections_ok:
        return
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    _connections_ok = True


def _embed_query(query: str) -> list[float]:
    if _embeddings:
        return _embeddings.embed_query(query)
    base = [0.0] * 8
    for i, ch in enumerate(query.encode("utf-8")):
        base[i % len(base)] += ch / 255
    return base


def _load_sparse_docs(source_type: str, tenant_id: str | None):
    key = f"{source_type}:{tenant_id or '*'}"
    if key in _bm25:
        return
    q = "SELECT doc_id, source_type, content, metadata_jsonb FROM rag_documents WHERE source_type=:st"
    params = {"st": source_type}
    if tenant_id:
        q += " AND (tenant_id=:tenant OR tenant_id IS NULL)"
        params["tenant"] = tenant_id
    with _engine.begin() as conn:
        rows = conn.execute(text(q), params).mappings().all()
    docs = [dict(r) for r in rows]
    tokenized = [d["content"].split() for d in docs] or [[""]]
    _bm25[key] = BM25Okapi(tokenized)
    _bm25_docs[key] = docs


def _sparse_search(query: str, source_type: str, top_k: int, tenant_id: str | None):
    _load_sparse_docs(source_type, tenant_id)
    key = f"{source_type}:{tenant_id or '*'}"
    bm25 = _bm25[key]
    docs = _bm25_docs[key]
    if not docs:
        return []
    scores = bm25.get_scores(query.split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "text": docs[i]["content"],
            "source_type": source_type,
            "doc_id": docs[i]["doc_id"],
            "score": float(sc),
            "metadata": {"sparse_score": float(sc), "dense_score": 0.0, "metadata": docs[i].get("metadata_jsonb") or {}},
            "rank": rank + 1,
        }
        for rank, (i, sc) in enumerate(ranked)
    ]


def _dense_search(query: str, source_type: str, top_k: int, tenant_id: str | None):
    try:
        _ensure_milvus()
        coll = Collection(f"rag_{source_type}")
        coll.load()
        expr = f'tenant_id == "{tenant_id}"' if tenant_id else None
        res = coll.search(
            data=[_embed_query(query)],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["doc_id", "text", "source_type", "tenant_id", "metadata"],
            expr=expr,
        )
        hits = []
        for rank, h in enumerate(res[0], start=1):
            ent = h.entity
            hits.append(
                {
                    "text": ent.get("text"),
                    "source_type": source_type,
                    "doc_id": ent.get("doc_id"),
                    "score": float(h.score),
                    "metadata": {"dense_score": float(h.score), "sparse_score": 0.0, "metadata": ent.get("metadata") or {}},
                    "rank": rank,
                }
            )
        return hits
    except Exception:
        return []


def _rrf_fusion(dense_hits: list[dict[str, Any]], sparse_hits: list[dict[str, Any]], final_top_k: int, rrf_k: int = 60):
    pool: dict[str, dict[str, Any]] = {}
    for hit in dense_hits:
        key = f"{hit['source_type']}::{hit['doc_id']}"
        pool.setdefault(key, hit)
        pool[key]["fusion_score"] = pool[key].get("fusion_score", 0.0) + 1.0 / (rrf_k + hit["rank"])
    for hit in sparse_hits:
        key = f"{hit['source_type']}::{hit['doc_id']}"
        if key not in pool:
            pool[key] = hit
        pool[key]["fusion_score"] = pool[key].get("fusion_score", 0.0) + 1.0 / (rrf_k + hit["rank"])
        pool[key].setdefault("metadata", {}).setdefault("sparse_score", hit.get("score", 0.0))
    merged = sorted(pool.values(), key=lambda x: x.get("fusion_score", 0.0), reverse=True)[:final_top_k]
    return merged


@mcp.tool()
def rag_search(
    query: str,
    sources: list[str] | None = None,
    top_k: int = 10,
    tenant_id: str | None = None,
    dense_top_k: int = 20,
    sparse_top_k: int = 50,
) -> dict[str, Any]:
    srcs = sources or ["policy", "manual", "script"]
    dense_hits: list[dict[str, Any]] = []
    sparse_hits: list[dict[str, Any]] = []
    for src in srcs:
        dense_hits.extend(_dense_search(query, src, dense_top_k, tenant_id))
        sparse_hits.extend(_sparse_search(query, src, sparse_top_k, tenant_id))
    fused = _rrf_fusion(dense_hits, sparse_hits, final_top_k=top_k, rrf_k=60)
    return {
        "hits": fused,
        "sources": [f"{h['source_type']}:{h['doc_id']}" for h in fused],
        "debug": {"dense_count": len(dense_hits), "sparse_count": len(sparse_hits), "fusion": "rrf"},
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
def vision_triage(attachments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    attachments = attachments or []
    if not attachments or genai is None or not GOOGLE_API_KEY:
        return {"damage": 0.0, "misdelivery": 0.0, "unclear": 1.0, "notes": "no_image_or_api_key"}

    client = genai.Client(api_key=GOOGLE_API_KEY)
    contents: list[Any] = []
    for att in attachments:
        src = att.get("path") or att.get("url") or att.get("filename")
        if not src:
            continue
        f = client.files.upload(file=str(_download_if_needed(src)))
        contents.append(f)
    contents.append(
        "이 이미지(들)를 보고 파손(damage), 오배송(misdelivery), 불명확(unclear) 확률을 0~1로 JSON만 반환하라. 키: damage, misdelivery, unclear, notes"
    )
    response = client.models.generate_content(model=MM_MODEL, contents=contents)
    text_resp = getattr(response, "text", "") or ""
    import json

    try:
        return json.loads(text_resp)
    except Exception:
        return {"damage": 0.0, "misdelivery": 0.0, "unclear": 1.0, "notes": text_resp[:200]}


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


app = mcp.http_app(path="/")
