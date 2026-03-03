"""Milvus native hybrid_search sanity 검증 스크립트."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.mcp_server.app.main import rag_search  # noqa: E402


def main():
    query = "환불 정책"
    out = rag_search(query=query, sources=["policy"], top_k=5, tenant_id="tenant-1")
    print("debug:", out.get("debug"))
    for h in out.get("hits", []):
        print(h.get("doc_id"), h.get("score"), h.get("source_type"))


if __name__ == "__main__":
    main()
