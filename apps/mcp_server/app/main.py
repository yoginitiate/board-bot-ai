from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("board-bot-mcp")


@mcp.tool()
def rag_search(query: str) -> dict[str, Any]:
    return {
        "hits": [
            {"text": "배송 지연 시 1영업일 내 안내", "source_type": "policy", "source_id": "P-001"},
            {"text": "지연 안내 템플릿", "source_type": "script", "source_id": "S-101"},
        ],
        "sources": ["policy:P-001", "script:S-101"],
    }


@mcp.tool()
def vision_triage(attachments: list[dict[str, Any]] | None = None) -> dict[str, float]:
    return {"damage": 0.71, "misdelivery": 0.15, "unclear": 0.2}


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
