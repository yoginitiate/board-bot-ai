"""FastMCP 도구 서버.

MVP에서는 더미/샘플 응답을 반환하며, 인터페이스 계약(입출력 형태) 검증 용도로 사용한다.
Security 원칙상 고객 식별/민감 정보는 최소 필드만 반환해야 한다.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("board-bot-mcp")


@mcp.tool()
def rag_search(query: str) -> dict[str, Any]:
    """RAG 검색 결과를 반환한다.

    Args:
        query: 마스킹된 질의 텍스트.
    """

    return {
        "hits": [
            {"text": "배송 지연 시 1영업일 내 안내", "source_type": "policy", "source_id": "P-001"},
            {"text": "지연 안내 템플릿", "source_type": "script", "source_id": "S-101"},
        ],
        "sources": ["policy:P-001", "script:S-101"],
    }


@mcp.tool()
def vision_triage(attachments: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """이미지 선판단 신호를 반환한다."""

    return {"damage": 0.71, "misdelivery": 0.15, "unclear": 0.2}


@mcp.tool()
def image_match_products(image_url: str = "") -> dict[str, Any]:
    """이미지 기반 SKU 후보를 반환한다."""

    return {"candidates": [{"sku": "SKU-001", "score": 0.88, "image_url": image_url}]}


@mcp.tool()
def ocr_extract(image_url: str = "") -> dict[str, str]:
    """OCR 추출 결과를 반환한다."""

    return {"text": "운송장 1234-5678-9000", "image_url": image_url}


@mcp.tool()
def get_order(order_id: str = "") -> dict[str, Any]:
    """주문 요약 정보를 조회한다(민감 필드 제외)."""

    return {"order_id": order_id, "status": "SHIPPED"}


@mcp.tool()
def get_shipping(order_id: str = "") -> dict[str, Any]:
    """배송 요약 정보를 조회한다."""

    return {"order_id": order_id, "eta": "2026-01-03"}


@mcp.tool()
def get_claims(customer_id: str = "") -> dict[str, Any]:
    """클레임 요약 정보를 조회한다."""

    return {"customer_id": customer_id, "claim_count": 1}


@mcp.tool()
def get_customer(customer_id: str = "") -> dict[str, Any]:
    """고객 요약 정보를 조회한다(최소 필드)."""

    return {"customer_id": customer_id, "grade": "silver"}


@mcp.tool()
def get_inventory(sku: str = "") -> dict[str, Any]:
    """재고 요약 정보를 조회한다."""

    return {"sku": sku, "available": 14}


@mcp.tool()
def post_reply(case_id: str, content: str, mode: str = "AUTO_POST", dry_run: bool = False) -> dict[str, Any]:
    """게시판 답변/추가질문 등록을 모사한다."""

    return {"posted": not dry_run, "reply_id": "R-123", "case_id": case_id, "mode": mode, "preview": content[:120]}


app = mcp.http_app(path="/")
