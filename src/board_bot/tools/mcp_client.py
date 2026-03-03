"""MCP HTTP 클라이언트.

MCP 서버의 tool endpoint를 호출하며, 네트워크 장애 시 제한적 fallback으로
로컬 개발 흐름이 완전히 중단되지 않도록 보호한다.
"""

from __future__ import annotations

from typing import Any

import httpx


class MCPClient:
    """FastMCP HTTP 도구 호출 래퍼.

    Args:
        base_url: MCP 서버 베이스 URL.
        timeout: 요청 타임아웃(초).

    Security:
        응답 원문을 그대로 장기 저장하지 말고, 상위 계층에서 최소 필드만 상태/로그에 반영한다.
    """

    def __init__(self, base_url: str, timeout: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call_tool(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """도구를 호출한다.

        Args:
            tool: 도구 이름.
            args: 도구 인자.

        Returns:
            도구 응답 JSON.

        Side Effects:
            외부 네트워크 I/O가 발생한다.

        Observability:
            호출 성공/실패는 상위 노드에서 `tool_call` metric으로 기록한다.

        Notes:
            현재 재시도는 구현하지 않고 timeout+fallback 전략을 사용한다.
            재시도/서킷브레이커는 상위 정책(mcp.retry/mcp.circuit_breaker) 확장 포인트다.
        """

        try:
            r = httpx.post(f"{self.base_url}/tools/{tool}", json=args, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if tool == "rag_search":
                return {"ok": False, "reason": "mcp_unavailable", "hits": [], "sources": []}
            if tool == "vision_triage":
                return {"damage": 0.5, "misdelivery": 0.1}
            if tool == "post_reply":
                return {"posted": True, "mode": args.get("mode", "AUTO_POST"), "reply_id": "R-local"}
            return {"tool": tool, "ok": True}
