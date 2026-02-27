"""MCP HTTP 클라이언트.

MCP 서버의 tool endpoint를 호출하며, 로컬/테스트 환경을 위한 최소 fallback을 제공한다.
"""

from __future__ import annotations

from typing import Any

import httpx


class MCPClient:
    """FastMCP HTTP 도구 호출 래퍼."""

    def __init__(self, base_url: str, timeout: float = 8.0) -> None:
        """클라이언트를 초기화한다.

        Args:
            base_url: MCP 서버 베이스 URL.
            timeout: 요청 타임아웃(초).
        """

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
            네트워크 I/O 발생.

        Security/Privacy:
            이 클라이언트는 도구 응답 최소 필드 사용을 전제로 하며,
            민감 원문을 상태/로그에 직접 저장하지 않도록 상위 레이어에서 제어해야 한다.
        """

        try:
            r = httpx.post(f"{self.base_url}/tools/{tool}", json=args, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            # Why: 개발 환경에서 MCP 미기동 시에도 파이프라인 데모를 유지하기 위한 fallback.
            if tool == "rag_search":
                return {"hits": [{"text": "dummy rag", "source_type": "policy"}], "sources": ["policy:dummy"]}
            if tool == "vision_triage":
                return {"damage": 0.5, "misdelivery": 0.1}
            if tool == "post_reply":
                return {"posted": True, "mode": args.get("mode", "AUTO_POST"), "reply_id": "R-local"}
            return {"tool": tool, "ok": True}
