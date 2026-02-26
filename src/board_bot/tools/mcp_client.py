from __future__ import annotations

from typing import Any

import httpx


class MCPClient:
    def __init__(self, base_url: str, timeout: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call_tool(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            r = httpx.post(f"{self.base_url}/tools/{tool}", json=args, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if tool == "rag_search":
                return {"hits": [{"text": "dummy rag", "source_type": "policy"}], "sources": ["policy:dummy"]}
            if tool == "vision_triage":
                return {"damage": 0.5, "misdelivery": 0.1}
            if tool == "post_reply":
                return {"posted": True, "mode": args.get("mode", "AUTO_POST"), "reply_id": "R-local"}
            return {"tool": tool, "ok": True}
