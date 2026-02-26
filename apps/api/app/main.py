from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from board_bot.graphs.pipeline import compile_graph
from board_bot.services.config import AppConfig
from board_bot.services.db import PostgresRepo
from board_bot.tools.mcp_client import MCPClient

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None


class CasePayload(BaseModel):
    type: str
    subtype: str
    title: str
    body: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ProcessRequest(BaseModel):
    tenant_id: str
    case_id: str | None = None
    payload: CasePayload


cfg = AppConfig("config/app.yaml")
repo = PostgresRepo(cfg.get("postgres", "dsn", default="postgresql+psycopg://postgres:postgres@localhost:5432/boardbot"))
mcp = MCPClient(cfg.get("mcp", "url", default="http://mcp:9000"))
policies = {"배송::지연": {"mode": "SCENARIO", "allow_auto_post": True}, "클레임::파손": {"mode": "AGENT", "allow_auto_post": False}}

llm = None
if ChatGoogleGenerativeAI and cfg.get("llm", "api_key"):
    llm = ChatGoogleGenerativeAI(model=cfg.get("llm", "model", default="gemini-1.5-flash"), google_api_key=cfg.get("llm", "api_key"), temperature=0)


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return mcp.call_tool(name, args)


def logger(node: str, state: dict[str, Any]) -> None:
    case_id = state["input"]["case_id"]
    repo.log_event(case_id, node, "state", {"decision": state.get("decision"), "prompt_version": "v1", "config_version": cfg.version})
    repo.metric(case_id, "decision", 1.0, {"decision": state.get("decision", {}).get("type", "UNKNOWN")})
    repo.upsert_case(case_id, case_id, state)


graph = compile_graph(llm=llm, tool_call=tool_call, policy_map=policies, logger=logger)
app = FastAPI(title="Board Bot API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/cases/process")
def process_case(req: ProcessRequest) -> dict[str, Any]:
    case_id = req.case_id or str(uuid.uuid4())
    state = {
        "input": {
            "tenant_id": req.tenant_id,
            "case_id": case_id,
            "type": req.payload.type,
            "subtype": req.payload.subtype,
            "title": req.payload.title,
            "body": req.payload.body,
            "attachments": req.payload.attachments,
        },
        "safety": {},
        "signals": {},
        "policy": {},
        "tooling": {},
        "draft": {},
        "decision": {},
        "telemetry": {},
    }
    out = graph.invoke(state)
    return {"case_id": case_id, "decision": out.get("decision"), "draft": out.get("draft"), "tool_trace": out.get("tooling", {}).get("trace", [])}


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    item = repo.get_case(case_id)
    if not item:
        return {"error": "not_found"}
    return item
