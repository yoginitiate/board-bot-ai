"""게시판 Agent FastAPI 엔트리포인트.

이 모듈은 요청을 LangGraph 파이프라인에 전달하고, 실행 상태/메트릭을 Postgres에 기록한다.
"""

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
    """케이스 본문 입력 DTO."""

    type: str
    subtype: str
    title: str
    body: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ProcessRequest(BaseModel):
    """케이스 처리 요청 DTO."""

    tenant_id: str
    case_id: str | None = None
    payload: CasePayload


cfg = AppConfig("config/app.yaml")
repo = PostgresRepo(cfg.get("postgres", "dsn", default="postgresql+psycopg://postgres:postgres@localhost:5432/boardbot"))
mcp = MCPClient(cfg.get("mcp", "url", default="http://mcp:9000"))
policies = {"배송::지연": {"mode": "SCENARIO", "allow_auto_post": True}, "클레임::파손": {"mode": "AGENT", "allow_auto_post": False}}

llm = None
model_name = cfg.get("llm", "model", default="gemini-1.5-flash")
api_key = cfg.get("llm", "api_key") or __import__("os").getenv("GOOGLE_API_KEY") or __import__("os").getenv("GEMINI_API_KEY")
if ChatGoogleGenerativeAI and api_key:
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """MCP 도구 호출 위임 함수."""

    return mcp.call_tool(name, args)


def logger(node: str, state: dict[str, Any]) -> None:
    """그래프 실행 결과를 DB에 영속화한다.

    Side Effects:
        - execution_logs INSERT
        - metrics_events batch INSERT
        - cases upsert
    """

    case_id = state["input"]["case_id"]
    state.setdefault("telemetry", {})["config_version"] = cfg.version
    state.setdefault("telemetry", {})["prompt_version"] = "v1"
    state.setdefault("telemetry", {})["model"] = model_name
    repo.log_event(case_id, node, "state", {"decision": state.get("decision"), "prompt_version": "v1", "config_version": cfg.version})
    repo.insert_metrics(state.get("telemetry", {}).get("metrics", []))
    repo.upsert_case(case_id, case_id, state)


graph = compile_graph(llm=llm, tool_call=tool_call, policy_map=policies, logger=logger, repo=repo)
app = FastAPI(title="Board Bot API")


@app.get("/health")
def health() -> dict[str, str]:
    """헬스체크 엔드포인트."""

    return {"status": "ok"}


@app.post("/v1/cases/process")
def process_case(req: ProcessRequest) -> dict[str, Any]:
    """게시판 케이스를 처리한다.

    Security/Privacy:
        원문 PII는 노드 내부에서 마스킹되며, 메트릭에는 원문 저장 금지.
    """

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
        "telemetry": {"prompt_version": "v1", "config_version": cfg.version, "model": model_name, "metrics": []},
        "config": cfg.data,
    }
    out = graph.invoke(state)
    return {"case_id": case_id, "decision": out.get("decision"), "draft": out.get("draft"), "tool_trace": out.get("tooling", {}).get("trace", [])}


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    """저장된 케이스 상태를 조회한다."""

    item = repo.get_case(case_id)
    if not item:
        return {"error": "not_found"}
    return item
