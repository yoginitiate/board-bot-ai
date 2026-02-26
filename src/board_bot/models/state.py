from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


DecisionType = Literal["AUTO_POST", "DRAFT", "ASK_MORE", "HANDOFF"]
ModeType = Literal["SCENARIO", "AGENT", "AUTO"]


class Span(TypedDict):
    start: int
    end: int


class Intent(TypedDict):
    type: str
    subtype: str
    confidence: float


class Risk(TypedDict):
    level: Literal["LOW", "MEDIUM", "HIGH"]
    tags: list[str]


class Issue(TypedDict):
    issue_id: str
    summary: str
    span: Span
    intent: Intent
    required_tools: list[str]
    missing_slots: list[str]
    needs_image: bool
    needs_order_ref: bool
    risk: Risk


class ClassifierOutput(TypedDict):
    primary_intent: Intent
    secondary_intents: list[Intent]
    is_multi_issue: bool
    issues: list[Issue]
    mismatch: dict[str, Any]
    risk: dict[str, Any]
    needs: dict[str, Any]
    answer_strategy: dict[str, Any]


class CaseInput(TypedDict):
    tenant_id: str
    case_id: str
    type: str
    subtype: str
    title: str
    body: str
    attachments: list[dict[str, Any]]


class AgentState(TypedDict):
    input: CaseInput
    safety: dict[str, Any]
    signals: dict[str, Any]
    policy: dict[str, Any]
    tooling: dict[str, Any]
    draft: dict[str, Any]
    decision: dict[str, Any]
    telemetry: dict[str, Any]
    classification: NotRequired[ClassifierOutput]
    plan: NotRequired[dict[str, Any]]
