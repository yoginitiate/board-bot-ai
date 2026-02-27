"""LangGraph 상태 스키마를 정의한다.

이 모듈은 게시판 Agent 파이프라인에서 노드 간 공유되는 상태 구조를
`TypedDict`로 엄격히 선언한다. 상태 필드는 입력/안전/신호/정책/도구 실행/
초안/의사결정/텔레메트리로 분리되어, 추론 흐름과 관측 가능성을 높인다.

Security:
    - 본 스키마는 원문 PII 저장을 권장하지 않는다.
    - LLM 입력/출력은 마스킹된 텍스트를 우선 사용해야 한다.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

DecisionType = Literal["AUTO_POST", "DRAFT", "ASK_MORE", "HANDOFF"]
"""최종 처리 결정 타입."""

ModeType = Literal["SCENARIO", "AGENT", "AUTO"]
"""응답 생성 모드 타입."""


class Span(TypedDict):
    """본문 내 이슈 위치 범위를 나타낸다.

    Attributes:
        start: 마스킹된 본문(`body_masked`) 기준 시작 인덱스.
        end: 마스킹된 본문 기준 종료 인덱스(포함/미포함 정책은 호출부에서 일관 적용).
    """

    start: int
    end: int


class Intent(TypedDict):
    """문의 의도 분류 결과를 표현한다.

    Attributes:
        type: 대분류 유형(예: 배송, 클레임).
        subtype: 세부 유형(예: 지연, 파손).
        confidence: 분류 신뢰도(0~1).
    """

    type: str
    subtype: str
    confidence: float


class Risk(TypedDict):
    """리스크 수준과 태그를 표현한다."""

    level: Literal["LOW", "MEDIUM", "HIGH"]
    tags: list[str]


class Issue(TypedDict):
    """복합 문의를 구성하는 단일 이슈 스키마.

    각 필드는 "어떤 도구가 필요한지", "무엇이 부족한지", "이미지/주문참조가 필요한지"를
    명시한다. Executor는 이 메타데이터를 기반으로 SECTIONED_REPLY 또는 안전 전환을 수행한다.

    Security:
        요약/슬롯 정보는 최소한으로 유지하고, 고객 식별 정보 원문은 포함하지 않는다.
    """

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
    """분류/리스크/라우팅 통합 결과 스키마."""

    primary_intent: Intent
    secondary_intents: list[Intent]
    is_multi_issue: bool
    issues: list[Issue]
    mismatch: dict[str, Any]
    risk: dict[str, Any]
    needs: dict[str, Any]
    answer_strategy: dict[str, Any]


class CaseInput(TypedDict):
    """API 입력 페이로드의 정규화 형태."""

    tenant_id: str
    case_id: str
    type: str
    subtype: str
    title: str
    body: str
    attachments: list[dict[str, Any]]


class AgentState(TypedDict):
    """LangGraph 전체 상태 컨테이너.

    Attributes:
        input: 원본 입력(가능하면 민감정보 마스킹 전 최소 보관).
        safety: 마스킹 결과/안전 플래그.
        signals: 비전 선판단 등 경량 신호.
        policy: 테넌트/유형 기반 정책 로딩 결과.
        tooling: MCP 도구 호출 trace.
        draft: 응답 초안/인용/모드.
        decision: AUTO_POST/DRAFT/ASK_MORE/HANDOFF 결과.
        telemetry: metrics_events 적재용 이벤트 버퍼 및 버전 메타.
        classification: 분류 결과(선택).
        plan: Plan DSL 실행 계획(선택).
    """

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
