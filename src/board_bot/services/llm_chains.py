"""LLM Runnable chain 유틸.

분류/정책검증/에이전트 생성에서 PydanticOutputParser 기반의 안전한 구조화 출력을 제공한다.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class ClassificationSchema(BaseModel):
    primary_intent: dict[str, Any]
    secondary_intents: list[dict[str, Any]] = Field(default_factory=list)
    is_multi_issue: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
    mismatch: dict[str, Any] = Field(default_factory=lambda: {"is_mismatch": False, "reason": ""})
    risk: dict[str, Any] = Field(default_factory=lambda: {"level": "LOW", "tags": [], "handoff_required": False, "handoff_reason": ""})
    needs: dict[str, Any] = Field(default_factory=lambda: {"prevision": False})
    answer_strategy: dict[str, Any] = Field(default_factory=lambda: {"composition_hint": "SECTIONED_REPLY", "mode_hint": "SCENARIO"})


class PolicyValidationSchema(BaseModel):
    passed: bool = True
    fail_reason_code: str = ""
    revised_text: str = ""


class AgentDraftSchema(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)


def invoke_with_retry(llm, prompt: ChatPromptTemplate, parser: PydanticOutputParser, values: dict[str, Any]):
    """파싱 실패 시 한 번 더 복구 프롬프트로 재시도한다."""

    chain = prompt | llm | parser
    try:
        return chain.invoke(values)
    except Exception:
        retry_values = dict(values)
        retry_values["retry_instruction"] = "출력은 JSON만 반환하고 스키마를 정확히 준수하세요."
        return chain.invoke(retry_values)


def classification_chain(llm, masked_text: str):
    parser = PydanticOutputParser(pydantic_object=ClassificationSchema)
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 문의 분류기다. 아래 형식으로만 출력한다.
        {format_instructions}
        {retry_instruction}
        입력: {masked_text}
        """
    )
    out = invoke_with_retry(
        llm,
        prompt,
        parser,
        {
            "masked_text": masked_text,
            "format_instructions": parser.get_format_instructions(),
            "retry_instruction": "",
        },
    )
    return out.model_dump()


def policy_validation_chain(llm, text: str):
    parser = PydanticOutputParser(pydantic_object=PolicyValidationSchema)
    prompt = ChatPromptTemplate.from_template(
        """
        정책 위반 여부를 판단하고 수정안을 제시해라.
        {format_instructions}
        {retry_instruction}
        텍스트: {text}
        """
    )
    out = invoke_with_retry(
        llm,
        prompt,
        parser,
        {
            "text": text,
            "format_instructions": parser.get_format_instructions(),
            "retry_instruction": "",
        },
    )
    return out.model_dump()


def agent_generate_chain(llm, question: str, tool_summary: str):
    parser = PydanticOutputParser(pydantic_object=AgentDraftSchema)
    prompt = ChatPromptTemplate.from_template(
        """
        도구 결과를 반영해 답변을 작성하라.
        {format_instructions}
        {retry_instruction}
        문의: {question}
        도구요약: {tool_summary}
        """
    )
    out = invoke_with_retry(
        llm,
        prompt,
        parser,
        {
            "question": question,
            "tool_summary": tool_summary,
            "format_instructions": parser.get_format_instructions(),
            "retry_instruction": "",
        },
    )
    return out.model_dump()
