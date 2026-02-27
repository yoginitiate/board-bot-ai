"""LLM Runnable chain 유틸.

분류/정책검증/슬롯추출/에이전트생성에서 PydanticOutputParser 기반 구조화 출력을 제공한다.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

ALLOWED_INTENTS = ["상품 누락", "상품 오배송", "상품 미배송", "품절", "상품 파손", "주문 취소"]


class RiskSchema(BaseModel):
    level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    tags: list[str] = Field(default_factory=list)
    handoff_required: bool = False
    handoff_reason: str = ""


class IntentItem(BaseModel):
    label: Literal["상품 누락", "상품 오배송", "상품 미배송", "품절", "상품 파손", "주문 취소"]
    confidence: float = 0.0


class ClassificationSchema(BaseModel):
    intents: list[IntentItem] = Field(default_factory=list, max_length=3)
    is_multi_issue: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
    risk: RiskSchema = Field(default_factory=RiskSchema)


class PolicyValidationSchema(BaseModel):
    passed: bool = True
    fail_reason_code: str = ""
    revised_text: str = ""


class AgentDraftSchema(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)


class SlotSchema(BaseModel):
    product_name: str = ""
    quantity: str = ""
    date: str = ""
    order_status: str = ""
    shipping_status: str = ""


def invoke_with_retry(llm, prompt: ChatPromptTemplate, parser: PydanticOutputParser, values: dict[str, Any]):
    chain = prompt | llm | parser
    try:
        return chain.invoke(values)
    except Exception:
        retry_values = dict(values)
        retry_values["retry_instruction"] = "출력은 JSON만 반환하고 스키마를 정확히 준수하세요."
        return chain.invoke(retry_values)


def classification_chain(llm, masked_text: str, examples_context: str):
    parser = PydanticOutputParser(pydantic_object=ClassificationSchema)
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 문의 분류기다. intent는 반드시 아래 6개 중에서만 선택한다:
        {allowed_intents}
        고위험 표현(욕설/법적위협/보상요구)은 intent로 넣지 말고 risk 필드에만 넣어라.
        최대 intent 3개까지만 추출한다.
        참고 예문:
        {examples_context}
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
            "examples_context": examples_context,
            "allowed_intents": ", ".join(ALLOWED_INTENTS),
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
    out = invoke_with_retry(llm, prompt, parser, {"text": text, "format_instructions": parser.get_format_instructions(), "retry_instruction": ""})
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
        {"question": question, "tool_summary": tool_summary, "format_instructions": parser.get_format_instructions(), "retry_instruction": ""},
    )
    return out.model_dump()


def slot_extraction_chain(llm, masked_text: str):
    parser = PydanticOutputParser(pydantic_object=SlotSchema)
    prompt = ChatPromptTemplate.from_template(
        """
        문의에서 슬롯을 추출한다. 없으면 빈 문자열을 넣는다.
        {format_instructions}
        {retry_instruction}
        입력: {masked_text}
        """
    )
    out = invoke_with_retry(llm, prompt, parser, {"masked_text": masked_text, "format_instructions": parser.get_format_instructions(), "retry_instruction": ""})
    return out.model_dump()
