"""LLM/Embedding/MM 팩토리.

Provider(Google/OpenAI)를 설정으로 전환할 수 있도록 생성 책임을 분리한다.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel


def create_chat_llm(provider: str, model: str, api_key: str | None):
    """Provider에 맞는 Chat LLM 인스턴스를 생성한다."""

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


def create_embeddings(provider: str, model: str, api_key: str | None):
    """Provider에 맞는 임베딩 클라이언트를 생성한다."""

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=api_key)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def _image_to_data_url(path: str) -> str:
    p = Path(path)
    mime = "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def multimodal_triage(provider: str, model: str, api_key: str, image_paths: list[str], prompt: str) -> str:
    """이미지+텍스트 멀티모달 호출을 provider별로 수행한다.

    Returns:
        모델 원문 응답 텍스트(JSON 문자열 기대)
    """

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}])
        return resp.choices[0].message.content or ""

    from google import genai

    client = genai.Client(api_key=api_key)
    contents: list[Any] = []
    for p in image_paths:
        contents.append(client.files.upload(file=p))
    contents.append(prompt)
    resp = client.models.generate_content(model=model, contents=contents)
    return getattr(resp, "text", "") or ""
