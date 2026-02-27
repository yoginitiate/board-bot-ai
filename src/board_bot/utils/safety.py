"""안전 유틸리티(PII 마스킹/리스크 규칙 점수).

Security/Privacy:
    - 원문 PII를 외부 호출/로그로 전파하지 않도록 마스킹을 우선 적용한다.
"""

from __future__ import annotations

import re

PII_PATTERNS = [
    (re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{6}-\d{7}\b"), "[RRN]"),
    (re.compile(r"[\w.-]+@[\w.-]+\.\w+"), "[EMAIL]"),
]
RISK_KEYWORDS = {
    "HIGH": ["고소", "법적", "손해배상", "언론제보", "환불소송", "욕설", "미친"],
    "MEDIUM": ["불만", "항의", "환불"],
}


def mask_pii(text: str) -> str:
    """문자열에서 기본 PII 패턴을 마스킹한다.

    Args:
        text: 원본 문자열.

    Returns:
        마스킹된 문자열.
    """

    out = text
    for pattern, repl in PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def risk_score_rule(text: str) -> dict:
    """키워드 규칙 기반 리스크 점수를 산출한다.

    Returns:
        `{"level": LOW|MEDIUM|HIGH, "tags": [...]}` 형태 딕셔너리.
    """

    t = text.lower()
    high_tags = [kw for kw in RISK_KEYWORDS["HIGH"] if kw in t]
    if high_tags:
        return {"level": "HIGH", "tags": high_tags}
    med_tags = [kw for kw in RISK_KEYWORDS["MEDIUM"] if kw in t]
    if med_tags:
        return {"level": "MEDIUM", "tags": med_tags}
    return {"level": "LOW", "tags": []}
