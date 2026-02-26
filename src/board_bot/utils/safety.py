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
    out = text
    for pattern, repl in PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def risk_score_rule(text: str) -> dict:
    t = text.lower()
    high_tags = [kw for kw in RISK_KEYWORDS["HIGH"] if kw in t]
    if high_tags:
        return {"level": "HIGH", "tags": high_tags}
    med_tags = [kw for kw in RISK_KEYWORDS["MEDIUM"] if kw in t]
    if med_tags:
        return {"level": "MEDIUM", "tags": med_tags}
    return {"level": "LOW", "tags": []}
