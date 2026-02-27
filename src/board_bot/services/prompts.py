"""프롬프트 로더.

YAML 프롬프트를 읽고 버전 추적을 위해 내용 해시를 계산한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_prompt(path: str) -> dict[str, Any]:
    """프롬프트 YAML을 로드하고 `version_hash`를 부여한다."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = Path(path).read_bytes()
    data["version_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    return data
