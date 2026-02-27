"""애플리케이션 설정 로더.

YAML 설정을 읽고 일부 민감 값은 환경변수로 override한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class AppConfig:
    """`config/app.yaml` 기반 설정 객체."""

    def __init__(self, path: str = "config/app.yaml") -> None:
        """설정을 초기화한다."""

        self.path = Path(path)
        self.data = self._load_yaml(self.path)
        self.version = str(self.data.get("version", "v1"))
        self._apply_env_overrides()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """YAML 파일을 로드한다."""

        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _apply_env_overrides(self) -> None:
        """환경변수 override를 적용한다."""

        if key := os.getenv("GEMINI_API_KEY"):
            self.data.setdefault("llm", {})["api_key"] = key
        if dsn := os.getenv("POSTGRES_DSN"):
            self.data.setdefault("postgres", {})["dsn"] = dsn

    def get(self, *keys: str, default: Any = None) -> Any:
        """중첩 키를 안전하게 조회한다."""

        cur: Any = self.data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur
