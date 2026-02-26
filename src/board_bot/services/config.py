from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class AppConfig:
    def __init__(self, path: str = "config/app.yaml") -> None:
        self.path = Path(path)
        self.data = self._load_yaml(self.path)
        self.version = str(self.data.get("version", "v1"))
        self._apply_env_overrides()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _apply_env_overrides(self) -> None:
        if key := os.getenv("GEMINI_API_KEY"):
            self.data.setdefault("llm", {})["api_key"] = key
        if dsn := os.getenv("POSTGRES_DSN"):
            self.data.setdefault("postgres", {})["dsn"] = dsn

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur
