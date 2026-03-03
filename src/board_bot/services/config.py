"""설정 로더.

`system.yaml`(인프라/런타임)과 `policy.yaml`(서비스 정책)을 병합해
애플리케이션 전역에서 단일 Config 객체로 사용하도록 제공한다.

우선순위:
    env override > yaml 값 > 코드 default
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class AppConfig:
    """시스템/정책 설정을 병합 제공하는 구성 객체.

    Args:
        system_path: 시스템 운영 설정 파일 경로.
        policy_path: 서비스 정책 설정 파일 경로.
        compat_path: 하위호환 app.yaml 경로(선택).

    Side Effects:
        파일 I/O를 수행해 YAML을 로드한다.

    Security:
        API Key/DSN은 환경변수 override를 우선 적용해야 한다.
    """

    def __init__(self, system_path: str = "config/system.yaml", policy_path: str = "config/policy.yaml", compat_path: str = "config/app.yaml") -> None:
        self.system_path = Path(system_path)
        self.policy_path = Path(policy_path)
        self.compat_path = Path(compat_path)

        self.system = self._load_yaml(self.system_path)
        self.policy = self._load_yaml(self.policy_path)
        if "policy" in self.policy and isinstance(self.policy.get("policy"), dict):
            # 정책 파일이 policy: 루트 구조를 사용할 때 내부 딕셔너리를 실제 정책으로 사용
            version = self.policy.get("version", "v1")
            self.policy = {"version": version, **self.policy["policy"]}

        # 하위호환: 구형 app.yaml이 있는 경우 파일 위치 힌트를 읽어 대체 로딩 시도
        compat = self._load_yaml(self.compat_path)
        if not self.system and compat.get("system_config_path"):
            self.system = self._load_yaml(Path(compat["system_config_path"]))
        if not self.policy and compat.get("policy_config_path"):
            self.policy = self._load_yaml(Path(compat["policy_config_path"]))

        self.data = self._merge_dicts(deepcopy(self.system), deepcopy(self.policy))
        self.version = str(self.data.get("version", self.system.get("version", "v1")))
        self._build_aliases()
        self._apply_env_overrides()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = deepcopy(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = AppConfig._merge_dicts(out[k], v)
            else:
                out[k] = deepcopy(v)
        return out

    def _build_aliases(self) -> None:
        """기존 참조 키 호환 alias를 구성한다.

        Why:
            코드 전체를 대규모 수정하지 않고 설정 분리를 점진 도입하기 위함.
        """

        llm = self.data.get("llm", {})
        # provider 기본값은 google
        llm.setdefault("provider", "google")
        # 구형: llm.model / llm.api_key
        llm.setdefault("model", self.data.get("llm", {}).get("text_model", "gemini-1.5-flash"))
        llm.setdefault("api_key", "")
        self.data["llm"] = llm
        if "gemini" in self.data:
            self.data["llm"].setdefault("text_model", self.data["gemini"].get("text_model", "gemini-1.5-flash"))
            self.data["llm"].setdefault("mm_model", self.data["gemini"].get("mm_model", "gemini-2.0-flash"))
            self.data["llm"].setdefault("embedding_model", self.data["gemini"].get("embedding_model", "models/embedding-001"))

        # 구형 키를 새 정책 키와 동기화
        if "validation" in self.data:
            self.data.setdefault("grounding", {})["threshold"] = self.data["validation"].get("grounding_threshold", 0.7)
            self.data.setdefault("grounding", {})["max_rewrite"] = self.data["validation"].get("rewrite_max_attempts", 2)

    def _apply_env_overrides(self) -> None:
        """환경변수 override를 적용한다.

        우선순위:
            1) 명시 env(POSTGRES_DSN, GOOGLE_API_KEY ...)
            2) YAML
            3) 코드 default
        """

        if dsn := os.getenv("POSTGRES_DSN"):
            self.data.setdefault("postgres", {})["dsn"] = dsn

        api_env = self.data.get("llm", {}).get("api_key_env", "GOOGLE_API_KEY")
        if key := os.getenv(api_env) or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"):
            self.data.setdefault("llm", {})["api_key"] = key

        if mcp_url := os.getenv("MCP_URL"):
            self.data.setdefault("mcp", {})["url"] = mcp_url

        if m_host := os.getenv("MILVUS_HOST"):
            self.data.setdefault("milvus", {})["host"] = m_host
        if m_port := os.getenv("MILVUS_PORT"):
            try:
                self.data.setdefault("milvus", {})["port"] = int(m_port)
            except ValueError:
                self.data.setdefault("milvus", {})["port"] = m_port
        if m_col := os.getenv("MILVUS_COLLECTION"):
            self.data.setdefault("milvus", {})["collection_name"] = m_col


    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur
