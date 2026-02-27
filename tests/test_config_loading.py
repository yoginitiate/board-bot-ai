import pytest

pytest.importorskip("yaml")

from pathlib import Path

from board_bot.services.config import AppConfig


def test_system_policy_merge_and_env_override(tmp_path: Path, monkeypatch):
    system = tmp_path / "system.yaml"
    policy = tmp_path / "policy.yaml"
    compat = tmp_path / "app.yaml"

    system.write_text(
        """
version: v1
postgres:
  dsn: postgresql://from-system
mcp:
  url: http://mcp-system
""",
        encoding="utf-8",
    )
    policy.write_text(
        """
llm:
  text_model: gemini-1.5-flash
  api_key_env: GOOGLE_API_KEY
validation:
  grounding_threshold: 0.8
  rewrite_max_attempts: 3
""",
        encoding="utf-8",
    )
    compat.write_text("version: v1\n", encoding="utf-8")

    monkeypatch.setenv("POSTGRES_DSN", "postgresql://from-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")

    cfg = AppConfig(system_path=str(system), policy_path=str(policy), compat_path=str(compat))

    assert cfg.get("postgres", "dsn") == "postgresql://from-env"
    assert cfg.get("llm", "api_key") == "env-key"
    assert cfg.get("grounding", "threshold") == 0.8
    assert cfg.get("grounding", "max_rewrite") == 3
