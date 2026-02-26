from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_prompt(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = Path(path).read_bytes()
    data["version_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    return data
