"""용어사전(전문 용어) 캐시 서비스.

요구사항 반영:
- DB 테이블(board_bot.t_term_dictionary, board_bot.t_synonym_list)에서 용어/동의어/설명을 로드
- 정규화 인덱스로 질문에서 용어 추출
- 동일 description(또는 no)을 공유하는 항목을 동의어로 확장
- TTL 기반 캐시 리프레시(옵션)
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

_NORMALIZE_RE = re.compile(r"\s+")


@dataclass
class GlossaryTerm:
    term: str
    description: str


class GlossaryService:
    """DB 기반 용어사전 캐시 서비스."""

    def __init__(self, engine, refresh_enabled: bool = False, ttl_seconds: int = 600) -> None:
        self.engine = engine
        self.refresh_enabled = refresh_enabled
        self.ttl_seconds = ttl_seconds
        self._term_to_desc: dict[str, str] = {}
        self._normalized_to_term: dict[str, str] = {}
        self._group_to_terms: dict[str, list[str]] = defaultdict(list)
        self._loaded_at = 0.0

    @staticmethod
    def normalize_text(text: str) -> str:
        return _NORMALIZE_RE.sub("", (text or "").lower())

    def load_cache(self) -> None:
        """DB에서 용어/동의어/설명을 읽어 캐시를 구축한다."""

        with self.engine.begin() as conn:
            try:
                rows = conn.execute(
                    text(
                        """
                        SELECT d.dictionary_no, d.term, d.description, s.synonym
                        FROM board_bot.t_term_dictionary d
                        LEFT JOIN board_bot.t_synonym_list s ON s.dictionary_no = d.dictionary_no
                        ORDER BY d.dictionary_no
                        """
                    )
                ).mappings().all()
            except Exception:
                # 테스트/임시 환경(SQLite) 호환 fallback
                rows = conn.execute(
                    text(
                        """
                        SELECT d.no AS dictionary_no, d.term, d.description, s.synonym
                        FROM dictionary_table d
                        LEFT JOIN synonym_table s ON s.no = d.no
                        ORDER BY d.no
                        """
                    )
                ).mappings().all()

        term_to_desc: dict[str, str] = {}
        normalized_to_term: dict[str, str] = {}
        group_to_terms: dict[str, list[str]] = defaultdict(list)

        for row in rows:
            base_term = (row.get("term") or "").strip()
            synonym = (row.get("synonym") or "").strip()
            desc = (row.get("description") or "").strip()
            no = str(row.get("dictionary_no"))
            group_key = f"{no}:{desc}"

            if base_term:
                term_to_desc.setdefault(base_term, desc)
                normalized_to_term.setdefault(self.normalize_text(base_term), base_term)
                if base_term not in group_to_terms[group_key]:
                    group_to_terms[group_key].append(base_term)

            if synonym:
                term_to_desc.setdefault(synonym, desc)
                normalized_to_term.setdefault(self.normalize_text(synonym), synonym)
                if synonym not in group_to_terms[group_key]:
                    group_to_terms[group_key].append(synonym)

        self._term_to_desc = term_to_desc
        self._normalized_to_term = normalized_to_term
        self._group_to_terms = group_to_terms
        self._loaded_at = time.time()

    def _refresh_if_needed(self) -> None:
        if not self._term_to_desc:
            self.load_cache()
            return
        if self.refresh_enabled and (time.time() - self._loaded_at) >= self.ttl_seconds:
            self.load_cache()

    def extract_terms(self, question: str, max_terms: int = 8, description_max_chars: int = 120) -> list[dict[str, str]]:
        """질문에서 용어를 추출한다(정규화 포함 매칭)."""

        self._refresh_if_needed()
        normalized_q = self.normalize_text(question)
        found: list[dict[str, str]] = []

        candidates = sorted(self._normalized_to_term.items(), key=lambda x: len(x[0]), reverse=True)
        for norm_term, canonical in candidates:
            if not norm_term:
                continue
            if norm_term in normalized_q:
                desc = self._term_to_desc.get(canonical, "")[:description_max_chars]
                found.append({"term": canonical, "description": desc})
                if len(found) >= max_terms:
                    break
        return found

    def get_synonyms(self, term: str, max_synonyms: int = 3) -> list[str]:
        """동일 설명 그룹의 동의어를 반환한다."""

        self._refresh_if_needed()
        desc = self._term_to_desc.get(term, "")
        if not desc:
            return []

        out: list[str] = []
        for _, terms in self._group_to_terms.items():
            if term in terms:
                out = [t for t in terms if t != term]
                break
        return out[:max_synonyms]

    def known_terms(self) -> list[str]:
        self._refresh_if_needed()
        return list(self._term_to_desc.keys())
