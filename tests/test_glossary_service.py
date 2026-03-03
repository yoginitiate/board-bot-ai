from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, text

from board_bot.services.glossary import GlossaryService


def test_glossary_cache_extract_and_synonyms():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE dictionary_table (no INTEGER PRIMARY KEY, term TEXT, description TEXT)"))
        conn.execute(text("CREATE TABLE synonym_table (no INTEGER, synonym TEXT)"))
        conn.execute(text("INSERT INTO dictionary_table(no, term, description) VALUES (1, '역배송', '반품을 위해 고객에게 회수 택배를 보내는 절차')"))
        conn.execute(text("INSERT INTO synonym_table(no, synonym) VALUES (1, '회수배송')"))

    svc = GlossaryService(engine)
    svc.load_cache()

    terms = svc.extract_terms("회수배송 진행해주세요", max_terms=5)
    assert terms and terms[0]["description"].startswith("반품")

    syns = svc.get_synonyms("역배송", max_synonyms=3)
    assert "회수배송" in syns
