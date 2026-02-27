"""Streamlit 대시보드용 DB 유틸.

읽기 전용 쿼리 실행을 담당하며, 파라미터 바인딩으로 SQL 인젝션 위험을 완화한다.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text

DSN = os.getenv("POSTGRES_DSN", "postgresql+psycopg://postgres:postgres@postgres:5432/boardbot")
engine = create_engine(DSN, future=True)


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """SQL 결과를 DataFrame으로 반환한다."""

    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})
