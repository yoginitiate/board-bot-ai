"""대시보드 집계 로더.

DB 쿼리 실행과 필터 파라미터 매핑을 담당한다.
`st.cache_data`를 사용해 동일 필터 조합(tenant/date 등) 요청을 재활용한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from streamlit_app.lib import queries
from streamlit_app.lib.db import query_df


def default_range(days: int = 7) -> tuple[datetime, datetime]:
    """기본 조회 기간을 반환한다."""

    end = datetime.utcnow()
    return end - timedelta(days=days), end


@st.cache_data(ttl=120)
def load_df(sql: str, params: dict) -> object:
    """SQL 결과를 캐싱하여 조회한다.

    Args:
        sql: 실행할 SQL 문자열.
        params: 바인딩 파라미터(tenant/date/filter 포함).

    Returns:
        pandas DataFrame.

    Why:
        대시보드의 페이지 전환/필터 반복에서 DB 부하를 줄이기 위함.
        cache key는 함수 인자(sql + params dict)에 의해 자동 구성된다.
    """

    return query_df(sql, params)


def params_from_filters(filters: dict) -> dict:
    """Sidebar 필터를 SQL 파라미터로 변환한다."""

    return {
        "start_ts": filters["start_ts"],
        "end_ts": filters["end_ts"],
        "tenant_id": filters.get("tenant_id", ""),
        "type": filters.get("type", ""),
        "subtype": filters.get("subtype", ""),
        "decision": filters.get("decision", ""),
        "response_mode": filters.get("response_mode", ""),
        "risk_level": filters.get("risk_level", ""),
        "prompt_version": filters.get("prompt_version", ""),
        "model": filters.get("model", ""),
    }


def load_overview(filters: dict):
    """대시보드 전 페이지 공용 데이터셋을 로드한다."""

    p = params_from_filters(filters)
    return {
        "kpi": load_df(queries.KPI_OVERVIEW, p),
        "decision_trend": load_df(queries.DECISION_TREND, p),
        "tool": load_df(queries.TOOL_RELIABILITY, p),
        "validator": load_df(queries.VALIDATOR_TOP, p),
        "model": load_df(queries.MODEL_PERF, p),
        "business": load_df(queries.BUSINESS_TOP, p),
    }
