"""Streamlit 대시보드 루트 페이지.

운영자는 좌측 공통 필터를 설정한 뒤, 각 페이지에서 KPI를 비교/드릴다운한다.
"""

from __future__ import annotations

import streamlit as st

from streamlit_app.lib.metrics import default_range

st.set_page_config(page_title="Board Bot Dashboard", layout="wide")

start, end = default_range(7)
with st.sidebar:
    st.header("공통 필터")
    # Why: 모든 페이지에서 같은 필터 컨텍스트를 공유해 지표 해석 일관성을 유지한다.
    st.session_state["filters"] = {
        "start_ts": st.date_input("시작일", value=start.date()),
        "end_ts": st.date_input("종료일", value=end.date()),
        "tenant_id": st.text_input("tenant_id", ""),
        "type": st.text_input("type", ""),
        "subtype": st.text_input("subtype", ""),
        "decision": st.selectbox("decision", ["", "AUTO_POST", "DRAFT", "ASK_MORE", "HANDOFF"]),
        "response_mode": st.selectbox("response_mode", ["", "SCENARIO", "AGENT"]),
        "risk_level": st.selectbox("risk_level", ["", "LOW", "MEDIUM", "HIGH"]),
        "prompt_version": st.text_input("prompt_version", ""),
        "model": st.text_input("model", ""),
    }

st.title("Business Insight + AI KPI Dashboard")
st.write("운영 의사결정: 자동화율/안전성/도구 안정성/비용을 빠르게 진단하고 Case Explorer로 드릴다운하세요.")
