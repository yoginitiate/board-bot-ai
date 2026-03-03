"""Executive Overview 페이지.

운영자는 전체 처리량, 자동화율, 지연시간, 비용을 한 화면에서 확인해
당일 운영 상태를 빠르게 판단한다.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.lib.metrics import load_overview

filters = st.session_state.get("filters", {})
data = load_overview(filters)
kpi = data["kpi"].iloc[0] if not data["kpi"].empty else {}

st.title("Executive Overview")
cols = st.columns(8)
labels = [
    ("Total Volume", "total_cases"),
    ("AUTO_POST Rate", "auto_post_rate"),
    ("DRAFT Rate", "draft_rate"),
    ("ASK_MORE Rate", "ask_more_rate"),
    ("HANDOFF Rate", "handoff_rate"),
    ("Avg latency", "avg_latency"),
    ("P95 latency", "p95_latency"),
    ("Cost/case", "cost_per_case"),
]
for c, (name, key) in zip(cols, labels):
    c.metric(name, f"{kpi.get(key, 0) or 0:.3f}" if key != "total_cases" else int(kpi.get(key, 0) or 0))

# 드릴다운 흐름: 이 추세에서 이상 구간을 찾고 Tool/Case Explorer 페이지로 이동한다.
if not data["decision_trend"].empty:
    fig = px.area(data["decision_trend"], x="dt", y="cnt", color="decision", title="Decision 비율 추세")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Tool success rate 요약")
st.dataframe(data["tool"])
