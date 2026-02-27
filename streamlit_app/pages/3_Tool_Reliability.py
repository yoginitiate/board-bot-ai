"""Tool Reliability 페이지.

운영자는 도구별 성공률/지연을 비교해 장애 대응 및 타임아웃 튜닝 대상을 선정한다.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.lib.metrics import load_overview

filters = st.session_state.get("filters", {})
tool = load_overview(filters)["tool"]

st.title("Tool Reliability")
if not tool.empty:
    st.plotly_chart(px.bar(tool, x="tool_name", y="success_rate", title="Tool success rate"), use_container_width=True)
    st.plotly_chart(px.bar(tool, x="tool_name", y="p95_latency", title="Tool p95 latency"), use_container_width=True)
    st.dataframe(tool[["tool_name", "calls", "success_rate", "p95_latency"]])
else:
    st.info("No tool data")
