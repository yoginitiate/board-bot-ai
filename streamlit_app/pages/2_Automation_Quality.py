"""Automation & Quality 페이지.

운영자는 자동화 퍼널 누수 지점과 Validator 실패 원인을 확인해
정책/프롬프트/도구 개선 우선순위를 결정한다.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.lib.metrics import load_overview

filters = st.session_state.get("filters", {})
data = load_overview(filters)

st.title("Automation & Quality")
if not data["decision_trend"].empty:
    latest = data["decision_trend"].groupby("decision", as_index=False)["cnt"].sum()
    fig = px.funnel(latest, x="cnt", y="decision", title="Received -> Decision Funnel")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Validator 실패 Top")
st.dataframe(data["validator"])

if not data["validator"].empty:
    g = data["validator"][data["validator"]["phase"] == "GROUNDING"]
    if not g.empty:
        fig = px.bar(g, x="reason", y="cnt", title="Grounding fail")
        st.plotly_chart(fig, use_container_width=True)
