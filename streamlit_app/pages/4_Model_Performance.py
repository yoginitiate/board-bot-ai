"""Model/Prompt Performance 페이지.

운영자는 mismatch/multi-issue/risk 분포를 통해 분류 품질과 프롬프트 버전 안정성을 점검한다.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.lib.metrics import load_overview

filters = st.session_state.get("filters", {})
model = load_overview(filters)["model"]

st.title("Model/Prompt Performance")
st.dataframe(model)
if not model.empty:
    st.plotly_chart(px.pie(model, names="risk_level", values="cnt", title="Risk 분포"), use_container_width=True)
