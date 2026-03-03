"""Business Insight 페이지.

운영자는 유형/상세유형 상위 분포를 통해 문의 급증 영역을 식별하고
운영 정책/인력 배치를 조정한다.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.lib.metrics import load_overview

filters = st.session_state.get("filters", {})
biz = load_overview(filters)["business"]

st.title("Business Insight")
st.dataframe(biz)
if not biz.empty:
    st.plotly_chart(px.bar(biz, x="type", y="cnt", color="subtype", title="유형/상세유형 Top"), use_container_width=True)
