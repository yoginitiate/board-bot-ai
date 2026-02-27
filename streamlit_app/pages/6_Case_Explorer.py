"""Case Explorer 페이지.

차트에서 이상 구간을 발견했을 때 개별 케이스 상태를 점검하는 드릴다운 종착지다.
"""

from __future__ import annotations

import json

import streamlit as st

from streamlit_app.lib.db import query_df
from streamlit_app.lib.queries import CASE_LIST

filters = st.session_state.get("filters", {})
rows = query_df(CASE_LIST, {"tenant_id": filters.get("tenant_id", "")})

st.title("Case Explorer")
st.dataframe(rows[["case_id", "updated_at"]] if not rows.empty else rows)
case_id = st.selectbox("case_id", rows["case_id"].tolist() if not rows.empty else [])
if case_id and not rows.empty:
    detail = rows[rows["case_id"] == case_id].iloc[0]["state_jsonb"]
    st.json(detail.get("input", {}))
    st.subheader("issues")
    st.json(detail.get("classification", {}).get("issues", []))
    st.subheader("plan")
    st.json(detail.get("plan", {}))
    st.subheader("tool trace")
    st.json(detail.get("tooling", {}).get("trace", []))
    st.subheader("decision")
    st.json(detail.get("decision", {}))
    st.subheader("validator metrics")
    vm = [m for m in detail.get("telemetry", {}).get("metrics", []) if m.get("metric_name") == "validator_result"]
    st.code(json.dumps(vm, ensure_ascii=False, indent=2))
