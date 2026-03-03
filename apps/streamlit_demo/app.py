from __future__ import annotations

import json

import requests
import streamlit as st

API_URL = st.sidebar.text_input("API URL", "http://api:8000")

st.title("게시판 Agent MVP 데모")
with st.form("case_form"):
    ctype = st.selectbox("문의유형", ["배송", "클레임", "반품"])
    subtype = st.selectbox("상세유형", ["지연", "파손", "오배송"])
    title = st.text_input("제목", "배송 지연 문의")
    body = st.text_area("문의사항", "주문한 상품이 도착하지 않았습니다.")
    uploaded = st.file_uploader("이미지", accept_multiple_files=True)
    submit = st.form_submit_button("실행")

if submit:
    attachments = [{"filename": f.name, "content_type": f.type} for f in uploaded] if uploaded else []
    payload = {
        "tenant_id": "tenant-a",
        "payload": {"type": ctype, "subtype": subtype, "title": title, "body": body, "attachments": attachments},
    }
    res = requests.post(f"{API_URL}/v1/cases/process", json=payload, timeout=30)
    st.json(res.json())

case_id = st.text_input("case_id 조회")
if st.button("리플레이") and case_id:
    res = requests.get(f"{API_URL}/v1/cases/{case_id}", timeout=30)
    st.code(json.dumps(res.json(), ensure_ascii=False, indent=2))
