# Board Bot AI MVP + Dashboard

LangGraph 기반 게시판 Agent와 Business Insight/AI KPI 대시보드입니다.

## 서비스 구성
- API: FastAPI (`/v1/cases/process`, `/v1/cases/{case_id}`)
- MCP: FastMCP tool 서버 (`rag_search`, `vision_triage` 포함)
- Dashboard: Streamlit 멀티 페이지 (`streamlit_app/`)
- Storage: Postgres, Milvus, MinIO

## 실행
```bash
docker compose up --build
```
- API: http://localhost:8000
- MCP: http://localhost:9002
- Streamlit Dashboard: http://localhost:8501

## 환경변수
- `GOOGLE_API_KEY`: Gemini Text/MM/Embedding 공통 API 키
- `GEMINI_API_KEY`: 하위호환 키(선택)

## Hybrid RAG 설정
- Dense: Milvus (`rag_policy`, `rag_manual`, `rag_script`) + Gemini Embedding
- Sparse: Postgres `rag_documents` + BM25(rank-bm25)
- Fusion: RRF (dense+sparse 결합)
- 기본 값은 `config/app.yaml`의 `rag.hybrid` 및 `gemini.embedding_model` 참조

## 샘플 데이터 적재
```bash
python scripts/load_sample_rag.py
python scripts/seed_dashboard_data.py
```
또는 컨테이너 내부:
```bash
docker compose exec api python scripts/load_sample_rag.py
docker compose exec api python scripts/seed_dashboard_data.py
```

## Agent 모드 Tool Calling 검증 방법
1. `type=클레임`, `subtype=파손`으로 `/v1/cases/process` 호출
2. 응답의 `tool_trace` 또는 저장된 case state의 `tooling.agent_steps` 확인
3. `metrics_events`에서 `metric_name='tool_call'` 이벤트가 1건 이상 기록되는지 확인
4. 최종 `draft.text`에 도구 실행 결과 반영 여부 확인

## 테스트/스모크 체크 예시
```bash
# unit
PYTHONPATH=src python -m pytest -q

# rag_search 툴 (MCP 컨테이너 실행 후)
curl -X POST http://localhost:9002/tools/rag_search -H "content-type: application/json" -d '{"query":"환불 정책","sources":["policy"],"top_k":5,"tenant_id":"tenant-1"}'

# vision triage 툴
curl -X POST http://localhost:9002/tools/vision_triage -H "content-type: application/json" -d '{"attachments":[{"path":"/app/tests/data/sample.jpg"}]}'
```

## 문서/보안 원칙
- PII 원문 저장 금지 (로그/지표는 코드/카운트/요약만 저장)
- 분류/검증/생성 노드는 Runnable chain + Pydantic parser 기반으로 JSON 파싱 안정성 확보
- 고위험 판단 시 HANDOFF 우선으로 도구 호출을 최소화
