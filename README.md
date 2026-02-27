# Board Bot AI MVP + Dashboard

LangGraph 기반 게시판 Agent와 Business Insight/AI KPI 대시보드 프로토타입입니다.

## 서비스 구성
- API: FastAPI (`/v1/cases/process`, `/v1/cases/{case_id}`)
- MCP: FastMCP tool 서버
- Dashboard: Streamlit 멀티 페이지 (`streamlit_app/`)
- Storage: Postgres, Milvus, MinIO

## 실행
```bash
docker compose up --build
```
- API: http://localhost:8000
- MCP: http://localhost:9002
- Streamlit Dashboard: http://localhost:8501

## 샘플 데이터 생성
```bash
python scripts/seed_dashboard_data.py
```
또는 컨테이너 내부에서:
```bash
docker compose exec api python scripts/seed_dashboard_data.py
```

## 대시보드 페이지
1. Executive Overview
2. Automation & Quality
3. Tool Reliability
4. Model/Prompt Performance
5. Business Insight
6. Case Explorer

## 공통 필터
Sidebar에서 기간(시작/종료), tenant_id, type/subtype, decision, response_mode, risk_level, prompt_version, model 필터를 공통 적용합니다.

## KPI 정의
- AUTO_POST Rate = AUTO_POST / total decisions
- STP = AUTO_POST / (AUTO_POST + DRAFT + ASK_MORE + HANDOFF)
- P95 latency = decision_made.latency_ms 95퍼센타일
- Tool success rate = success true / total tool_call
- Grounding fail rate = validator_result(phase=GROUNDING, passed=false) / phase=GROUNDING 전체
- Cost/case = llm_usage.estimated_cost_usd 합 / case 수

## 이벤트 로깅 규격
`metrics_events.metric_name`:
- case_received
- classification_completed
- llm_usage
- tool_call
- validator_result
- decision_made
- post_reply_result

공통 tags에는 tenant_id/case_id/type/subtype/response_mode/decision/prompt_version/config_version/model/success/error_code/latency_ms 등을 저장합니다(PII 원문 저장 금지).

## SQL 인덱스
`sql/init.sql`에 다음 인덱스를 포함했습니다.
- (metric_name, created_at)
- ((tags_jsonb->>'tenant_id'), created_at)
- ((tags_jsonb->>'decision'), created_at)
- GIN(tags_jsonb)


## 한글 문서/주석 정책
- Docstring 스타일은 **Google 스타일**로 통일합니다.
- 모든 public 함수/클래스/모듈에 한글 docstring을 작성합니다.
- PII/민감정보는 원문 저장 금지이며, 로그/메트릭에는 요약/코드/카운트만 기록합니다.
- LangGraph 노드/게이트/예산 정책은 "왜" 중심 주석을 우선 작성합니다.
- Dashboard SQL에는 KPI 산식, 필터 범위(tenant/date), 성능 고려(인덱스), 함정(time zone/중복)을 주석으로 명시합니다.

## metrics_events 이벤트/태그 요약
- metric_name: `case_received`, `classification_completed`, `llm_usage`, `tool_call`, `validator_result`, `decision_made`, `post_reply_result`
- 공통 tags: `tenant_id`, `case_id`, `type`, `subtype`, `response_mode`, `decision`, `prompt_version`, `config_version`, `model`, `success`, `error_code`, `latency_ms`
- 원칙: 이벤트는 관측 목적 메타데이터만 저장하고, 고객 원문/PII는 저장하지 않습니다.
