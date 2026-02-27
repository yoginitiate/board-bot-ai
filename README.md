# Board Bot AI MVP + Dashboard

## 실행
```bash
docker compose up --build
```

## 필수 환경변수
- `GOOGLE_API_KEY` (Gemini Text/MM/Embedding)
- 선택: `GEMINI_API_KEY` (호환)

## 데이터 시드
```bash
python scripts/load_sample_rag.py
python scripts/load_intent_examples.py
python scripts/seed_dashboard_data.py
```

## 이번 구현 핵심
- `normalize_and_mask`: 마스킹 + 경량 전처리만 수행
- `classify_risk_route`:
  - intent 예문(`intent_examples`)을 Milvus에서 top_k 검색 후 LLM 컨텍스트로 사용
  - intent 허용 라벨 6개 고정: 상품 누락/오배송/미배송/품절/파손/주문 취소
  - risk는 intent와 분리하여 `risk` 필드로만 반환
- HANDOFF 분기:
  - `HandoffNotify` 노드 추가
  - `notify_handoff` MCP tool로 webhook 알림
  - Postgres `handoff_notifications`로 idempotent 보장
- `prevision`:
  - 지정 intent + 첨부 이미지가 있을 때만 실행
  - Gemini multimodal(`google-genai`) 호출
  - `label/confidence/unknown_reason`를 state에 저장
- `plan_builder`:
  - policy 기반 SCENARIO/AGENT 결정
  - 다중 문의(최대 3개)와 `max_parallel` 계획 포함
  - 슬롯 추출(체인+Pydantic) 후 state 저장
- `scenario_subgraph`:
  - YAML 템플릿 렌더 + sanitize(PII/정책문구)
- `agent_generate_subgraph`:
  - LangChain tool-calling agent + MCP Tool 래핑
  - tool trace/state/metrics 기록
- `validate_grounding`:
  - 문의 커버리지 + citation + 정책 점수화
  - 임계치 미달 시 기본 2회 재작성 루프
- `decision_node`:
  - `AUTO_POST` 또는 `DRAFT`만 수행
  - HANDOFF는 decision 이전 분기에서 종료
- 공통 로깅:
  - 모든 노드 `node_execution`(latency/success/fail)
  - PII 원문 저장 금지(마스킹 텍스트 길이 제한 태그만 기록)

## Agent 모드 Tool Calling 검증
1. `type=클레임`, `subtype=파손` 요청으로 `/v1/cases/process` 호출
2. 응답 `tool_trace` 또는 저장된 state의 `tooling.agent_steps` 확인
3. `metrics_events`에서 `metric_name='tool_call'` 이벤트 확인

## 테스트
```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall src apps scripts streamlit_app
```
