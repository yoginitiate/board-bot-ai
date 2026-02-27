# Board Bot AI MVP + Dashboard

## 설정 구조 분리 (중요)
기존 단일 `config/app.yaml`에서 아래 두 파일로 역할을 분리했습니다.

- `config/system.yaml`:
  - 인프라/런타임(POSTGRES, Milvus, MinIO, MCP endpoint, timeout/retry)
  - **플랫폼/인프라 팀**이 환경별(dev/stg/prod)로 관리
- `config/policy.yaml`:
  - 서비스 정책(LLM 모델군, RAG top_k/fusion, prevision 임계치, validation/decision/routing 정책)
  - **서비스/운영/ML 팀**이 품질/비용/리스크 정책으로 관리

`config/app.yaml`은 하위호환용 포인터 파일로 유지됩니다.

## Config 로딩 우선순위
`AppConfig` 우선순위는 다음과 같습니다.

1. 환경변수 override (`POSTGRES_DSN`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `MCP_URL`)
2. `config/system.yaml` + `config/policy.yaml`
3. 코드 기본값

## 실행
```bash
docker compose up --build
```

### 주요 엔드포인트
- API: `http://localhost:8000`
- MCP: `http://localhost:9002`
- Streamlit: `http://localhost:8501`

## 필수/권장 환경변수
- 필수(운영): `GOOGLE_API_KEY`
- 호환: `GEMINI_API_KEY`
- 선택: `POSTGRES_DSN`, `MCP_URL`

> 보안 주의: 운영 환경에서는 키/비밀번호를 YAML에 직접 쓰지 말고 Secret/Env로 주입하세요.

## 데이터 시드
```bash
python scripts/load_sample_rag.py
python scripts/load_intent_examples.py
python scripts/seed_dashboard_data.py
```

## 정책/런타임 변경 가이드
- 인프라 변경(호스트/포트/타임아웃/재시도): `config/system.yaml`
- 모델/검증/라우팅/결정 정책 변경: `config/policy.yaml`

## Agent 동작 요약
- Master: normalize/mask → classify/risk → prevision gate → plan
- HANDOFF: 별도 `HandoffNotify` 노드로 분기 + idempotent 알림
- Executor: scenario/agent 생성 → PII/Policy/Grounding 검증 → Decision(auto_post/draft)

## 테스트
```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall src apps scripts streamlit_app
```
