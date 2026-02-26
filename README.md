# Board Bot AI MVP

LangGraph 기반 Master/Executor 분리형 게시판 Agent MVP입니다.

## 구성
- FastAPI: `POST /v1/cases/process`, `GET /v1/cases/{case_id}`
- FastMCP: 도메인 조회/RAG/비전/등록 Tool
- Streamlit: 입력/실행/리플레이 데모
- 저장소: Postgres, Milvus, MinIO

## 로컬 실행 (uv)
```bash
uv sync
uv run uvicorn apps.api.app.main:app --reload --port 8000
uv run uvicorn apps.mcp_server.app.main:app --reload --port 9000
uv run streamlit run apps/streamlit_demo/app.py
```

## Docker Compose
```bash
docker compose up --build
```

## 환경변수
- `GEMINI_API_KEY` : Gemini API Key
- `POSTGRES_DSN` : PostgreSQL 접속 문자열

## 테스트
```bash
uv run pytest
```

## 프롬프트/설정 버전
- 프롬프트는 `prompts/*.yaml`에 저장되고 해시(version_hash) 계산
- 설정은 `config/app.yaml` 버전 필드 사용
- 실행로그에 `prompt_version`, `config_version` 적재

## 참고
- HANDOFF 발생 시 tool 호출 없이 즉시 종료
- Validator Gate: PII -> Policy -> Grounding 순서
