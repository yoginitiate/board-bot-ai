

## 0) 외부 인프라 운영 전제(중요)

- PostgreSQL/Milvus는 **기존 운영 중인 외부 서비스**를 사용합니다.
- docker-compose 기본 기동 대상은 애플리케이션(API/MCP/Streamlit)만 포함합니다.
- 필수 환경변수 예시:
  - `POSTGRES_DSN=postgresql+psycopg://<user>:<password>@<db-host>:5432/<db>`
  - `OPENAI_API_KEY=...` (또는 `GOOGLE_API_KEY`)
  - `MILVUS_HOST=10.101.2.210`, `MILVUS_PORT=19530`

운영 DB 반영 시에는 `sql/init.sql`을 DBA 승인 절차에 따라 적용하세요.
해당 SQL은 `CREATE SCHEMA IF NOT EXISTS board_bot` + `CREATE TABLE IF NOT EXISTS` 기반으로 idempotent 하게 작성되어 있습니다.
# Board Bot Agent 플랫폼 (MVP)

게시판 문의를 자동 분류/분기/검증/응답 생성하는 **LangGraph 기반 AI 운영 플랫폼**입니다.
본 문서는 서비스 개요부터 아키텍처, 설정, 모델 전환, 검증 전략까지 전체 설계를 설명합니다.

---

## 1) 서비스 개요

### 게시판 Agent란?
상담AP에서 유입되는 문의를 받아 다음을 자동 수행합니다.
- 개인정보 마스킹
- 의도 분류 + 고위험 탐지
- 이미지 사전판단(prevision)
- 실행계획 수립(SCENARIO/AGENT)
- 도구 호출(MCP) + 근거 검증
- 최종 의사결정(AUTO_POST / DRAFT)

### 문제 정의
- 문의량 증가로 인한 상담 지연
- 유형 분류/정책 준수의 일관성 부족
- 고위험 민원에 대한 즉시 대응 필요
- 모델/정책 변경 시 운영 복잡도 증가

### 해결 접근
- Master/Executor 분리로 책임 명확화
- YAML 정책 + 환경별 런타임 설정 분리
- Hybrid RAG(Dense+Sparse+RRF)로 근거 강화
- Validator 3단계(PII/Policy/Grounding)로 안전성 확보

---

## 2) End-to-End 처리 프로세스

### 단계 설명
1. Normalize & Mask
2. Classify + Risk
3. Handoff 분기(고위험 즉시 알림)
4. PreVision(조건부)
5. Plan Builder
6. Scenario/Agent 실행
7. Validate(PII/Policy/Grounding)
8. Decision(AUTO_POST/DRAFT)
9. Logging & Metrics

### ASCII 다이어그램
```text
[Request]
   |
   v
[NormalizeAndMask] --(PII 제거)--> [ClassifyRiskRoute]
                                   |             |
                                   | HANDOFF     | NORMAL
                                   v             v
                             [HandoffNotify]  [PreVisionNeedGate]
                                   |             |
                                   v             v
                             [Telemetry]      [PreVision] -> [PolicyLoader] -> [PlanBuilder]
                                                                  |
                                                                  v
                                                           [PlanEngine]
                                                                  |
                                                                  v
                                                           [ResponseModeGate]
                                                          /                  \
                                                [ScenarioSubgraph]   [AgentGenerateSubgraph]
                                                          \                  /
                                                           v                v
                                                          [ValidatePII] -> [ValidatePolicy] -> [ValidateGrounding]
                                                                                             |
                                                                                             v
                                                                                         [Decision]
                                                                                             |
                                                                                             v
                                                                                          [Output]
                                                                                             |
                                                                                             v
                                                                                         [Telemetry]
```

---

## 3) 아키텍처 구성

- **Master 영역**: 입력 안전화, 분류/위험 판정, 사전 신호, 정책/계획 수립
- **Executor 영역**: 계획 실행, 응답 생성, 검증, 최종 출력
- **MCP 서버**: RAG/비전/도메인 조회/알림 도구 제공
- **저장소**:
  - PostgreSQL: 상태/로그/지표/알림 중복방지
  - Milvus: 문서/의도 예문 벡터 검색
  - MinIO: 첨부 파일 저장(운영 확장)
- **배포**: 외부 PostgreSQL/Milvus를 사용하고 docker-compose는 기본적으로 API/MCP/Streamlit만 실행

---

## 4) 기술 스택

- Python 3.12
- LangChain / LangGraph
- Gemini / OpenAI (설정 기반 전환)
- Milvus
- PostgreSQL
- FastAPI / FastMCP
- Streamlit
- Docker / uv

---

## 5) 설정 구조

설정은 두 파일로 분리합니다.

- `config/system.yaml` (인프라/런타임)
  - postgres/milvus/minio/mcp/observability
  - timeout/retry/circuit_breaker
- `config/policy.yaml` (서비스 정책)
  - llm provider/model/api_key_env
  - rag/prevision/validation/routing/tools/decision 정책

`config/app.yaml`은 하위호환 포인터 파일(Deprecated)입니다.

### 우선순위
`env > (system.yaml + policy.yaml) > 코드 기본값`

주요 env:
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`
- `POSTGRES_DSN`
- `MCP_URL`

---

## 6) 모델 교체 방법 (Google ↔ OpenAI)

`config/policy.yaml`의 `policy.llm.provider`를 변경합니다.

### Google 예시
```yaml
policy:
  llm:
    provider: google
    text_model: gemini-1.5-flash
    mm_model: gemini-2.0-flash
    embedding_model: text-embedding-004
    api_key_env: GOOGLE_API_KEY
```

### OpenAI 예시
```yaml
policy:
  llm:
    provider: openai
    text_model: gpt-4o-mini
    mm_model: gpt-4o
    embedding_model: text-embedding-3-large
    api_key_env: OPENAI_API_KEY
```

> 멀티모달 주의: OpenAI 사용 시 이미지는 data URL로 변환되어 전송됩니다.

---

## 7) RAG / Hybrid Retriever

- Dense: Milvus 벡터 검색
- Sparse: PostgreSQL 문서 코퍼스 기반 BM25
- Fusion: RRF(Reciprocal Rank Fusion)

최종 score는 `fusion_score`로 정렬하며, 각 hit의 metadata에 dense/sparse 점수를 함께 남깁니다.

---

## 8) Multi-Issue 처리 방식

- 분류 결과에서 최대 3개 issue 추출
- 독립 issue는 `max_parallel` 범위 내 병렬 처리 후보로 표시
- 최종 답변은 섹션 단위로 합성

---

## 9) Validator 전략

### PII 검증
출력 직전 2차 마스킹으로 재노출 방지

### Policy 검증
금칙/정책 위반 표현 제거 또는 수정

### Grounding 검증
- 문의 커버리지
- citation 존재 여부
- 정책 준수

복합 점수가 임계치 미달이면 재작성 루프(기본 2회) 수행 후 DRAFT 전환

---

## 10) 확장 전략

- 테넌트별 policy override
- MCP Tool 추가(재고/클레임/물류 외부 시스템)
- 시나리오 템플릿 확장
- 비용 최적화(캐시, 모델 라우팅, budget 정책)
- 알림 채널 확장(webhook/email/push)

---

## 실행 가이드

```bash
docker compose up --build

# 로컬 인프라가 꼭 필요할 때만(기본 OFF)
# docker compose --profile local-infra up -d postgres_local milvus_local
```

### 데이터 시드
```bash
python scripts/load_sample_rag.py
python scripts/load_intent_examples.py
python scripts/seed_dashboard_data.py
```

### 테스트
```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall src apps scripts streamlit_app
```


## 11) Milvus Native Hybrid Search 구현

본 프로젝트의 `rag_search`는 Milvus 공식 방식의 **native hybrid_search**를 사용합니다.

### 컬렉션 스키마
- `dense_vector`: `FLOAT_VECTOR`
- `sparse_vector`: `SPARSE_FLOAT_VECTOR`
- 메타 필드: `doc_id`, `source_type`, `page_content`, `metadata`, `tenant_id`

### 인덱스
- `sparse_vector`: `SPARSE_INVERTED_INDEX` + `IP`
- `dense_vector`: `AUTOINDEX` + `IP`

### 검색 흐름
1. Dense query embedding 생성(provider: Google/OpenAI 전환 가능)
2. BM25 sparse query vector 생성(LLM provider와 무관)
3. `AnnSearchRequest` 2개 생성(`dense_vector`, `sparse_vector`)
4. `WeightedRanker(w_sparse, w_dense)` 적용
5. `Collection.hybrid_search(...)` 실행

### 토크나이저/스파스 전략
- `rag.hybrid.tokenizer.use_kiwi=true`면 Kiwi 우선, 미설치 시 정규식 fallback
- `rag.hybrid.vocab.mode`
  - `fixed`: 기존 vocab/idf를 고정 사용(미등록 토큰 무시)
  - `rebuild`: 적재 시 코퍼스로 vocab/idf 재구축

### 가중치 튜닝
`config/policy.yaml`:
- `rag.hybrid.weights.dense`
- `rag.hybrid.weights.sparse`
- `rag.hybrid.search_limit_per_field`
- `rag.hybrid.final_top_k`

### 검증
- `scripts/verify_hybrid_search.py` 실행 시 `debug.path = milvus_native_hybrid_search` 확인
- `pytest -q tests/test_hybrid_retriever.py`로 hybrid 호출 경로/가중치 sanity 확인

## 12) 용어사전(전문 용어) 기능

### 데이터 흐름
1. `NormalizeAndMask` 이후 `TermExtract` 노드에서 **masked_text만** 사용해 용어 추출
2. 분류(`classify_risk_route`)에서 용어 정의를 컨텍스트 주입(append) 또는 query_expand 방식으로 활용
3. RAG 계획(`plan_builder`)에서
   - `dense_query_text`: 원문 + 용어 정의 요약
   - `sparse_query_text`: 원문 + 용어/동의어 키워드
   를 분리 생성해 `rag_search`에 전달
4. 검증(`validate_grounding`)에서 답변의 용어 사용 맥락을 soft rule로 점검

### 운영 정책 (`config/policy.yaml`)
- `glossary.enabled`
- `glossary.max_terms`
- `glossary.max_synonyms_per_term`
- `glossary.description_max_chars`
- `glossary.injection.mode` (`append_context` | `query_expand`)
- `glossary.expand.for_dense`
- `glossary.expand.for_sparse`
- `glossary.expose_to_customer`
- `glossary.refresh.enabled`, `glossary.refresh.ttl_seconds`

### PII/로그 정책
- 용어 추출 입력은 `masked_text`만 사용
- telemetry에는 `term_count`, `query_expanded_term_count`, `dense_query_length`, `sparse_query_length` 등 통계만 기록(원문/설명 저장 금지)


## 13) 운영 초기화/검증 절차 (외부 PostgreSQL/Milvus)

1. PostgreSQL 초기 스키마 생성(승인 후)
   - `sql/init.sql` 적용 → `board_bot.t_*` 테이블 생성
2. Milvus 컬렉션 검증
   - `PYTHONPATH=src python scripts/init_milvus_collections.py --collection semas_v2`
   - 필수 필드(`dense_vector`, `sparse_vector`) 없으면 hybrid 검색 불가
3. BM25 vocab/idf 생성
   - `PYTHONPATH=src python scripts/build_bm25_vocab.py`
4. RAG 문서를 Milvus로 적재
   - `PYTHONPATH=src python scripts/load_rag_to_milvus.py`
5. MCP rag_search 호출 검증
   - `PYTHONPATH=src python scripts/verify_hybrid_search.py`

### 하이브리드 튜닝 포인트
- `hybrid_search.dense.nprobe`, `dense.limit`: dense recall/latency 균형
- `hybrid_search.sparse.k`, `sparse.limit`: 키워드 매칭 폭 제어
- `hybrid_search.ranker.type`, `rrf_k`, `weights`: dense/sparse 융합 전략
- `rag.llm_docs_limit`: LLM 입력 문서 수 상한(비용/지연 제어)
