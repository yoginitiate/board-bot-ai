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
- **배포**: docker-compose로 API/MCP/DB/Milvus/MinIO/Streamlit 동시 실행

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
- 메타 필드: `doc_id`, `source_type`, `text`, `metadata`, `tenant_id`

### 인덱스
- `sparse_vector`: `SPARSE_INVERTED_INDEX` + `IP`
- `dense_vector`: `AUTOINDEX` + `IP`

### 검색 흐름
1. query dense embedding 생성
2. query sparse(BM25) vector 생성
3. `AnnSearchRequest` 2개 생성(dense/sparse)
4. `WeightedRanker(sparse_weight, dense_weight)` 적용
5. `Collection.hybrid_search(...)` 실행

### 가중치 튜닝
`config/policy.yaml`:
- `rag.hybrid.dense_weight`
- `rag.hybrid.sparse_weight`
- `rag.hybrid.search_limit_per_field`
- `rag.hybrid.final_top_k`

### 검증
- `scripts/verify_hybrid_search.py` 실행 시 `debug.path = milvus_native_hybrid_search` 확인
