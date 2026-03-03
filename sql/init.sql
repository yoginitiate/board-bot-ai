-- ============================================================================
-- board_bot 초기 스키마 (idempotent)
-- ----------------------------------------------------------------------------
-- 목적:
--   - 운영 중인 외부 PostgreSQL에 board_bot 스키마/테이블이 없을 때만 생성
--   - 애플리케이션이 사용하는 최소 테이블 구조를 일관되게 보장
-- 보안 원칙:
--   - 원문 PII/민감정보 저장 금지 (마스킹/요약/통계만 저장)
--   - metrics/events에는 원문 본문을 넣지 말고 집계 가능한 태그만 기록
-- 운영 주의:
--   - 운영 DB 반영은 DBA 승인/권한 정책에 따라 수행
--   - 본 SQL은 CREATE IF NOT EXISTS 기반이라 재실행 가능
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS board_bot;
SET search_path TO board_bot;

-- 케이스 상태 스냅샷 (마스킹된 상태 중심 저장)
CREATE TABLE IF NOT EXISTS board_bot.t_cases (
  case_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  state_jsonb JSONB NOT NULL
);

-- 실행 이벤트 로그 (원문 대신 노드/요약 payload)
CREATE TABLE IF NOT EXISTS board_bot.t_execution_logs (
  id BIGSERIAL PRIMARY KEY,
  case_id TEXT NOT NULL,
  node TEXT NOT NULL,
  event TEXT NOT NULL,
  payload_jsonb JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

-- 메트릭 이벤트 (PII 금지, 통계 태그만)
CREATE TABLE IF NOT EXISTS board_bot.t_metrics_events (
  id BIGSERIAL PRIMARY KEY,
  case_id TEXT,
  metric_name TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  tags_jsonb JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS board_bot.t_tenant_policies (
  tenant_id TEXT NOT NULL,
  type TEXT NOT NULL,
  subtype TEXT NOT NULL,
  policy_jsonb JSONB NOT NULL,
  version TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY(tenant_id, type, subtype)
);

CREATE INDEX IF NOT EXISTS idx_t_metrics_name_created ON board_bot.t_metrics_events(metric_name, created_at);
CREATE INDEX IF NOT EXISTS idx_t_metrics_tenant_created ON board_bot.t_metrics_events((tags_jsonb->>'tenant_id'), created_at);
CREATE INDEX IF NOT EXISTS idx_t_metrics_decision_created ON board_bot.t_metrics_events((tags_jsonb->>'decision'), created_at);
CREATE INDEX IF NOT EXISTS idx_t_metrics_tags_gin ON board_bot.t_metrics_events USING GIN(tags_jsonb);

-- RAG 문서 코퍼스 (원문 정책에 따라 최소 범위 저장)
CREATE TABLE IF NOT EXISTS board_bot.t_rag_documents (
  doc_id TEXT PRIMARY KEY,
  tenant_id TEXT,
  source_type TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_t_rag_docs_source ON board_bot.t_rag_documents(source_type);
CREATE INDEX IF NOT EXISTS idx_t_rag_docs_tenant ON board_bot.t_rag_documents(tenant_id);

CREATE TABLE IF NOT EXISTS board_bot.t_handoff_notifications (
  case_id TEXT PRIMARY KEY,
  payload_jsonb JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- BM25 sparse_vector 생성을 위한 vocab/idf 저장소
-- 왜 Postgres에 저장하는가?
--   - 운영 Milvus와 분리해 vocab 재생성 정책(fixed/rebuild)을 제어
--   - 쿼리 시 동일 vocab/idf를 사용해 sparse score 일관성 확보
CREATE TABLE IF NOT EXISTS board_bot.t_rag_vocab (
  term TEXT PRIMARY KEY,
  term_index INT NOT NULL,
  df INT NOT NULL,
  idf DOUBLE PRECISION NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS board_bot.t_rag_corpus_stats (
  id INT PRIMARY KEY,
  doc_count INT NOT NULL,
  avgdl DOUBLE PRECISION NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 용어사전(전문 용어)
-- 분류 컨텍스트 주입 + RAG 쿼리 확장(키워드/동의어)에 사용
CREATE TABLE IF NOT EXISTS board_bot.t_term_dictionary (
  dictionary_no int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL,
  term text NOT NULL,
  regist_date timestamp(0) DEFAULT CURRENT_TIMESTAMP NULL,
  update_date timestamp NULL,
  description text NULL,
  CONSTRAINT t_term_dictionary_pkey PRIMARY KEY (dictionary_no)
);

CREATE TABLE IF NOT EXISTS board_bot.t_synonym_list (
  seq serial4 NOT NULL,
  synonym text NOT NULL,
  dictionary_no int4 NOT NULL,
  regist_date timestamp(0) DEFAULT CURRENT_TIMESTAMP NULL,
  CONSTRAINT t_synonym_list_pkey PRIMARY KEY (seq)
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'fk_t_synonym_list_dictionary_no'
      AND conrelid = 'board_bot.t_synonym_list'::regclass
  ) THEN
    ALTER TABLE board_bot.t_synonym_list
      ADD CONSTRAINT fk_t_synonym_list_dictionary_no
      FOREIGN KEY (dictionary_no)
      REFERENCES board_bot.t_term_dictionary(dictionary_no)
      ON DELETE CASCADE;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_t_term_dictionary_lower_term ON board_bot.t_term_dictionary (lower(term));
CREATE INDEX IF NOT EXISTS idx_t_synonym_list_lower_synonym ON board_bot.t_synonym_list (lower(synonym));
CREATE INDEX IF NOT EXISTS idx_t_synonym_list_dictionary_no ON board_bot.t_synonym_list (dictionary_no);
