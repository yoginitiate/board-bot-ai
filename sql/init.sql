CREATE TABLE IF NOT EXISTS cases (
  case_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  state_jsonb JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_logs (
  id BIGSERIAL PRIMARY KEY,
  case_id TEXT NOT NULL,
  node TEXT NOT NULL,
  event TEXT NOT NULL,
  payload_jsonb JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics_events (
  id BIGSERIAL PRIMARY KEY,
  case_id TEXT,
  metric_name TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  tags_jsonb JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tenant_policies (
  tenant_id TEXT NOT NULL,
  type TEXT NOT NULL,
  subtype TEXT NOT NULL,
  policy_jsonb JSONB NOT NULL,
  version TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY(tenant_id, type, subtype)
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_created ON metrics_events(metric_name, created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_tenant_created ON metrics_events((tags_jsonb->>'tenant_id'), created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_decision_created ON metrics_events((tags_jsonb->>'decision'), created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_tags_gin ON metrics_events USING GIN(tags_jsonb);
