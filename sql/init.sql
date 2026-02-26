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
  case_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  tags_jsonb JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
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
