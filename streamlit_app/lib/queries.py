"""대시보드 KPI SQL 모음.

각 쿼리는 공통 필터(기간/tenant/type/subtype/decision/response_mode/risk/prompt/model)를 적용한다.
성능을 위해 `metrics_events` 인덱스(metric_name, tenant expr, decision expr, GIN tags)를 전제로 설계했다.

주의사항:
    - 시계열은 DB timezone 영향이 있으므로 운영 환경에서 TZ 일관성 유지가 필요하다.
    - 이벤트 중복 적재가 있으면 카운트가 과대 집계될 수 있다.
"""

BASE_FILTER = """
WHERE created_at BETWEEN :start_ts AND :end_ts
  AND (:tenant_id = '' OR tags_jsonb->>'tenant_id' = :tenant_id)
  AND (:type = '' OR tags_jsonb->>'type' = :type)
  AND (:subtype = '' OR tags_jsonb->>'subtype' = :subtype)
  AND (:decision = '' OR tags_jsonb->>'decision' = :decision)
  AND (:response_mode = '' OR tags_jsonb->>'response_mode' = :response_mode)
  AND (:risk_level = '' OR tags_jsonb->>'risk_level' = :risk_level)
  AND (:prompt_version = '' OR tags_jsonb->>'prompt_version' = :prompt_version)
  AND (:model = '' OR tags_jsonb->>'model' = :model)
"""

# KPI 정의:
# - total_cases: 기간 내 고유 case 수
# - avg/p95 latency: decision_made.latency_ms 기반
# - cost_per_case: llm_usage.estimated_cost_usd / case_count
# - decision rate: decision_made 이벤트 비율
KPI_OVERVIEW = f"""
SELECT
  count(DISTINCT case_id) AS total_cases,
  avg((tags_jsonb->>'latency_ms')::float) FILTER (WHERE metric_name='decision_made') AS avg_latency,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY (tags_jsonb->>'latency_ms')::float)
    FILTER (WHERE metric_name='decision_made' AND (tags_jsonb->>'latency_ms') IS NOT NULL) AS p95_latency,
  sum((tags_jsonb->>'estimated_cost_usd')::float) FILTER (WHERE metric_name='llm_usage') / nullif(count(DISTINCT case_id),0) AS cost_per_case,
  avg(CASE WHEN tags_jsonb->>'decision'='AUTO_POST' THEN 1 ELSE 0 END) FILTER (WHERE metric_name='decision_made') AS auto_post_rate,
  avg(CASE WHEN tags_jsonb->>'decision'='DRAFT' THEN 1 ELSE 0 END) FILTER (WHERE metric_name='decision_made') AS draft_rate,
  avg(CASE WHEN tags_jsonb->>'decision'='ASK_MORE' THEN 1 ELSE 0 END) FILTER (WHERE metric_name='decision_made') AS ask_more_rate,
  avg(CASE WHEN tags_jsonb->>'decision'='HANDOFF' THEN 1 ELSE 0 END) FILTER (WHERE metric_name='decision_made') AS handoff_rate
FROM metrics_events
{BASE_FILTER}
"""

# Decision 트렌드(일 단위): stacked area chart 데이터셋
DECISION_TREND = f"""
SELECT date_trunc('day', created_at) AS dt, tags_jsonb->>'decision' AS decision, count(*) AS cnt
FROM metrics_events
{BASE_FILTER} AND metric_name='decision_made'
GROUP BY 1,2 ORDER BY 1
"""

# Tool 신뢰성:
# - success_rate = 성공 호출 / 전체 호출
# - p95_latency = 도구 지연 95퍼센타일
TOOL_RELIABILITY = f"""
SELECT tags_jsonb->>'tool_name' AS tool_name,
       avg(CASE WHEN (tags_jsonb->>'success')::boolean THEN 1 ELSE 0 END) AS success_rate,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY coalesce((tags_jsonb->>'latency_ms')::float,0)) AS p95_latency,
       count(*) AS calls
FROM metrics_events
{BASE_FILTER} AND metric_name='tool_call'
GROUP BY 1 ORDER BY calls DESC
"""

# Validator 실패 파레토(phase/reason 상위)
VALIDATOR_TOP = f"""
SELECT tags_jsonb->>'phase' AS phase, tags_jsonb->>'fail_reason_code' AS reason, count(*) AS cnt
FROM metrics_events
{BASE_FILTER} AND metric_name='validator_result' AND (tags_jsonb->>'passed')='false'
GROUP BY 1,2 ORDER BY cnt DESC LIMIT 20
"""

# 모델 성능:
# - mismatch_rate / multi_issue_rate는 classification_completed 기반 평균
MODEL_PERF = f"""
SELECT
  avg(CASE WHEN (tags_jsonb->>'is_mismatch')='true' THEN 1 ELSE 0 END) AS mismatch_rate,
  avg(CASE WHEN (tags_jsonb->>'is_multi_issue')='true' THEN 1 ELSE 0 END) AS multi_issue_rate,
  tags_jsonb->>'risk_level' AS risk_level,
  count(*) AS cnt
FROM metrics_events
{BASE_FILTER} AND metric_name='classification_completed'
GROUP BY risk_level
"""

# Business Insight:
# 유형/상세유형 접수량 Top N
BUSINESS_TOP = f"""
SELECT tags_jsonb->>'type' AS type, tags_jsonb->>'subtype' AS subtype, count(*) AS cnt
FROM metrics_events
{BASE_FILTER} AND metric_name='case_received'
GROUP BY 1,2 ORDER BY cnt DESC LIMIT 20
"""

# Case Explorer 목록:
# tenant_id 필터는 state_jsonb.input.tenant_id에 적용
CASE_LIST = f"""
SELECT case_id, updated_at, state_jsonb
FROM cases
WHERE (:tenant_id = '' OR state_jsonb->'input'->>'tenant_id' = :tenant_id)
ORDER BY updated_at DESC
LIMIT 200
"""
