-- Requires: glx_events, glx_grammar_events, glx_deltas
CREATE OR REPLACE VIEW glx_ab AS
SELECT
  g.subject AS unit,
  COALESCE(SUM(CASE WHEN d.layer='AST' THEN d.dS ELSE 0 END),0) AS S,
  COALESCE(SUM(CASE WHEN d.layer='MOZAIKA' THEN d.dH ELSE 0 END),0) AS H,
  CASE WHEN (COALESCE(SUM(CASE WHEN d.layer='AST' THEN d.dS ELSE 0 END),0) +
             COALESCE(SUM(CASE WHEN d.layer='MOZAIKA' THEN d.dH ELSE 0 END),0)) > 0
       THEN (COALESCE(SUM(CASE WHEN d.layer='AST' THEN d.dS ELSE 0 END),0)::double precision) /
            (COALESCE(SUM(CASE WHEN d.layer='AST' THEN d.dS ELSE 0 END),0) +
             COALESCE(SUM(CASE WHEN d.layer='MOZAIKA' THEN d.dH ELSE 0 END),0))
       ELSE NULL END AS alpha,
  CASE WHEN (COALESCE(SUM(CASE WHEN d.layer='AST' THEN d.dS ELSE 0 END),0) +
             COALESCE(SUM(CASE WHEN d.layer='MOZAIKA' THEN d.dH ELSE 0 END),0)) > 0
       THEN (COALESCE(SUM(CASE WHEN d.layer='MOZAIKA' THEN d.dH ELSE 0 END),0)::double precision) /
            (COALESCE(SUM(CASE WHEN d.layer='AST' THEN d.dS ELSE 0 END),0) +
             COALESCE(SUM(CASE WHEN d.layer='MOZAIKA' THEN d.dH ELSE 0 END),0))
       ELSE NULL END AS beta
FROM glx_deltas d
JOIN glx_grammar_events g ON d.ge_id = g.id
GROUP BY unit;

-- Start→(Done|Error) latency (ms)
CREATE OR REPLACE VIEW glx_path_run AS
SELECT s.id AS start_id, s.ts AS start_ts, e.id AS end_id, e.ts AS end_ts,
       EXTRACT(EPOCH FROM (e.ts - s.ts))*1000.0 AS ms
FROM glx_events s
JOIN glx_events e ON e.correlation_id = s.correlation_id
WHERE s.topic='core.run.start' AND e.topic IN ('core.run.done','core.run.error');

-- ΔH>0 bez realnego celu (heurystyka)
CREATE OR REPLACE VIEW glx_violations_empty_bridges AS
SELECT g.id, g.kind, g.subject, g.object, g.topic, g.ts
FROM glx_grammar_events g
JOIN glx_deltas d ON d.ge_id = g.id
LEFT JOIN glx_topics t ON t.topic = g.object
WHERE d.layer='MOZAIKA' AND d.dH > 0 AND (g.object IS NULL OR t.topic IS NULL);
