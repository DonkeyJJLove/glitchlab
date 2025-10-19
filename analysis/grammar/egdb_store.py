from __future__ import annotations
"""
EGDB (Event Grammar DB) adapter for Postgres + snapshoty artefaktów (projektowy graf, metryki, metasoczewki).

Usage:
  export GLX_PG_DSN="postgresql://user:pass@host:5432/glitchlab"
  from glitchlab.analysis.grammar.egdb_store import EGDB
  eg = EGDB()           # reads DSN from env
  eg.ensure_schema()    # create tables & indexes (including snapshots)
  eg.ensure_views(".glx/grammar/views.sql")

Zależności: psycopg v3 (jeśli brak – moduł rzuci RuntimeError).
"""

import os, json, typing as t, datetime as dt
from dataclasses import dataclass
from uuid import uuid4

try:
    import psycopg  # psycopg v3
except Exception as e:  # pragma: no cover
    psycopg = None

# Kontrakty topiców BUS (nazwy + wersjonowanie)
try:
    from glitchlab.analysis.grammar.events import (
        EVENTS_SCHEMA_VERSION,
        TOPIC_ANALYTICS_DELTA_READY,
        TOPIC_ANALYTICS_INVARIANTS_VIOLATION,
        TOPIC_SCOPE_METRICS_UPDATED,
        TOPIC_SCOPE_META_READY,
    )
except Exception:  # pragma: no cover
    # fallback – gdy import modułu events nie jest dostępny w środowisku
    EVENTS_SCHEMA_VERSION = "v1"
    TOPIC_ANALYTICS_DELTA_READY = "analytics.delta.ready"
    TOPIC_ANALYTICS_INVARIANTS_VIOLATION = "analytics.invariants.violation"
    TOPIC_SCOPE_METRICS_UPDATED = "analytics.scope.metrics.updated"
    TOPIC_SCOPE_META_READY = "analytics.scope.meta.ready"

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _now_iso() -> str:
    return _now_utc().strftime(ISO)


def _parse_ts(ts: t.Union[str, dt.datetime]) -> dt.datetime:
    if isinstance(ts, dt.datetime):
        return ts
    try:
        if isinstance(ts, str) and ts.endswith("Z"):
            ts2 = ts[:-1]
            if "." not in ts2:
                ts2 = ts2 + ".000000"
            return dt.datetime.fromisoformat(ts2).replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(ts).astimezone(dt.timezone.utc)
    except Exception:
        return _now_utc()


def _is_mapping(x: t.Any) -> bool:
    from collections.abc import Mapping
    return isinstance(x, Mapping)


def _graph_counts(graph: t.Mapping[str, t.Any]) -> tuple[int, int]:
    """Szacuje (n_nodes, n_edges) dla JSON-owego grafu {nodes: list|dict, edges: list}."""
    n_nodes = 0
    n_edges = 0
    if not _is_mapping(graph):
        return (0, 0)
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if isinstance(nodes, list):
        n_nodes = len(nodes)
    elif _is_mapping(nodes):
        n_nodes = len(nodes)
    if isinstance(edges, list):
        n_edges = len(edges)
    return (int(n_nodes), int(n_edges))


@dataclass
class EGDB:
    dsn: str | None = None

    def __post_init__(self):
        if self.dsn is None:
            self.dsn = os.environ.get("GLX_PG_DSN", "postgresql://localhost/glitchlab")
        if psycopg is None:
            raise RuntimeError("psycopg is required for Postgres EGDB")

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    # ---------- DDL ----------
    def ensure_schema(self):
        """
        Tworzy tabele bazowe + snapshoty artefaktów (project_graph, graph_metrics, meta_lens).
        """
        ddl = '''
        -- usługi / topiki / węzły
        CREATE TABLE IF NOT EXISTS glx_services (
          id SERIAL PRIMARY KEY, name TEXT UNIQUE, version TEXT, caps JSONB, announced_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS glx_topics (
          topic TEXT PRIMARY KEY, plane TEXT CHECK (plane IN ('event','data')),
          schema_uri TEXT, version INT
        );
        CREATE TABLE IF NOT EXISTS glx_nodes (
          id BIGSERIAL PRIMARY KEY, service TEXT, component TEXT, ast_path TEXT,
          tile TEXT, bucket INT, caps JSONB, tags JSONB,
          UNIQUE(service, component, ast_path)
        );

        -- zdarzenia (BUS)
        CREATE TABLE IF NOT EXISTS glx_events (
          id TEXT PRIMARY KEY, ts TIMESTAMPTZ NOT NULL, plane TEXT, topic TEXT REFERENCES glx_topics(topic),
          source TEXT, schema_uri TEXT, correlation_id TEXT, causation_id TEXT,
          payload JSONB, tags JSONB, fallback_plan TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ev_topic_ts ON glx_events(topic, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_ev_payload ON glx_events USING GIN (payload jsonb_path_ops);

        -- grammar events + delty (istniejące)
        CREATE TABLE IF NOT EXISTS glx_grammar_events (
          id BIGSERIAL PRIMARY KEY, origin TEXT, kind TEXT, topic TEXT,
          subject TEXT, object TEXT, module TEXT, file_path TEXT, line INT,
          ts TIMESTAMPTZ, meta JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_ge_kind_ts ON glx_grammar_events(kind, ts DESC);

        CREATE TABLE IF NOT EXISTS glx_deltas (
          id BIGSERIAL PRIMARY KEY, ge_id BIGINT REFERENCES glx_grammar_events(id) ON DELETE CASCADE,
          layer TEXT CHECK (layer IN ('AST','MOZAIKA')),
          dS DOUBLE PRECISION, dH DOUBLE PRECISION, dZ DOUBLE PRECISION, weights JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_deltas_layer ON glx_deltas(layer);

        -- pliki i tagi (istniejące)
        CREATE TABLE IF NOT EXISTS glx_files (
          id SERIAL PRIMARY KEY, module TEXT, path TEXT UNIQUE, sha TEXT, extracted_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS glx_tags  (
          id SERIAL PRIMARY KEY, file_id INT REFERENCES glx_files(id) ON DELETE CASCADE,
          line INT, key TEXT, value TEXT, raw TEXT
        );

        -- NOWE: snapshoty artefaktów (projektowy graf, metryki, metasoczewki)
        CREATE TABLE IF NOT EXISTS glx_artifacts (
          id BIGSERIAL PRIMARY KEY,
          kind TEXT CHECK (kind IN ('project_graph','graph_metrics','meta_lens')) NOT NULL,
          hash TEXT,                 -- np. hash grafu (dla graph_metrics), bądź hash pełnego artefaktu
          level TEXT,                -- meta_lens: poziom (project/module/file/func/bus/...)
          name  TEXT,                -- meta_lens: nazwa (np. 'glitchlab.core' / 'auto')
          paths JSONB,               -- ścieżki do plików w .glx (json/dot/metrics)
          header JSONB,              -- nagłówki / metadane (np. counts, wersje, etc.)
          payload JSONB,             -- minimalny snapshot (np. skrócone metryki, top-N)
          created_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_kind_ts  ON glx_artifacts(kind, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_artifacts_hash     ON glx_artifacts(hash);
        '''
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(ddl)

    def ensure_views(self, views_sql_path: str):
        with open(views_sql_path, "r", encoding="utf-8") as f:
            sql = f.read()
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(sql)

    # ---------- Topics ----------
    def upsert_topic(self, topic: str, plane: str, schema_uri: str, version: int | None = None):
        q = """
        INSERT INTO glx_topics(topic, plane, schema_uri, version)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (topic) DO UPDATE SET plane=EXCLUDED.plane, schema_uri=EXCLUDED.schema_uri,
                                         version=COALESCE(EXCLUDED.version, glx_topics.version);
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, (topic, plane, schema_uri, version))

    def upsert_default_topics(self):
        """Rejestruje podstawowe topiki BUS (event-plane) dla analityki."""
        schema_uri = f"glitchlab://events/{EVENTS_SCHEMA_VERSION}"
        for tpc in (
            TOPIC_ANALYTICS_DELTA_READY,
            TOPIC_ANALYTICS_INVARIANTS_VIOLATION,
            TOPIC_SCOPE_METRICS_UPDATED,
            TOPIC_SCOPE_META_READY,
        ):
            self.upsert_topic(tpc, plane="event", schema_uri=schema_uri, version=int(EVENTS_SCHEMA_VERSION.strip("v") or "1"))

    # ---------- Events ----------
    def insert_event(self, evt: dict):
        q = """
        INSERT INTO glx_events(id, ts, plane, topic, source, schema_uri, correlation_id, causation_id,
                               payload, tags, fallback_plan)
        VALUES (%(id)s, %(ts)s, %(plane)s, %(topic)s, %(source)s, %(schema_uri)s, %(correlation_id)s,
                %(causation_id)s, %(payload)s, %(tags)s, %(fallback_plan)s)
        ON CONFLICT (id) DO NOTHING;
        """
        evt = evt.copy()
        evt.setdefault("id", uuid4().hex)
        evt["ts"] = _parse_ts(evt.get("ts", _now_iso()))
        # upewnij się, że JSONB dostaje str (dla pewności zgodności sterownika)
        if "payload" in evt and _is_mapping(evt["payload"]):
            evt["payload"] = json.dumps(evt["payload"])
        if "tags" in evt and _is_mapping(evt["tags"]):
            evt["tags"] = json.dumps(evt["tags"])
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, evt)

    # Wygodne wrappery do publikacji istotnych zdarzeń
    def publish_scope_metrics_updated(self, *, metrics_path: str, graph_hash: str, metrics_hash: str,
                                      summary: dict | None = None, source: str | None = None):
        payload = {
            "kind": "graph_metrics",
            "paths": {"metrics": metrics_path},
            "meta": {"graph_hash": graph_hash, "metrics_hash": metrics_hash, "ts": _now_iso(), "version": EVENTS_SCHEMA_VERSION},
            "summary": summary or {},
        }
        self.insert_event({
            "plane": "event",
            "topic": TOPIC_SCOPE_METRICS_UPDATED,
            "source": source or "analysis.graph_metrics",
            "schema_uri": f"glitchlab://events/{EVENTS_SCHEMA_VERSION}",
            "payload": payload,
            "tags": {},
            "fallback_plan": None,
        })

    def publish_scope_meta_ready(self, *, level: str, name: str, path_json: str, path_dot: str | None,
                                 anchors_count: int | None, window_metrics: dict | None, source: str | None = None):
        payload = {
            "kind": "meta_lens",
            "level": level,
            "name": name,
            "paths": {"json": path_json, **({"dot": path_dot} if path_dot else {})},
            "anchors_count": int(anchors_count or 0),
            "window_metrics": window_metrics or {},
            "meta": {"ts": _now_iso(), "version": EVENTS_SCHEMA_VERSION},
        }
        self.insert_event({
            "plane": "event",
            "topic": TOPIC_SCOPE_META_READY,
            "source": source or "analysis.scope_meta",
            "schema_uri": f"glitchlab://events/{EVENTS_SCHEMA_VERSION}",
            "payload": payload,
            "tags": {},
            "fallback_plan": None,
        })

    # ---------- Grammar ----------
    def insert_grammar_event(self, ge: dict) -> int:
        q = """
        INSERT INTO glx_grammar_events(origin, kind, topic, subject, object, module, file_path, line, ts, meta)
        VALUES (%(origin)s,%(kind)s,%(topic)s,%(subject)s,%(object)s,%(module)s,%(file_path)s,%(line)s,%(ts)s,%(meta)s)
        RETURNING id;
        """
        ge = ge.copy()
        ts = ge.get("ts")
        ge["ts"] = _parse_ts(ts) if ts else None
        if "meta" in ge and _is_mapping(ge["meta"]):
            ge["meta"] = json.dumps(ge["meta"])
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, ge)
                ge_id = cur.fetchone()[0]
        return ge_id

    def insert_delta(self, ge_id: int, layer: str, dS: float, dH: float, dZ: float, weights: dict | None = None) -> int:
        q = """
        INSERT INTO glx_deltas(ge_id, layer, dS, dH, dZ, weights)
        VALUES (%s,%s,%s,%s,%s,%s) RETURNING id;
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, (ge_id, layer, dS, dH, dZ, json.dumps(weights or {})))
                return cur.fetchone()[0]

    # ---------- Snapshoty artefaktów ----------
    def insert_project_graph_snapshot(self, *,
                                      graph: dict,
                                      graph_hash: str | None = None,
                                      paths: dict | None = None,
                                      header: dict | None = None,
                                      payload: dict | None = None) -> int:
        """
        Zapisuje snapshot projektowego grafu.
        - header: minimalne metadane (uzupełniane automatycznie counts/version/ts).
        - payload: opcjonalny skrótowy payload (np. top-moduły, rozkłady stopni).
        """
        n_nodes, n_edges = _graph_counts(graph)
        hdr = {
            "schema_version": EVENTS_SCHEMA_VERSION,
            "ts": _now_iso(),
            "nodes": n_nodes,
            "edges": n_edges,
        }
        if header:
            hdr.update(header)

        q = """
        INSERT INTO glx_artifacts(kind, hash, level, name, paths, header, payload)
        VALUES ('project_graph', %s, NULL, NULL, %s, %s, %s)
        RETURNING id;
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    q,
                    (
                        graph_hash,
                        json.dumps(paths or {}),
                        json.dumps(hdr),
                        json.dumps(payload or {}),
                    ),
                )
                return int(cur.fetchone()[0])

    def insert_graph_metrics_snapshot(self, *,
                                      metrics: dict,
                                      graph_hash: str,
                                      metrics_hash: str | None = None,
                                      paths: dict | None = None,
                                      header: dict | None = None,
                                      payload: dict | None = None) -> int:
        """
        Zapisuje snapshot globalnych metryk grafowych. Kluczowe hashe:
          - graph_hash: hash grafu, na którym liczono metryki,
          - metrics_hash: hash samego pliku metryk (opcjonalny, ale zalecany).
        """
        hdr = {
            "schema_version": EVENTS_SCHEMA_VERSION,
            "ts": _now_iso(),
            "graph_hash": graph_hash,
            "metrics_hash": metrics_hash,
            "metrics_list": sorted(list(metrics.keys())) if _is_mapping(metrics) else [],
        }
        if header:
            hdr.update(header)

        q = """
        INSERT INTO glx_artifacts(kind, hash, level, name, paths, header, payload)
        VALUES ('graph_metrics', %s, NULL, NULL, %s, %s, %s)
        RETURNING id;
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    q,
                    (
                        graph_hash,
                        json.dumps(paths or {}),
                        json.dumps(hdr),
                        json.dumps(payload or {}),
                    ),
                )
                return int(cur.fetchone()[0])

    def insert_meta_lens_snapshot(self, *,
                                  level: str,
                                  name: str,
                                  paths: dict,
                                  anchors_count: int | None = None,
                                  window_metrics: dict | None = None,
                                  header: dict | None = None,
                                  payload: dict | None = None) -> int:
        """
        Zapisuje snapshot metasoczewki (nagłówek + minimalny payload).
        level: "project"|"module"|"file"|"func"|"bus"|...
        name:  np. "glitchlab.core" albo "auto"
        """
        hdr = {
            "schema_version": EVENTS_SCHEMA_VERSION,
            "ts": _now_iso(),
            "level": level,
            "name": name,
            "anchors_count": int(anchors_count or 0),
            "window_metrics": window_metrics or {},
        }
        if header:
            hdr.update(header)

        q = """
        INSERT INTO glx_artifacts(kind, hash, level, name, paths, header, payload)
        VALUES ('meta_lens', NULL, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    q,
                    (
                        level,
                        name,
                        json.dumps(paths or {}),
                        json.dumps(hdr),
                        json.dumps(payload or {}),
                    ),
                )
                return int(cur.fetchone()[0])

    # ---------- Queries ----------
    def query(self, sql: str, params: tuple | None = None) -> list[dict]:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(sql, params or ())
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
