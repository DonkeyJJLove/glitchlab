from __future__ import annotations
"""
EGDB (Event Grammar DB) adapter for Postgres.

Usage:
  export GLX_PG_DSN="postgresql://user:pass@host:5432/glitchlab"
  from analysis.grammar.egdb_store import EGDB
  eg = EGDB()           # reads DSN from env
  eg.ensure_schema()    # create tables & indexes
  eg.ensure_views(".glx/grammar/views.sql")

This module intentionally focuses on DDL + simple inserts/queries for Stage A.
"""
import os, json, typing as t, datetime as dt
from dataclasses import dataclass
try:
    import psycopg  # psycopg v3
except Exception as e:  # pragma: no cover
    psycopg = None

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

def _parse_ts(ts: t.Union[str, dt.datetime]) -> dt.datetime:
    if isinstance(ts, dt.datetime): return ts
    try:
        if isinstance(ts, str) and ts.endswith("Z"):
            ts2 = ts[:-1]
            if "." not in ts2: ts2 = ts2 + ".000000"
            return dt.datetime.fromisoformat(ts2).replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(ts).astimezone(dt.timezone.utc)
    except Exception:
        return dt.datetime.now(dt.timezone.utc)

@dataclass
class EGDB:
    dsn: str | None = None

    def __post_init__(self):
        if self.dsn is None:
            self.dsn = os.environ.get("GLX_PG_DSN","postgresql://localhost/glitchlab")
        if psycopg is None:
            raise RuntimeError("psycopg is required for Postgres EGDB")

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    # ---------- DDL ----------
    def ensure_schema(self):
        ddl = '''
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
        CREATE TABLE IF NOT EXISTS glx_events (
          id TEXT PRIMARY KEY, ts TIMESTAMPTZ NOT NULL, plane TEXT, topic TEXT REFERENCES glx_topics(topic),
          source TEXT, schema_uri TEXT, correlation_id TEXT, causation_id TEXT,
          payload JSONB, tags JSONB, fallback_plan TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ev_topic_ts ON glx_events(topic, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_ev_payload ON glx_events USING GIN (payload jsonb_path_ops);

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

        CREATE TABLE IF NOT EXISTS glx_files (
          id SERIAL PRIMARY KEY, module TEXT, path TEXT UNIQUE, sha TEXT, extracted_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS glx_tags  (
          id SERIAL PRIMARY KEY, file_id INT REFERENCES glx_files(id) ON DELETE CASCADE,
          line INT, key TEXT, value TEXT, raw TEXT
        );
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
    def upsert_topic(self, topic: str, plane: str, schema_uri: str, version: int|None=None):
        q = """
        INSERT INTO glx_topics(topic, plane, schema_uri, version)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (topic) DO UPDATE SET plane=EXCLUDED.plane, schema_uri=EXCLUDED.schema_uri,
                                         version=COALESCE(EXCLUDED.version, glx_topics.version);
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, (topic, plane, schema_uri, version))

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
        evt["ts"] = _parse_ts(evt.get("ts"))
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, evt)

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
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, ge)
                ge_id = cur.fetchone()[0]
        return ge_id

    def insert_delta(self, ge_id: int, layer: str, dS: float, dH: float, dZ: float, weights: dict|None=None) -> int:
        q = """
        INSERT INTO glx_deltas(ge_id, layer, dS, dH, dZ, weights)
        VALUES (%s,%s,%s,%s,%s,%s) RETURNING id;
        """
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(q, (ge_id, layer, dS, dH, dZ, json.dumps(weights or {})))
                return cur.fetchone()[0]

    # ---------- Queries ----------
    def query(self, sql: str, params: tuple|None=None) -> list[dict]:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(sql, params or ())
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
