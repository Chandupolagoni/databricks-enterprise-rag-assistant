"""Optional Neo4j export of the policy knowledge graph (Cypher ``MERGE`` upserts).

The Delta tables ``silver_kg_nodes`` / ``silver_kg_edges`` stay the source of truth (governed
by Unity Catalog, versioned, fact-validated). Neo4j is an optional read replica for analysts
who want Bloom/Browser exploration or multi-hop Cypher ("which transaction types end up
requiring the AML policy?"), so the export must be safe to re-run and must not let extracted
text become Cypher:

* every write is ``MERGE`` on the deterministic ``node_id`` (unique constraint per label) or
  on the (source, relation, target) triple, so re-running the job updates instead of
  duplicating;
* labels and relationship types cannot be Cypher parameters, so they are interpolated only
  after being checked against the fixed schema (``ENTITY_TYPES`` / ``RELATIONS``); all values
  — names, evidence spans with quotes or braces — travel as parameters (``UNWIND $rows``);
* rows are re-validated with ``validate_graph`` first, so an edge to a missing node or an
  off-schema relation never reaches Neo4j;
* each export stamps an ``export_id``; ``prune=True`` then removes nodes and relationships the
  current Delta snapshot no longer contains (e.g. facts from a superseded policy version).

``export_statements`` is driver-agnostic (list of ``(cypher, params)``); ``run_export`` sends
them in one write transaction with the official ``neo4j`` driver; ``to_cypher_script``
renders a self-contained ``.cypher`` file for ``cypher-shell`` with escaped literals.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .kg_schema import ENTITY_TYPES, RELATIONS, KGEdge, KGNode, entity_type_of, validate_graph


def _label(entity_type: str) -> str:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"refusing to export unknown label {entity_type!r}")
    return entity_type


def _rel_type(relation: str) -> str:
    if relation not in RELATIONS:
        raise ValueError(f"refusing to export unknown relationship type {relation!r}")
    return relation


def constraint_statements() -> list[str]:
    return [f"CREATE CONSTRAINT kg_{t.lower()}_node_id IF NOT EXISTS FOR (n:{t}) REQUIRE n.node_id IS UNIQUE"
            for t in ENTITY_TYPES]


def _batches(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def export_statements(nodes: Iterable[KGNode], edges: Iterable[KGEdge], export_id: str, batch_size: int = 500,
                      prune: bool = False) -> list[tuple[str, dict]]:
    """Ordered ``(cypher, params)`` statements: constraints, node upserts, edge upserts, prune."""
    g = validate_graph(nodes, edges)
    if g.rejected:
        raise ValueError(f"{len(g.rejected)} rows violate the graph schema; first: {g.rejected[0][1]}")
    stmts: list[tuple[str, dict]] = [(c, {}) for c in constraint_statements()]
    by_label: dict[str, list[dict]] = {}
    for n in g.nodes:
        by_label.setdefault(_label(n.entity_type), []).append(
            {"node_id": n.node_id, "props": {"name": n.name, "doc_id": n.doc_id, "version": n.version,
                                             "evidence": n.evidence, "export_id": export_id}})
    for label, rows in sorted(by_label.items()):
        for batch in _batches(rows, batch_size):
            stmts.append((f"UNWIND $rows AS row MERGE (n:{label} {{node_id: row.node_id}}) SET n += row.props",
                          {"rows": batch}))
    by_rel: dict[tuple[str, str, str], list[dict]] = {}
    for e in g.edges:
        key = (_label(entity_type_of(e.src_id)), _rel_type(e.relation), _label(entity_type_of(e.dst_id)))
        by_rel.setdefault(key, []).append(
            {"src_id": e.src_id, "dst_id": e.dst_id,
             "props": {"edge_id": e.edge_id, "doc_id": e.doc_id, "version": e.version, "evidence": e.evidence,
                       "export_id": export_id}})
    for (src, rel, dst), rows in sorted(by_rel.items()):
        for batch in _batches(rows, batch_size):
            stmts.append((f"UNWIND $rows AS row MATCH (a:{src} {{node_id: row.src_id}}) "
                          f"MATCH (b:{dst} {{node_id: row.dst_id}}) MERGE (a)-[r:{rel}]->(b) SET r += row.props",
                          {"rows": batch}))
    if prune:
        rels = "|".join(RELATIONS)
        labels = " OR ".join(f"n:{t}" for t in ENTITY_TYPES)
        stmts += [(f"MATCH ()-[r:{rels}]->() WHERE r.export_id <> $export_id DELETE r", {"export_id": export_id}),
                  (f"MATCH (n) WHERE ({labels}) AND n.export_id <> $export_id DETACH DELETE n",
                   {"export_id": export_id})]
    return stmts


def run_export(driver, nodes: Iterable[KGNode], edges: Iterable[KGEdge], export_id: str, database: str | None = None,
               prune: bool = False) -> dict:
    """Run the export with a ``neo4j.Driver``; schema DDL first, then data in one write transaction."""
    stmts = export_statements(nodes, edges, export_id, prune=prune)
    ddl = [s for s in stmts if s[0].startswith("CREATE CONSTRAINT")]
    data = [s for s in stmts if not s[0].startswith("CREATE CONSTRAINT")]
    with driver.session(database=database) as session:
        for cypher, _ in ddl:                       # schema changes cannot share a transaction with writes
            session.run(cypher)

        def _write(tx):
            for cypher, params in data:
                tx.run(cypher, **params)

        session.execute_write(_write)
    return {"statements": len(stmts), "export_id": export_id,
            "nodes": sum(len(p["rows"]) for c, p in data if "MERGE (n:" in c),
            "edges": sum(len(p["rows"]) for c, p in data if "MERGE (a)-" in c)}


def cypher_literal(value) -> str:
    """Cypher literal for scalars / maps / lists (strings single-quoted and escaped)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"`{k}`: {cypher_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cypher_literal(v) for v in value) + "]"
    s = str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    return f"'{s}'"


def to_cypher_script(nodes: Iterable[KGNode], edges: Iterable[KGEdge], export_id: str, prune: bool = False) -> str:
    """Self-contained script for ``cypher-shell -f``: parameters inlined as escaped literals."""
    out = []
    for cypher, params in export_statements(nodes, edges, export_id, prune=prune):
        # single pass over the template, so text inside an inserted literal is never re-substituted
        out.append(re.sub(r"\$(\w+)", lambda m, p=params: cypher_literal(p[m.group(1)]), cypher) + ";")
    return "\n".join(out) + "\n"
