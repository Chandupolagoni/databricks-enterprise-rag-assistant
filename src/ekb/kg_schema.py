"""Fixed schema for the policy knowledge graph (GraphRAG).

Vector and keyword retrieval find passages that *look like* the question; they cannot answer
"which policies does SR 11-7 govern?" or "what else must I follow before processing a wire?",
because those answers live in relationships spread across documents. A knowledge graph holds
those relationships — but only if extraction is constrained. An LLM asked to "extract
entities" invents labels ("Rule", "Law", "Regulator", "Team") and relations ("is subject to",
"falls under"), and a free-form graph cannot be validated, queried reliably or audited.

So the graph has a closed vocabulary:

* entity types  ``Policy``, ``Regulation``, ``Department``, ``EmployeeRole``, ``TransactionType``
* relations     ``GOVERNED_BY``, ``APPLIES_TO``, ``OWNED_BY``, ``REQUIRES``

each relation with an explicit set of allowed (source type, target type) pairs. Every node and
edge carries the source ``doc_id``, ``version`` and an evidence span, so any graph-derived
answer can be traced back to the sentence that justifies it. ``validate_graph`` rejects
anything outside the schema *before* it is written to ``silver_kg_nodes`` / ``silver_kg_edges``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

ENTITY_TYPES: tuple[str, ...] = ("Policy", "Regulation", "Department", "EmployeeRole", "TransactionType")

# relation -> allowed (source entity type, target entity type) pairs
RELATIONS: dict[str, frozenset[tuple[str, str]]] = {
    # a policy is governed by an external regulation or by a parent internal policy
    "GOVERNED_BY": frozenset({("Policy", "Regulation"), ("Policy", "Policy")}),
    # who / what a policy covers
    "APPLIES_TO": frozenset({("Policy", "Department"), ("Policy", "EmployeeRole"),
                             ("Policy", "TransactionType")}),
    # accountable owner of a policy
    "OWNED_BY": frozenset({("Policy", "Department")}),
    # a policy builds on another policy; a transaction type triggers a control policy
    "REQUIRES": frozenset({("Policy", "Policy"), ("TransactionType", "Policy")}),
}

RELATION_DESCRIPTIONS: dict[str, str] = {
    "GOVERNED_BY": "the policy implements or is constrained by the target regulation or parent policy",
    "APPLIES_TO": "the policy covers the target department, employee role or transaction type",
    "OWNED_BY": "the target department is accountable for maintaining the policy",
    "REQUIRES": "the source cannot be carried out without complying with the target policy",
}

MAX_EVIDENCE_CHARS = 500


def normalize_name(name: str) -> str:
    """Canonical surface form: lowercase, punctuation stripped, whitespace collapsed.
    Full alias merging (e.g. "AML policy" vs "Anti-Money Laundering and KYC Policy") is
    entity resolution and happens in a later step; this only removes formatting noise."""
    name = name.lower().replace("&", " and ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def node_id(entity_type: str, name: str) -> str:
    """Deterministic id, e.g. ``policy:investment_suitability_policy``; stable across runs so
    MERGE upserts on the Delta tables are idempotent."""
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type {entity_type!r}")
    return f"{entity_type.lower()}:{normalize_name(name).replace(' ', '_')}"


@dataclass(frozen=True)
class KGNode:
    entity_type: str
    name: str
    doc_id: str
    version: int
    evidence: str

    @property
    def node_id(self) -> str:
        return node_id(self.entity_type, self.name)

    def as_row(self) -> dict:
        return dict(asdict(self), node_id=self.node_id)


@dataclass(frozen=True)
class KGEdge:
    src_id: str
    relation: str
    dst_id: str
    doc_id: str
    version: int
    evidence: str

    @property
    def edge_id(self) -> str:
        return f"{self.src_id}|{self.relation}|{self.dst_id}"

    def as_row(self) -> dict:
        return dict(asdict(self), edge_id=self.edge_id)


def entity_type_of(nid: str) -> str | None:
    prefix = nid.split(":", 1)[0]
    return next((t for t in ENTITY_TYPES if t.lower() == prefix), None)


def validate_node(node: KGNode) -> list[str]:
    errors = []
    if node.entity_type not in ENTITY_TYPES:
        errors.append(f"unknown entity type {node.entity_type!r}")
    if not normalize_name(node.name):
        errors.append("empty name")
    errors += _provenance_errors(node.doc_id, node.evidence)
    return errors


def validate_edge(edge: KGEdge, node_ids: set[str] | None = None) -> list[str]:
    """Reasons the edge violates the schema (empty list = valid). ``node_ids`` optionally
    checks that both endpoints exist, so no edge points at a node that was never written."""
    errors = []
    allowed = RELATIONS.get(edge.relation)
    src_t, dst_t = entity_type_of(edge.src_id), entity_type_of(edge.dst_id)
    if allowed is None:
        errors.append(f"unknown relation {edge.relation!r}")
    elif (src_t, dst_t) not in allowed:
        errors.append(f"{edge.relation} not allowed from {src_t} to {dst_t}")
    if edge.src_id == edge.dst_id:
        errors.append("self loop")
    if node_ids is not None:
        errors += [f"missing node {n}" for n in (edge.src_id, edge.dst_id) if n not in node_ids]
    errors += _provenance_errors(edge.doc_id, edge.evidence)
    return errors


def _provenance_errors(doc_id: str, evidence: str) -> list[str]:
    errors = []
    if not doc_id:
        errors.append("missing doc_id")
    if not evidence or not evidence.strip():
        errors.append("missing evidence span")
    elif len(evidence) > MAX_EVIDENCE_CHARS:
        errors.append("evidence span too long")
    return errors


@dataclass
class GraphValidation:
    nodes: list[KGNode]
    edges: list[KGEdge]
    rejected: list[tuple[object, list[str]]]

    @property
    def reject_rate(self) -> float:
        total = len(self.nodes) + len(self.edges) + len(self.rejected)
        return round(len(self.rejected) / total, 4) if total else 0.0


def validate_graph(nodes: Iterable[KGNode], edges: Iterable[KGEdge]) -> GraphValidation:
    """Split extracted nodes/edges into schema-valid ones and rejects (with reasons).
    Duplicate nodes (same ``node_id``) keep the first occurrence; duplicate edges likewise."""
    kept_nodes: dict[str, KGNode] = {}
    rejected: list[tuple[object, list[str]]] = []
    for n in nodes:
        errs = validate_node(n)
        if errs:
            rejected.append((n, errs))
        else:
            kept_nodes.setdefault(n.node_id, n)
    kept_edges: dict[str, KGEdge] = {}
    for e in edges:
        errs = validate_edge(e, set(kept_nodes))
        if errs:
            rejected.append((e, errs))
        else:
            kept_edges.setdefault(e.edge_id, e)
    return GraphValidation(list(kept_nodes.values()), list(kept_edges.values()), rejected)


def schema_prompt() -> str:
    """Schema description to embed in the LLM extraction prompt."""
    rels = "\n".join(
        f"- {r}: {RELATION_DESCRIPTIONS[r]}. Allowed: "
        + ", ".join(f"{s} -> {d}" for s, d in sorted(pairs))
        for r, pairs in RELATIONS.items())
    return (f"Entity types (use exactly these): {', '.join(ENTITY_TYPES)}.\n"
            f"Relations (use exactly these, in this direction):\n{rels}\n"
            "Every entity and relation must quote a verbatim evidence span from the passage.")


def kg_table_ddl(nodes_table: str, edges_table: str) -> list[str]:
    """Delta DDL for the graph tables (Unity Catalog, three-part names)."""
    types = ", ".join(f"'{t}'" for t in ENTITY_TYPES)
    rels = ", ".join(f"'{r}'" for r in RELATIONS)
    return [
        f"""CREATE TABLE IF NOT EXISTS {nodes_table} (
  node_id STRING NOT NULL, entity_type STRING NOT NULL, name STRING NOT NULL,
  doc_id STRING NOT NULL, version INT NOT NULL, evidence STRING NOT NULL,
  extracted_at TIMESTAMP,
  CONSTRAINT entity_type_valid CHECK (entity_type IN ({types}))
) COMMENT 'Policy knowledge graph nodes (fixed schema, see ekb.kg_schema)'""",
        f"""CREATE TABLE IF NOT EXISTS {edges_table} (
  edge_id STRING NOT NULL, src_id STRING NOT NULL, relation STRING NOT NULL, dst_id STRING NOT NULL,
  doc_id STRING NOT NULL, version INT NOT NULL, evidence STRING NOT NULL,
  extracted_at TIMESTAMP,
  CONSTRAINT relation_valid CHECK (relation IN ({rels}))
) COMMENT 'Policy knowledge graph edges with source evidence (fixed schema, see ekb.kg_schema)'""",
    ]
