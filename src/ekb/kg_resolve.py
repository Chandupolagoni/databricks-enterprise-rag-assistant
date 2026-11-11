"""Entity resolution for the policy knowledge graph.

``normalize_name`` only removes formatting noise, so the same real-world entity still lands
on several nodes: one passage says "Anti-Money Laundering and KYC Policy", another "the AML
policy", a third "AML and KYC Policy"; regulations appear as "Bank Secrecy Act" and "BSA";
departments as "Compliance", "the Compliance department" and "Compliance team". Split nodes
break graph retrieval — a one-hop expansion from one alias misses every edge attached to the
others — and inflate the graph with near-duplicates nobody can audit.

Resolution is deliberately conservative, because a wrong merge silently mixes two policies'
obligations (worse than a missed merge, which only costs recall):

* names are compared on a *resolution key*: lowercase tokens with generic words ("policy",
  "SOP", "department", "the") and the firm's name removed, known acronyms expanded;
* an alias maps to a canonical name only within the same entity type, via (1) equal keys,
  (2) acronym = initials of the canonical name ("BSA" -> "Bank Secrecy Act"), (3) all alias
  tokens contained in exactly one candidate, or (4) a near-identical key (typos);
* ambiguous aliases ("risk policy" matches three risk documents) are left unmerged;
* candidates come from an optional catalog (document titles, business units) plus the
  extracted names themselves, ranked so an alias only ever maps to a higher-ranked name — no
  merge cycles, and chains collapse to one canonical root.

``resolve_graph`` rewrites node ids, remaps edge endpoints, drops duplicates and self loops a
merge creates, and returns alias rows for ``silver_kg_aliases`` so every merge is reviewable.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from .kg_schema import KGEdge, KGNode, node_id, normalize_name

GENERIC_WORDS = frozenset({
    "the", "a", "an", "our", "this", "policy", "policies", "sop", "kb", "procedure", "procedures",
    "standard", "department", "dept", "team", "unit", "function", "group", "desk", "meridian", "trust", "co",
})
CONNECTORS = frozenset({"and", "of", "for", "on", "to", "in"})

# fictional firm glossary; extend from a reference table in production
DEFAULT_ACRONYMS: dict[str, str] = {
    "aml": "anti money laundering", "kyc": "know your customer", "bsa": "bank secrecy act",
    "sar": "suspicious activity report", "nav": "net asset value", "bcp": "business continuity plan",
    "hr": "human resources", "pto": "paid time off", "genai": "generative ai", "it": "technology",
}


def resolution_key(name: str, acronyms: Mapping[str, str] = DEFAULT_ACRONYMS) -> tuple[str, ...]:
    """Significant tokens of a name in order, generic words removed, acronyms expanded."""
    tokens: list[str] = []
    for tok in normalize_name(name).split():
        tokens += acronyms.get(tok, tok).split()
    return tuple(t for t in tokens if t not in GENERIC_WORDS)


def _initials(name: str) -> str:
    return "".join(t[0] for t in normalize_name(name).split() if t not in GENERIC_WORDS | CONNECTORS)


@dataclass
class EntityResolver:
    """Resolve surface names to canonical names per entity type.

    ``catalog`` maps entity type -> authoritative names (e.g. Policy -> document titles);
    catalog names always outrank extracted ones, then fuller names outrank shorter aliases.
    """
    catalog: Mapping[str, Iterable[str]] = field(default_factory=dict)
    acronyms: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ACRONYMS))
    min_containment: float = 1.0
    fuzzy_ratio: float = 0.92

    def _key(self, name: str) -> tuple[str, ...]:
        return resolution_key(name, self.acronyms)

    def match(self, name: str, candidates: list[str]) -> tuple[str | None, str]:
        """Best candidate for ``name`` and the rule that matched (or ``None`` and why not).
        ``candidates`` are expected in preference order; ties between spellings of the same
        key go to the first one."""
        key = self._key(name)
        if not key:
            return None, "empty key"
        cands = [c for c in candidates if normalize_name(c) != normalize_name(name)]
        same = [c for c in cands if self._key(c) == key]
        if same:
            return same[0], "key"
        raw = normalize_name(name).replace(" ", "")
        if len(raw) >= 2 and " " not in normalize_name(name):
            acro = [c for c in cands if len(self._key(c)) >= 2 and _initials(c) == raw]
            if len(acro) == 1:
                return acro[0], "acronym"
        scored = []
        for c in cands:
            ckey = set(self._key(c))
            if ckey:
                scored.append((len(set(key) & ckey) / len(set(key)), c))
        best = max((s for s, _ in scored), default=0.0)
        if best >= self.min_containment:
            hits = [c for s, c in scored if s == best]
            if len({self._key(c) for c in hits}) == 1:  # several spellings of one entity: take the first
                return hits[0], "containment"
            return None, f"ambiguous ({len(hits)} candidates)"
        fuzzy = [c for c in cands
                 if difflib.SequenceMatcher(None, " ".join(key), " ".join(self._key(c))).ratio() >= self.fuzzy_ratio]
        if fuzzy and len({self._key(c) for c in fuzzy}) == 1:
            return fuzzy[0], "fuzzy"
        return None, "no match"

    def build_alias_map(self, typed_names: Iterable[tuple[str, str]]) -> dict[tuple[str, str], tuple[str, str]]:
        """Map (type, name) -> (canonical name, rule) for every name that merges into another."""
        counts = Counter(typed_names)
        by_type: dict[str, list[str]] = {}
        for (etype, nm) in counts:
            by_type.setdefault(etype, []).append(nm)
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for etype, names in by_type.items():
            catalog = list(dict.fromkeys(self.catalog.get(etype, [])))

            def rank(nm: str, catalog=catalog, etype=etype) -> tuple:
                return (nm in catalog, len(self._key(nm)), len(normalize_name(nm)), counts[(etype, nm)], nm)

            pool = sorted(set(names) | set(catalog), key=rank, reverse=True)
            parent: dict[str, tuple[str, str]] = {}
            for nm in names:
                higher = [c for c in pool if rank(c) > rank(nm)]
                target, rule = self.match(nm, higher)
                if target is not None:
                    parent[nm] = (target, rule)
            for nm, (target, rule) in parent.items():
                seen = {nm}
                while target in parent and target not in seen:  # collapse chains to the root
                    seen.add(target)
                    target = parent[target][0]
                out[(etype, nm)] = (target, rule)
        return out


@dataclass
class ResolutionResult:
    nodes: list[KGNode]
    edges: list[KGEdge]
    aliases: list[dict]
    stats: dict


def resolve_graph(nodes: Iterable[KGNode], edges: Iterable[KGEdge], resolver: EntityResolver) -> ResolutionResult:
    """Merge alias nodes into canonical ones and remap edges; returns alias rows for review."""
    nodes, edges = list(nodes), list(edges)
    alias_map = resolver.build_alias_map((n.entity_type, n.name) for n in nodes)
    id_map: dict[str, str] = {}
    alias_rows = []
    for (etype, nm), (canon, rule) in sorted(alias_map.items()):
        src, dst = node_id(etype, nm), node_id(etype, canon)
        if src != dst:
            id_map[src] = dst
            alias_rows.append({"alias_id": src, "alias_name": nm, "canonical_id": dst, "canonical_name": canon,
                               "entity_type": etype, "rule": rule})
    merged: dict[str, KGNode] = {}
    for n in nodes:
        canon = alias_map.get((n.entity_type, n.name), (n.name, ""))[0]
        merged.setdefault(node_id(n.entity_type, canon), replace(n, name=canon))
    out_edges: dict[str, KGEdge] = {}
    self_loops = 0
    for e in edges:
        e2 = replace(e, src_id=id_map.get(e.src_id, e.src_id), dst_id=id_map.get(e.dst_id, e.dst_id))
        if e2.src_id == e2.dst_id:
            self_loops += 1
            continue
        out_edges.setdefault(e2.edge_id, e2)
    distinct_in = len({n.node_id for n in nodes})
    stats = {"nodes_in": distinct_in, "nodes_out": len(merged), "aliases_merged": len(alias_rows),
             "edges_in": len(edges), "edges_out": len(out_edges), "self_loops_dropped": self_loops}
    return ResolutionResult(list(merged.values()), list(out_edges.values()), alias_rows, stats)


def catalog_from_documents(rows: Iterable[Mapping]) -> dict[str, list[str]]:
    """Authoritative names from document metadata: titles -> Policy, business units -> Department."""
    rows = list(rows)
    return {"Policy": [r["title"] for r in rows],
            "Department": sorted({re.sub(r"_", " ", r["business_unit"]).title() for r in rows})}
