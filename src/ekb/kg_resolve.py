"""Entity resolution for the policy knowledge graph.

``node_id`` only strips formatting noise, so the LLM's different surface forms of one entity
become different nodes: "Anti-Money Laundering and KYC Policy", "AML & KYC Policy" and
"AML policy" are three policies, "Bank Secrecy Act" and "BSA" two regulations, "Compliance"
and "Compliance Department" two departments. Relations then split across duplicates and a
one-hop graph expansion from the real policy misses most of its neighbours.

``EntityResolver`` maps every node onto one canonical entity, per entity type, using rules
that are deterministic and auditable (each merge records the rule that fired):

1. ``alias``     an explicit alias table (curated by the policy owners);
2. ``signature`` same core tokens after dropping stop words, generic type words ("policy",
   "SOP", "department", ...) and plural ``s`` — "Compliance Department" == "compliance";
3. ``acronym``   a one-token name equal to the acronym of a known entity ("BSA", "OFAC");
4. ``catalog``   (policies only) every core token of the name is contained in exactly one
   catalogued document title, where a token may also be an acronym of consecutive title
   words — "AML policy" and "Wire Transfer SOP" resolve to their document titles.

Rule 4 is deliberately limited to the document catalog and to unique matches: open-ended
fuzzy matching would merge "Operations" into "Payments Operations", which is a worse error
for an audit-facing graph than a missed merge. Edges are rewritten to the canonical ids,
self loops created by a merge are dropped and duplicates collapse via ``validate_graph``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from .kg_schema import KGEdge, KGNode, node_id, normalize_name, validate_graph

STOP_WORDS = frozenset({"a", "an", "and", "the", "of", "for", "on", "to", "in", "our"})
GENERIC_WORDS: dict[str, frozenset[str]] = {
    "Policy": frozenset({"policy", "policies", "sop", "kb", "manual", "guideline", "guidelines",
                         "procedure", "procedures", "standard", "document", "meridian", "trust", "co"}),
    "Department": frozenset({"department", "dept", "team", "group", "unit", "division", "function"}),
    "Regulation": frozenset({"regulation", "rule"}),
    "EmployeeRole": frozenset({"staff", "employee"}),
    "TransactionType": frozenset({"transaction", "transactions"}),
}
MAX_ACRONYM_RUN = 5


def _singular(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def core_tokens(entity_type: str, name: str) -> list[str]:
    """Ordered content tokens of a name, without stop words or generic type words."""
    generic = GENERIC_WORDS.get(entity_type, frozenset())
    toks = [_singular(t) for t in normalize_name(name).split() if t not in STOP_WORDS]
    core = [t for t in toks if t not in generic]
    return core or toks  # a name made only of generic words keeps them ("Policy" stays non-empty)


def signature(entity_type: str, name: str) -> str:
    return " ".join(sorted(set(core_tokens(entity_type, name))))


def acronym(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens) if len(tokens) > 1 else ""


def expanded_tokens(tokens: list[str]) -> set[str]:
    """Tokens plus the acronyms of every run of 2..5 consecutive tokens."""
    out = set(tokens)
    for i in range(len(tokens)):
        for j in range(i + 2, min(len(tokens), i + MAX_ACRONYM_RUN) + 1):
            out.add(acronym(tokens[i:j]))
    return out


@dataclass(frozen=True)
class AliasLink:
    entity_type: str
    alias_name: str
    alias_id: str
    canonical_name: str
    canonical_id: str
    method: str

    def as_row(self) -> dict:
        return {"entity_type": self.entity_type, "alias_name": self.alias_name, "alias_id": self.alias_id,
                "canonical_name": self.canonical_name, "canonical_id": self.canonical_id, "method": self.method}


@dataclass
class ResolutionResult:
    nodes: list[KGNode]
    edges: list[KGEdge]
    aliases: list[AliasLink]
    rejected: list[tuple[object, list[str]]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def alias_rows(self) -> list[dict]:
        return [a.as_row() for a in self.aliases]


class EntityResolver:
    """Resolve node names to canonical entities.

    ``catalog`` maps an entity type to its known canonical names (e.g. ``{"Policy": [document
    titles]}``); ``aliases`` maps ``(entity_type, alias)`` to a canonical name.
    """

    def __init__(self, catalog: dict[str, Iterable[str]] | None = None,
                 aliases: dict[tuple[str, str], str] | None = None):
        self.catalog = {t: list(dict.fromkeys(names)) for t, names in (catalog or {}).items()}
        self.aliases = {(t, normalize_name(a)): c for (t, a), c in (aliases or {}).items()}

    def _known(self, entity_type: str, batch_names: Iterable[str]) -> list[str]:
        return list(dict.fromkeys([*self.catalog.get(entity_type, []), *batch_names]))

    def resolve_name(self, entity_type: str, name: str, known: list[str] | None = None) -> tuple[str, str]:
        """Return ``(canonical_name, method)``; method ``"none"`` means the name stands alone."""
        known = self._known(entity_type, []) if known is None else known
        hit = self.aliases.get((entity_type, normalize_name(name)))
        if hit:
            return hit, "alias"
        core = core_tokens(entity_type, name)
        sig = " ".join(sorted(set(core)))
        catalog = self.catalog.get(entity_type, [])
        for cand in catalog:  # catalog names win over batch surface forms
            if cand != name and signature(entity_type, cand) == sig:
                return cand, "signature"
        if len(core) == 1 and 2 <= len(core[0]) <= 6:
            matches = [c for c in known if c != name and acronym(core_tokens(entity_type, c)) == core[0]]
            if len(matches) == 1:
                return matches[0], "acronym"
        if entity_type == "Policy" and catalog:
            matches = [c for c in catalog if c != name and set(core) <= expanded_tokens(core_tokens(entity_type, c))]
            if len(matches) == 1:
                return matches[0], "catalog"
        return name, "none"

    def resolve(self, nodes: Iterable[KGNode], edges: Iterable[KGEdge]) -> ResolutionResult:
        nodes, edges = list(nodes), list(edges)
        by_type: dict[str, list[str]] = defaultdict(list)
        for n in nodes:
            by_type[n.entity_type].append(n.name)
        canonical: dict[str, tuple[str, str]] = {}  # old node_id -> (canonical name, method)
        for etype, names in by_type.items():
            # within the batch, identical signatures collapse onto the most frequent (then longest) form
            counts = Counter(names)
            groups: dict[str, list[str]] = defaultdict(list)
            for nm in counts:
                groups[signature(etype, nm)].append(nm)
            rep = {nm: max(g, key=lambda x: (counts[x], len(x))) for g in groups.values() for nm in g}
            known = self._known(etype, dict.fromkeys(rep.values()))
            for nm in counts:
                target, method = self.resolve_name(etype, rep[nm], known)
                if method == "none" and rep[nm] != nm:
                    method = "signature"
                if method != "none":
                    if target not in self.catalog.get(etype, []):
                        # one more step so alias -> batch form -> catalog chains end at the catalog name
                        target = self.resolve_name(etype, target, known)[0]
                    canonical[node_id(etype, nm)] = (target, method)

        id_map: dict[str, str] = {}
        links: list[AliasLink] = []
        out_nodes: list[KGNode] = []
        for n in nodes:
            target, method = canonical.get(n.node_id, (n.name, "none"))
            new = replace(n, name=target) if target != n.name else n
            if new.node_id != n.node_id:
                id_map[n.node_id] = new.node_id
                links.append(AliasLink(n.entity_type, n.name, n.node_id, target, new.node_id, method))
            out_nodes.append(new)
        out_edges = []
        for e in edges:
            src, dst = id_map.get(e.src_id, e.src_id), id_map.get(e.dst_id, e.dst_id)
            if src == dst and e.src_id != e.dst_id:
                continue  # both ends merged into one entity: not a relation any more
            out_edges.append(replace(e, src_id=src, dst_id=dst))
        g = validate_graph(out_nodes, out_edges)
        unique_links = list({a.alias_id: a for a in links}.values())
        stats = {"nodes_in": len({n.node_id for n in nodes}), "nodes_out": len(g.nodes),
                 "edges_in": len({e.edge_id for e in edges}), "edges_out": len(g.edges),
                 "merged_aliases": len(unique_links),
                 "by_method": dict(Counter(a.method for a in unique_links))}
        return ResolutionResult(g.nodes, g.edges, unique_links, g.rejected, stats)
