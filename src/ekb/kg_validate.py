"""Fact validation for the policy knowledge graph: keep an edge only when its evidence is real.

The extraction prompt asks the LLM to quote a verbatim evidence span, but nothing forces it
to: models paraphrase, merge two sentences, or invent a plausible sentence to justify a
relation the passage never states ("governed by the Dodd-Frank Act"). A graph edge without
real evidence is a hallucination that GraphRAG would then feed back to the assistant as
context — and, for a bank, an unsupported claim that a policy implements a regulation is a
compliance finding waiting to happen.

``validate_facts`` re-reads the source text of each edge's ``(doc_id, version)`` and keeps
the edge only when

1. its evidence span occurs in that text — exactly after whitespace / quote / dash
   normalisation, or as a near-verbatim match (token window, ``difflib`` ratio) to tolerate
   small reformatting; the source is PII-masked first, because the extractor saw the masked
   passage and may quote ``[EMAIL]`` placeholders; and
2. the span actually mentions the edge's target entity (resolution-key tokens, acronyms
   expanded), so a real sentence cannot be reused to justify an unrelated relation.

Run it on the raw extraction output *before* entity resolution, while edges still carry the
names the model used and each duplicate keeps its own evidence.
"""
from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .guardrails import mask_pii
from .kg_resolve import CONNECTORS, resolution_key
from .kg_schema import KGEdge, KGNode

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", " ": " "})


def normalize_text(text: str) -> str:
    text = (text or "").translate(_QUOTES).lower()
    text = re.sub(r"[*_`#>|]+", " ", text)            # markdown / table formatting
    return re.sub(r"\s+", " ", text).strip(" .;:,")


def find_evidence(evidence: str, source: str, min_ratio: float = 0.9) -> tuple[bool, float, str]:
    """Does ``evidence`` occur in ``source``? Returns (found, score, method)."""
    ev, src = normalize_text(evidence), normalize_text(mask_pii(source).text)
    if not ev:
        return False, 0.0, "empty"
    if ev in src:
        return True, 1.0, "exact"
    ev_toks, src_toks = ev.split(), src.split()
    n = len(ev_toks)
    best = 0.0
    if n >= 4:  # very short spans must match exactly
        anchors = set(ev_toks)
        for size in (n - 1, n, n + 1):
            for i in range(0, max(1, len(src_toks) - size + 1)):
                window = src_toks[i:i + size]
                if window and window[0] not in anchors and window[-1] not in anchors:
                    continue
                best = max(best, difflib.SequenceMatcher(None, ev, " ".join(window)).ratio())
                if best >= 0.999:
                    break
    if best >= min_ratio:
        return True, round(best, 3), "fuzzy"
    return False, round(best, 3), "not found"


def mentions_entity(evidence: str, name: str, min_overlap: float = 0.5) -> bool:
    """Share of the entity's significant name tokens present in the evidence span."""
    want = [t for t in resolution_key(name) if t not in CONNECTORS]
    if not want:
        return True
    have = set(resolution_key(evidence))
    return sum(t in have for t in want) / len(want) >= min_overlap


def _name_from_id(nid: str) -> str:
    return nid.split(":", 1)[-1].replace("_", " ")


@dataclass
class FactValidation:
    edges: list[KGEdge]
    nodes: list[KGNode]
    rejected: list[tuple[object, list[str]]]
    stats: dict

    def reject_rows(self) -> list[dict]:
        return [{"item": repr(item), "reasons": "; ".join(reasons)} for item, reasons in self.rejected]


def validate_facts(edges: Iterable[KGEdge], nodes: Iterable[KGNode], sources: Mapping[tuple[str, int], str],
                   min_ratio: float = 0.9, check_target: bool = True) -> FactValidation:
    """Keep edges whose evidence is found in their source document version (and names the
    target); keep nodes that are grounded themselves or are an endpoint of a kept edge."""
    nodes = list(nodes)
    names = {n.node_id: n.name for n in nodes}
    kept: list[KGEdge] = []
    rejected: list[tuple[object, list[str]]] = []
    methods = {"exact": 0, "fuzzy": 0}
    for e in edges:
        src = sources.get((e.doc_id, e.version))
        if src is None:
            rejected.append((e, [f"source text not found for {e.doc_id} v{e.version}"]))
            continue
        found, score, method = find_evidence(e.evidence, src, min_ratio)
        reasons = [] if found else [f"evidence not in source (best match {score})"]
        if found and check_target and not mentions_entity(e.evidence, names.get(e.dst_id, _name_from_id(e.dst_id))):
            reasons.append(f"evidence does not mention target {e.dst_id}")
        if reasons:
            rejected.append((e, reasons))
        else:
            methods[method] += 1
            kept.append(e)
    endpoints = {i for e in kept for i in (e.src_id, e.dst_id)}
    kept_nodes = []
    for n in nodes:
        src = sources.get((n.doc_id, n.version), "")
        if n.node_id in endpoints or (src and find_evidence(n.evidence, src, min_ratio)[0]):
            kept_nodes.append(n)
        else:
            rejected.append((n, ["node evidence not in source and no grounded edge"]))
    total = len(kept) + sum(isinstance(i, KGEdge) for i, _ in rejected)
    stats = {"edges_in": total, "edges_kept": len(kept), "exact": methods["exact"], "fuzzy": methods["fuzzy"],
             "edge_reject_rate": round(1 - len(kept) / total, 4) if total else 0.0,
             "nodes_in": len(nodes), "nodes_kept": len(kept_nodes)}
    return FactValidation(kept, kept_nodes, rejected, stats)


def sources_from_passages(passages: Iterable[Mapping]) -> dict[tuple[str, int], str]:
    """Concatenate parent passages into one source text per (doc_id, version)."""
    out: dict[tuple[str, int], list[str]] = {}
    for p in passages:
        out.setdefault((p["doc_id"], int(p.get("version", 1))), []).append(p["text"])
    return {k: "\n".join(v) for k, v in out.items()}
