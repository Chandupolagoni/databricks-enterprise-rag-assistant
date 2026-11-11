"""Graph retrieval step: expand the top chunks' policies by one hop in the knowledge graph.

Hybrid search ranks passages by similarity to the question, so it finds the SOP a user asks
about but not the policies that SOP depends on. "What must be done before a wealth client's
account is opened?" retrieves the onboarding SOP; the AML/KYC and Suitability policies it
*requires* use different words and are often cut off by the top-n limit. The knowledge graph
already holds those links (``REQUIRES`` / ``GOVERNED_BY`` between policies, each backed by an
evidence span), so after reranking:

1. every reranked chunk's document is mapped to its ``Policy`` node;
2. outgoing ``REQUIRES`` / ``GOVERNED_BY`` edges to other policies are followed one hop
   (seed order = rerank order, so links of the best chunk come first);
3. for each linked policy not already in the context, its passage that overlaps the question
   most is appended — only if the user is authorised to read it (the graph never widens
   RBAC) and it fits a separate token budget, capped at ``max_added`` documents.

Added items carry ``graph_link`` (e.g. ``SOP: Wealth Client Onboarding REQUIRES Investment
Suitability Policy``) so the trace shows why a document was pulled in. One hop and a small
cap keep the step cheap and stop popular hub policies from flooding the context.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from .kg_schema import KGEdge, entity_type_of, node_id
from .rbac import UserContext

DEFAULT_RELATIONS = ("REQUIRES", "GOVERNED_BY")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the and or of to in on for is are be by with what which who when how do does must i".split())


def _terms(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP}


class GraphExpander:
    def __init__(self, edges: Iterable[KGEdge], passages: Iterable[Mapping], doc_titles: Mapping[str, str],
                 relations: Iterable[str] = DEFAULT_RELATIONS, max_added: int = 2, max_added_tokens: int = 600):
        self.relations = frozenset(relations)
        self.max_added, self.max_added_tokens = max_added, max_added_tokens
        self.doc_policy = {d: node_id("Policy", t) for d, t in doc_titles.items()}
        self.policy_doc = {pid: d for d, pid in self.doc_policy.items()}
        self.titles = dict(doc_titles)
        self.out_links: dict[str, list[tuple[str, str]]] = {}
        for e in edges:
            if e.relation in self.relations and entity_type_of(e.dst_id) == "Policy":
                links = self.out_links.setdefault(e.src_id, [])
                if (e.relation, e.dst_id) not in links:
                    links.append((e.relation, e.dst_id))
        self.passages: dict[str, list[dict]] = {}
        for p in passages:
            self.passages.setdefault(p["doc_id"], []).append(dict(p))

    def linked_docs(self, doc_ids: Iterable[str]) -> list[tuple[str, str, str]]:
        """(seed doc, relation, linked doc) one hop away, in seed order, de-duplicated."""
        seeds = list(dict.fromkeys(doc_ids))
        out, seen = [], set(seeds)
        for d in seeds:
            for rel, dst in self.out_links.get(self.doc_policy.get(d, ""), []):
                linked = self.policy_doc.get(dst)
                if linked and linked not in seen:
                    seen.add(linked)
                    out.append((d, rel, linked))
        return out

    def _best_passage(self, question: str, doc_id: str, user: UserContext) -> dict | None:
        q = _terms(question)
        readable = [p for p in self.passages.get(doc_id, []) if user.is_authorized(p)]
        if not readable:
            return None
        return max(readable, key=lambda p: (len(q & _terms(p["text"])), -readable.index(p)))

    def expand(self, question: str, chunks: list[dict], user: UserContext) -> list[dict]:
        out = list(chunks)
        budget, added = self.max_added_tokens, 0
        for seed, rel, doc in self.linked_docs(c["doc_id"] for c in chunks):
            if added >= self.max_added:
                break
            p = self._best_passage(question, doc, user)
            if p is None:
                continue
            n = int(p.get("n_tokens") or len(p["text"].split()))
            if n > budget:
                continue
            budget -= n
            added += 1
            out.append(dict(p, graph_link=f"{self.titles.get(seed, seed)} {rel} {self.titles.get(doc, doc)}"))
        return out


def edges_from_fixture(rows: Iterable[Mapping]) -> list[KGEdge]:
    """Edges from ``{source_type, source, relation, target_type, target, doc_id, evidence}`` rows."""
    return [KGEdge(node_id(r["source_type"], r["source"]), r["relation"], node_id(r["target_type"], r["target"]),
                   r["doc_id"], int(r.get("version", 1)), r["evidence"]) for r in rows]
