"""Parent-child ("small-to-big") retrieval.

Small chunks are what make retrieval precise: a 60-token paragraph about the Fedwire
cut-off scores far better than a whole SOP. But small chunks are a poor *context*: the
exception, the approval step or the definition the answer depends on often sits in the
neighbouring section, so the LLM answers half the question or refuses.

So we index the children (``silver_chunks``) and generate from their parents:

* ``assign_parents`` groups consecutive chunks of one document into parent passages of
  at most ``max_parent_tokens`` (never across documents, so a parent inherits exactly one
  ``acl_tag``) and writes ``parent_id`` onto each chunk row. Sliding-window overlap
  between chunks of the same section is removed so the parent text is not repeated.
* ``ParentExpander`` runs after reranking: it swaps each top chunk for its parent,
  de-duplicates children that share a parent (keeping the best rank), and stays inside a
  context token budget — falling back to the child when a parent does not fit.

RBAC is never widened: a parent is used only if its ``acl_tag`` equals the child's.
"""
from __future__ import annotations

from .chunking import count_tokens, split_sentences

_META_KEYS = ("title", "business_unit", "category", "access_level", "acl_tag", "relative_path", "version")


def _new_parent(doc_id: str, n: int, first: dict) -> dict:
    return {"parent_id": f"{doc_id}::p{n:02d}", "doc_id": doc_id, "chunk_ids": [], "sections": [],
            "_parts": [], **{k: first[k] for k in _META_KEYS if k in first}}


def _finish(parent: dict) -> dict:
    blocks, last_section = [], None
    for section, sentences in parent.pop("_parts"):
        body = " ".join(sentences)
        if section != last_section:
            blocks.append(f"## {section}\n{body}" if section else body)
        else:
            blocks[-1] += " " + body
        last_section = section
    parent["text"] = "\n\n".join(blocks)
    parent["n_tokens"] = count_tokens(parent["text"])
    return parent


def assign_parents(chunk_rows: list[dict], max_parent_tokens: int = 400) -> list[dict]:
    """Return parent rows and set ``parent_id`` on every chunk row (in place)."""
    by_doc: dict[str, list[dict]] = {}
    for row in chunk_rows:
        by_doc.setdefault(row["doc_id"], []).append(row)

    parents: list[dict] = []
    for doc_id, rows in by_doc.items():
        rows.sort(key=lambda r: r["chunk_index"])
        current, size, prev, n_parents = None, 0, None, 0
        for row in rows:
            sentences = split_sentences(row["text"])
            if prev is not None and prev["section"] == row["section"]:
                seen = set(split_sentences(prev["text"]))   # drop sliding-window overlap
                while sentences and sentences[0] in seen:
                    sentences.pop(0)
            t = count_tokens(" ".join(sentences))
            if current is None or (current["chunk_ids"] and size + t > max_parent_tokens):
                if current is not None:
                    parents.append(_finish(current))
                current, size = _new_parent(doc_id, n_parents, row), 0
                n_parents += 1
                sentences = split_sentences(row["text"])     # a new parent starts with the full chunk
                t = count_tokens(row["text"])
            current["chunk_ids"].append(row["chunk_id"])
            if row["section"] not in current["sections"]:
                current["sections"].append(row["section"])
            current["_parts"].append((row["section"], sentences))
            size += t
            row["parent_id"] = current["parent_id"]
            prev = row
        if current is not None:
            parents.append(_finish(current))
    return parents


class ParentExpander:
    """Replace reranked child chunks with their parent passages under a token budget."""

    def __init__(self, parents: list[dict] | dict[str, dict], max_context_tokens: int = 1500):
        self.parents = parents if isinstance(parents, dict) else {p["parent_id"]: p for p in parents}
        self.max_context_tokens = max_context_tokens

    def expand(self, chunks: list[dict]) -> list[dict]:
        out: list[dict] = []
        used_parents: dict[str, dict] = {}
        budget = self.max_context_tokens
        for c in chunks:
            pid = c.get("parent_id")
            if pid in used_parents:                     # sibling of an already-included parent
                used_parents[pid]["matched_chunk_ids"].append(c["chunk_id"])
                continue
            parent = self.parents.get(pid) if pid else None
            if (parent is not None and parent.get("acl_tag") == c.get("acl_tag")
                    and parent["n_tokens"] <= budget):
                item = dict(c, text=parent["text"], child_text=c["text"], n_tokens=parent["n_tokens"],
                            parent_sections=list(parent["sections"]),
                            matched_chunk_ids=[c["chunk_id"]])
                used_parents[pid] = item
            else:
                n = c.get("n_tokens") or count_tokens(c["text"])
                if n > budget:
                    continue
                item = dict(c, matched_chunk_ids=[c["chunk_id"]])
            budget -= item.get("n_tokens") or count_tokens(item["text"])
            out.append(item)
        return out
