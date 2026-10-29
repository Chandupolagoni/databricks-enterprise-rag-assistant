"""Metadata-aware retrieval: prefer the latest document version and honour a document
category named in the question.

Hybrid search ranks chunks on text alone, so two things the metadata already knows get lost:

* **Versions.** When a policy is re-issued (``version`` 2 next to ``version`` 1 of the same
  ``doc_id``), both versions say almost the same thing and score almost the same. Quoting a
  superseded limit or deadline is a compliance finding at a bank, so chunks of superseded
  versions are demoted behind every current chunk (or dropped with ``drop_superseded=True``).
* **Categories.** "According to the wire transfer SOP ..." or "Is there a KB article on ..."
  tells us *which kind* of document the user trusts. When the question names a category,
  candidates are filtered to it — but only if enough candidates survive, so a wrongly
  detected category can never turn an answerable question into "I don't know".

``MetadataAwareRetriever`` wraps any retriever with the ``search(question, user, k, mode)``
interface (local or Vector Search), so RBAC filtering stays in the wrapped retriever and is
never widened here. Bare words such as "policy" are deliberately *not* category triggers:
users say "our policy" about SOPs and KB articles too.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# Explicit category phrases -> category value in documents.csv.
DEFAULT_CATEGORY_PATTERNS: dict[str, str] = {
    r"\bsops?\b|\bstandard operating procedures?\b": "sop",
    r"\bknowledge[- ]base\b|\bkb (?:article|entry|page)s?\b": "knowledge_base",
    r"\bcomplaints? (?:handling )?manual\b": "complaints_manual",
    r"\bpolicy (?:document|manual)s?\b|\bfinancial polic(?:y|ies)\b": "financial_policy",
}


def detect_category(question: str, patterns: Mapping[str, str] | None = None) -> str | None:
    """Return the single category the question names explicitly, else ``None``.
    Ambiguous questions (two different categories named) return ``None``."""
    found = {cat for pat, cat in (patterns or DEFAULT_CATEGORY_PATTERNS).items()
             if re.search(pat, question, re.I)}
    return found.pop() if len(found) == 1 else None


def latest_versions(rows: Iterable[Mapping]) -> dict[str, int]:
    """``doc_id -> highest version`` from chunk rows or ``documents.csv`` rows."""
    out: dict[str, int] = {}
    for r in rows:
        v = int(r.get("version") or 1)
        out[r["doc_id"]] = max(out.get(r["doc_id"], v), v)
    return out


@dataclass
class BoostReport:
    category: str | None = None
    category_applied: bool = False
    superseded: list[str] = field(default_factory=list)


class MetadataAwareRetriever:
    def __init__(self, base, catalog_versions: Mapping[str, int] | None = None,
                 category_patterns: Mapping[str, str] | None = None, min_category_hits: int = 2,
                 drop_superseded: bool = False, oversample: int = 2):
        self.base = base
        self.catalog_versions = dict(catalog_versions or {})
        self.category_patterns = category_patterns
        self.min_category_hits = min_category_hits
        self.drop_superseded = drop_superseded
        self.oversample = max(1, oversample)
        self.last_report = BoostReport()

    def _latest(self, rows: list[dict]) -> dict[str, int]:
        latest = latest_versions(rows)
        for doc_id, v in self.catalog_versions.items():
            if doc_id in latest:
                latest[doc_id] = max(latest[doc_id], v)
        return latest

    def search(self, question: str, user=None, k: int = 20, mode: str = "hybrid") -> list[dict]:
        rows = self.base.search(question, user=user, k=k * self.oversample, mode=mode)
        report = BoostReport(category=detect_category(question, self.category_patterns))

        if report.category:
            in_cat = [r for r in rows if r.get("category") == report.category]
            if len(in_cat) >= self.min_category_hits:
                rows, report.category_applied = in_cat, True

        latest = self._latest(rows)
        current, old = [], []
        for r in rows:
            is_old = int(r.get("version") or 1) < latest.get(r["doc_id"], 0)
            (old if is_old else current).append(dict(r, superseded=is_old))
        report.superseded = [r["chunk_id"] for r in old]
        self.last_report = report
        ranked = current if self.drop_superseded else current + old   # stable: keeps fused order
        return ranked[:k]
