"""Hybrid chunking = semantic chunking + sliding-window chunking.

1. **Structure first** — never cross a section heading (a "Holding Period" rule should not
   be glued to a "Restricted List" rule). Sections that fit in ``max_tokens`` stay whole.
2. **Semantic split** — inside a long section, embed each sentence and cut where the cosine
   similarity between neighbouring sentences drops below the Nth percentile, i.e. where
   the topic changes.
3. **Sliding window** — any semantic segment still longer than ``max_tokens`` is cut
   into overlapping windows (``overlap_tokens``) so no fact is lost at a boundary.

Each chunk carries the doc title + section heading as a prefix ("contextual header"),
which measurably helps both BM25 and vector retrieval on short policy paragraphs.

The embedding function is injected: on Databricks it calls the GTE endpoint; offline
and in tests a deterministic hashing embedder is used.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .parsing import ParsedDocument

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def count_tokens(text: str) -> int:
    """Cheap word-piece estimate (~1.3 tokens per word) — good enough for sizing chunks."""
    return math.ceil(len(text.split()) * 1.3)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def hashing_embed(texts: Sequence[str], dim: int = 256) -> list[list[float]]:
    """Deterministic bag-of-words hashing embedding (offline / test stand-in for GTE)."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in tokenize(t):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def semantic_segments(sentences: list[str], embed: EmbedFn, breakpoint_percentile: float) -> list[list[str]]:
    """Group consecutive sentences; break where similarity distance is in the top (100-p)%."""
    if len(sentences) <= 2:
        return [sentences] if sentences else []
    vecs = embed(sentences)
    distances = [1 - cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    threshold = _percentile(distances, breakpoint_percentile)
    segments, current = [], [sentences[0]]
    for i, d in enumerate(distances):
        if d > threshold:
            segments.append(current)
            current = []
        current.append(sentences[i + 1])
    segments.append(current)
    return segments


def sliding_window(sentences: list[str], max_tokens: int, overlap_tokens: int) -> list[list[str]]:
    """Pack sentences into windows <= max_tokens, carrying ~overlap_tokens into the next."""
    windows, current, size = [], [], 0
    for s in sentences:
        t = count_tokens(s)
        if current and size + t > max_tokens:
            windows.append(current)
            carry, carry_size = [], 0
            for prev in reversed(current):
                if carry_size + count_tokens(prev) > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_size += count_tokens(prev)
            current, size = carry, carry_size
        current.append(s)
        size += t
    if current:
        windows.append(current)
    return windows


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    section: str
    text: str                     # text sent to the LLM
    content: str                  # header + text: what gets embedded / keyword-indexed
    n_tokens: int
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "chunk_index": self.chunk_index,
            "section": self.section, "text": self.text, "content": self.content,
            "n_tokens": self.n_tokens, **self.metadata,
        }


def hybrid_chunk(
    doc: ParsedDocument,
    embed: EmbedFn = hashing_embed,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
    breakpoint_percentile: float = 80.0,
    metadata: dict | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        sentences = split_sentences(section.text)
        # Short sections are already one topic: keep them whole. Only long sections are
        # split semantically, so tiny policy paragraphs are not fragmented.
        if count_tokens(section.text) <= max_tokens:
            segments = [sentences]
        else:
            segments = semantic_segments(sentences, embed, breakpoint_percentile)
        for segment in segments:
            for window in sliding_window(segment, max_tokens, overlap_tokens):
                text = " ".join(window)
                header = f"{doc.title} > {section.heading}" if section.heading else doc.title
                idx = len(chunks)
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::{idx:03d}",
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    section=section.heading,
                    text=text,
                    content=f"{header}\n{text}",
                    n_tokens=count_tokens(text),
                    metadata=dict(metadata or {}),
                ))
    return chunks
