"""Hybrid retrieval: keyword (BM25) + semantic (vector) search, fused with Reciprocal
Rank Fusion, with RBAC applied as a pre-filter.

Two implementations share one interface (``search(question, user, k) -> list[dict]``):

* ``LocalHybridRetriever`` — pure Python BM25 + cosine over in-memory embeddings. Used in
  unit tests, CI and for offline experiments on the chunk table.
* ``DatabricksHybridRetriever`` — Mosaic AI Vector Search with ``query_type="HYBRID"``
  (the service runs BM25 + ANN and fuses them server side) and the RBAC filter pushed
  down into the index query.

Why hybrid: policy questions mix exact tokens ("SR 11-7", "Fedwire", "T+1") that dense
embeddings blur, with paraphrases ("how long do I keep KYC files") that keyword search
misses. Fusing both fixes each one's blind spot.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, Sequence

from .chunking import EmbedFn, cosine, hashing_embed, tokenize
from .rbac import UserContext

_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "what", "how",
         "do", "does", "i", "we", "my", "our", "be", "with", "by", "at", "it", "that", "this", "when",
         "which", "who", "can", "should", "must", "if", "than", "after", "before", "need"}


def _terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(_terms(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _terms(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], k: int = 60,
                           weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """RRF(d) = sum_r w_r / (k + rank_r(d)). Rank-based, so BM25 and cosine scores
    never need to be put on the same scale."""
    rankings = list(rankings)
    weights = list(weights or [1.0] * len(rankings))
    fused: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class LocalHybridRetriever:
    def __init__(self, chunks: list[dict], embed: EmbedFn = hashing_embed, rrf_k: int = 60,
                 weights: tuple[float, float] = (1.0, 1.0)):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.embed = embed
        self.rrf_k = rrf_k
        self.weights = weights
        self.bm25 = BM25([c["content"] for c in chunks])
        self.vectors = embed([c["content"] for c in chunks])

    def keyword_ranking(self, question: str, allowed: set[int]) -> list[str]:
        sc = self.bm25.scores(question)
        order = sorted((i for i in allowed if sc[i] > 0), key=lambda i: sc[i], reverse=True)
        return [self.chunks[i]["chunk_id"] for i in order]

    def vector_ranking(self, question: str, allowed: set[int]) -> list[str]:
        qv = self.embed([question])[0]
        order = sorted(allowed, key=lambda i: cosine(qv, self.vectors[i]), reverse=True)
        return [self.chunks[i]["chunk_id"] for i in order]

    def search(self, question: str, user: UserContext | None = None, k: int = 20,
               mode: str = "hybrid") -> list[dict]:
        tags = set(user.allowed_tags()) if user else None
        allowed = {i for i, c in enumerate(self.chunks) if tags is None or c.get("acl_tag") in tags}
        if mode == "keyword":
            ids = self.keyword_ranking(question, allowed)[:k]
            return [dict(self.by_id[i], score=None) for i in ids]
        if mode == "vector":
            ids = self.vector_ranking(question, allowed)[:k]
            return [dict(self.by_id[i], score=None) for i in ids]
        fused = reciprocal_rank_fusion(
            [self.keyword_ranking(question, allowed), self.vector_ranking(question, allowed)],
            k=self.rrf_k, weights=self.weights)
        return [dict(self.by_id[cid], score=round(s, 6)) for cid, s in fused[:k]]


class DatabricksHybridRetriever:
    """Thin wrapper over a Mosaic AI Vector Search Delta Sync index (managed embeddings)."""

    COLUMNS = ["chunk_id", "doc_id", "title", "section", "text", "content", "business_unit",
               "category", "access_level", "acl_tag", "relative_path", "version"]

    def __init__(self, endpoint_name: str, index_name: str, client=None):
        if client is None:
            from databricks.vector_search.client import VectorSearchClient

            client = VectorSearchClient(disable_notice=True)
        self.index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)

    def search(self, question: str, user: UserContext | None = None, k: int = 20,
               mode: str = "hybrid") -> list[dict]:
        kwargs = dict(query_text=question, columns=self.COLUMNS, num_results=k,
                      query_type="HYBRID" if mode == "hybrid" else "ANN")
        if user is not None:
            kwargs["filters"] = user.vector_search_filter()
        resp = self.index.similarity_search(**kwargs)
        cols = [c["name"] for c in resp["manifest"]["columns"]]
        rows = [dict(zip(cols, r)) for r in resp.get("result", {}).get("data_array", [])]
        if user is not None:                       # defence in depth
            rows = [r for r in rows if user.is_authorized(r)]
        return rows


Retriever = Callable[..., list[dict]]
