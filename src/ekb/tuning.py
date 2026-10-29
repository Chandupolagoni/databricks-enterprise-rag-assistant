"""Tune the Reciprocal Rank Fusion weights (keyword vs vector) on the golden set.

RRF(d) = w_kw / (k + rank_bm25(d)) + w_vec / (k + rank_vec(d)). Equal weights are a sane
default, but which side deserves more trust depends on the corpus and the embedding model:
policy questions full of exact tokens ("SR 11-7", "T+1") favour BM25, paraphrased questions
favour vectors. ``grid_search_rrf`` scores every (w_kw, w_vec, rrf_k) on the golden set with
the existing retrieval metrics; ``leave_one_out_rrf`` estimates how well the *selection*
generalises — with a ~20-question golden set, the best grid point is partly noise, and the
LOO score shows whether tuning helps beyond the balanced default.

Ties are broken towards the balanced setting (|log(w_kw / w_vec)| smallest), so the tuner
never moves away from 1:1 unless the metric actually improves.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .evaluation import evaluate_retrieval

DEFAULT_WEIGHT_GRID: tuple[tuple[float, float], ...] = (
    (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 1.5), (1.0, 2.0),
)


@dataclass(frozen=True)
class RRFTrial:
    keyword_weight: float
    vector_weight: float
    rrf_k: int
    hit_rate: float
    context_precision: float
    mrr: float

    @property
    def imbalance(self) -> float:
        return abs(math.log(self.keyword_weight / self.vector_weight))

    def sort_key(self) -> tuple:
        return (-self.mrr, -self.context_precision, -self.hit_rate, self.imbalance, abs(self.rrf_k - 60))


def _with_params(retriever, weights, rrf_k):
    old = (retriever.weights, retriever.rrf_k)
    retriever.weights, retriever.rrf_k = weights, rrf_k
    return old


def grid_search_rrf(retriever, golden: list[dict], users: dict, k: int = 5,
                    weight_grid: Iterable[tuple[float, float]] = DEFAULT_WEIGHT_GRID,
                    rrf_ks: Sequence[int] = (20, 60)) -> list[RRFTrial]:
    """Evaluate every grid point on ``golden``; best first. ``retriever`` must expose
    ``weights`` and ``rrf_k`` (``LocalHybridRetriever``); its settings are restored."""
    trials = []
    saved = (retriever.weights, retriever.rrf_k)
    try:
        for w in weight_grid:
            for rk in rrf_ks:
                _with_params(retriever, tuple(w), rk)
                rep = evaluate_retrieval(retriever, golden, users, k=k, mode="hybrid")
                trials.append(RRFTrial(w[0], w[1], rk, rep.hit_rate, rep.context_precision, rep.mrr))
    finally:
        retriever.weights, retriever.rrf_k = saved
    return sorted(trials, key=RRFTrial.sort_key)


def leave_one_out_rrf(retriever, golden: list[dict], users: dict, k: int = 5,
                      weight_grid: Iterable[tuple[float, float]] = DEFAULT_WEIGHT_GRID,
                      rrf_ks: Sequence[int] = (20, 60)) -> dict:
    """Pick the best grid point on n-1 questions, score it on the held-out one.
    Returns the mean held-out MRR and how often each setting was selected."""
    rows = [g for g in golden if g.get("relevant_doc_ids")]
    grid = list(weight_grid)
    # Per-question MRR for each grid point, computed once.
    per_q: dict[tuple, list[float]] = {}
    saved = (retriever.weights, retriever.rrf_k)
    try:
        for w in grid:
            for rk in rrf_ks:
                _with_params(retriever, tuple(w), rk)
                per_q[(w[0], w[1], rk)] = [evaluate_retrieval(retriever, [g], users, k=k).mrr for g in rows]
    finally:
        retriever.weights, retriever.rrf_k = saved

    def rank_key(item):
        (wk, wv, rk), score = item
        return (-score, abs(math.log(wk / wv)), abs(rk - 60))

    held_out, picks = [], {}
    for i in range(len(rows)):
        train = {p: sum(s[:i] + s[i + 1:]) for p, s in per_q.items()}
        best = min(train.items(), key=rank_key)[0]
        picks[best] = picks.get(best, 0) + 1
        held_out.append(per_q[best][i])
    return {"loo_mrr": round(sum(held_out) / len(held_out), 4) if held_out else 0.0,
            "selected": {f"{wk}:{wv}@k={rk}": n for (wk, wv, rk), n in sorted(picks.items())}}


def as_markdown(trials: Sequence[RRFTrial], top: int = 8) -> str:
    lines = ["| w_keyword | w_vector | rrf_k | hit@5 | precision@5 | MRR@5 |", "|---:|---:|---:|---:|---:|---:|"]
    for t in trials[:top]:
        lines.append(f"| {t.keyword_weight} | {t.vector_weight} | {t.rrf_k} | {t.hit_rate:.3f} | "
                     f"{t.context_precision:.3f} | {t.mrr:.3f} |")
    return "\n".join(lines)
