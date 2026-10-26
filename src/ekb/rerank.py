"""Second-stage reranking of the hybrid-search candidates.

First-stage retrieval optimises recall (top 20); the reranker optimises precision for the
5 chunks that actually go into the prompt.

* ``LexicalReranker`` — fast, dependency-free scorer: query-term coverage in the chunk
  body and its section heading, plus the first-stage fused score as a prior. Used in CI.
* ``LLMReranker``    — listwise reranking with a Foundation Model endpoint (the role a
  Cohere Rerank model plays in a Bedrock stack). Asks the model for the ids of the most
  relevant passages in order, and falls back to the input order if the reply can't be parsed.
"""
from __future__ import annotations

import json
import re

from .chunking import tokenize
from .retrieval import _STOP


class LexicalReranker:
    def __init__(self, heading_weight: float = 0.5, prior_weight: float = 0.3):
        self.heading_weight = heading_weight
        self.prior_weight = prior_weight

    def rerank(self, question: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        q = {t for t in tokenize(question) if t not in _STOP}
        if not q or not candidates:
            return candidates[:top_n]
        n = len(candidates)
        scored = []
        for rank, c in enumerate(candidates):
            body = set(tokenize(c.get("content", "")))
            head = set(tokenize(f"{c.get('title', '')} {c.get('section', '')}"))
            coverage = len(q & body) / len(q)
            heading = len(q & head) / len(q)
            prior = 1 - rank / n
            score = coverage + self.heading_weight * heading + self.prior_weight * prior
            scored.append((score, rank, c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [dict(c, rerank_score=round(s, 4)) for s, _, c in scored[:top_n]]


RERANK_PROMPT = """You are ranking passages for a question about internal company policy.
Return ONLY a JSON list of passage ids, most relevant first, at most {top_n} ids.

Question: {question}

Passages:
{passages}"""


class LLMReranker:
    def __init__(self, llm):
        self.llm = llm   # anything with .complete(prompt) -> str

    def rerank(self, question: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        if len(candidates) <= 1:
            return candidates[:top_n]
        passages = "\n".join(f"[{i}] {c['content'][:600]}" for i, c in enumerate(candidates))
        reply = self.llm.complete(RERANK_PROMPT.format(top_n=top_n, question=question, passages=passages))
        try:
            ids = json.loads(re.search(r"\[[\d,\s]*\]", reply).group())
            order = [int(i) for i in ids if 0 <= int(i) < len(candidates)]
        except Exception:
            order = []
        seen, out = set(), []
        for i in order + list(range(len(candidates))):
            if i not in seen:
                seen.add(i)
                out.append(candidates[i])
        return out[:top_n]
