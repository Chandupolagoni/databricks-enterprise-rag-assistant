"""RAG evaluation: retriever metrics + generation metrics.

Retriever (needs a golden set of question -> relevant doc ids):
  * hit_rate@k          — was any relevant doc retrieved?
  * context_precision@k — share of retrieved chunks that come from relevant docs
  * context_recall@k    — share of relevant docs that were retrieved
  * MRR                 — 1 / rank of the first relevant chunk

Generation:
  * faithfulness        — lexical groundedness from guardrails.verify_answer
  * answer_correctness  — share of expected key facts (numbers / phrases) present in the answer
  * refusal_accuracy    — out-of-scope / unauthorized questions should be refused
  * LLM-as-judge prompt — JUDGE_PROMPT, run with the judge endpoint for a semantic score
    (RAGAS or ``mlflow.genai.evaluate`` scorers can be plugged in the same way)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


def load_golden(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def hit_rate(retrieved_docs: list[str], relevant: set[str]) -> float:
    return float(any(d in relevant for d in retrieved_docs))


def context_precision(retrieved_docs: list[str], relevant: set[str]) -> float:
    return sum(d in relevant for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0


def context_recall(retrieved_docs: list[str], relevant: set[str]) -> float:
    return len(relevant & set(retrieved_docs)) / len(relevant) if relevant else 1.0


def reciprocal_rank(retrieved_docs: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved_docs, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def answer_correctness(answer: str, expected_facts: list[str]) -> float:
    if not expected_facts:
        return 1.0
    a = answer.lower()
    return sum(f.lower() in a for f in expected_facts) / len(expected_facts)


@dataclass
class RetrievalReport:
    mode: str
    k: int
    n: int
    hit_rate: float
    context_precision: float
    context_recall: float
    mrr: float

    def as_metrics(self) -> dict[str, float]:
        p = f"{self.mode}_at{self.k}"
        return {f"{p}_hit_rate": self.hit_rate, f"{p}_context_precision": self.context_precision,
                f"{p}_context_recall": self.context_recall, f"{p}_mrr": self.mrr}


def evaluate_retrieval(retriever, golden: list[dict], users: dict, k: int = 5, mode: str = "hybrid",
                       reranker=None) -> RetrievalReport:
    rows = [g for g in golden if g.get("relevant_doc_ids")]
    hr, cp, cr, rr = [], [], [], []
    for g in rows:
        user = users[g["user_id"]]
        cands = retriever.search(g["question"], user=user, k=max(k, 20) if reranker else k, mode=mode)
        if reranker:
            cands = reranker.rerank(g["question"], cands, top_n=k)
        docs = [c["doc_id"] for c in cands[:k]]
        rel = set(g["relevant_doc_ids"])
        hr.append(hit_rate(docs, rel))
        cp.append(context_precision(docs, rel))
        cr.append(context_recall(docs, rel))
        rr.append(reciprocal_rank(docs, rel))
    r = lambda xs: round(mean(xs), 4) if xs else 0.0  # noqa: E731
    return RetrievalReport(mode + ("+rerank" if reranker else ""), k, len(rows), r(hr), r(cp), r(cr), r(rr))


def evaluate_assistant(assistant, golden: list[dict], users: dict) -> tuple[dict, list[dict]]:
    details = []
    for g in golden:
        resp = assistant.ask(g["question"], users[g["user_id"]])
        should_refuse = g.get("expect") in ("refuse", "blocked")
        refused = resp.status != "answered"
        details.append({
            "id": g["id"], "question": g["question"], "user_id": g["user_id"], "status": resp.status,
            "answer": resp.answer, "cited": [c.doc_id for c in resp.citations],
            "groundedness": resp.groundedness,
            "correctness": None if should_refuse else answer_correctness(resp.answer, g.get("expected_facts", [])),
            "refusal_ok": refused == should_refuse,
            "citation_ok": None if should_refuse else bool(set(c.doc_id for c in resp.citations) & set(g.get("relevant_doc_ids", []))),
        })
    answerable = [d for d in details if d["correctness"] is not None]
    summary = {
        "n_questions": len(details),
        "refusal_accuracy": round(mean(d["refusal_ok"] for d in details), 4),
        "answer_correctness": round(mean(d["correctness"] for d in answerable), 4) if answerable else 0.0,
        "citation_accuracy": round(mean(bool(d["citation_ok"]) for d in answerable), 4) if answerable else 0.0,
        "mean_groundedness": round(mean(d["groundedness"] for d in details if d["groundedness"] is not None), 4)
        if any(d["groundedness"] is not None for d in details) else 0.0,
    }
    return summary, details


JUDGE_PROMPT = """You are grading an answer from a retrieval-augmented assistant.

Context passages:
{context}

Question: {question}
Answer: {answer}

Score each from 1 to 5 and reply with JSON only:
{{"faithfulness": <is every claim supported by the context?>,
  "relevance": <does it answer the question?>,
  "reason": "<one sentence>"}}"""


def parse_judge(reply: str) -> dict:
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}
