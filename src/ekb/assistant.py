"""End-to-end request flow of the Enterprise Knowledge Assistant.

    question ─► [1] input guardrail ─► optional follow-up rewrite (history → standalone
                question, re-checked by [1]) ─► mask PII in question
             ─► hybrid retrieval with [3] RBAC filter (top-k candidates)
             ─► rerank (top-n)
             ─► optional parent expansion (child chunk → parent passage, token budget)
             ─► [2] mask PII in context
             ─► prompt + LLM
             ─► [4] output verification (groundedness + citation check) ─► mask PII in answer
             ─► answer + citations   (or a refusal)

Every step is wrapped in an MLflow trace span when MLflow is installed, so a request can
be inspected step by step in the MLflow UI (the Databricks-native equivalent of LangSmith
traces).
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field

from .config import Settings
from .guardrails import check_input, mask_pii, strip_citations, verify_answer
from .prompting import build_messages
from .rbac import UserContext

try:  # optional dependency
    import mlflow

    def _span(name: str, span_type: str = "UNKNOWN"):
        return mlflow.start_span(name=name, span_type=span_type)
except Exception:  # pragma: no cover
    mlflow = None

    @contextlib.contextmanager
    def _span(name: str, span_type: str = "UNKNOWN"):
        class _Noop:
            def set_inputs(self, *_):  # noqa: D401
                pass

            def set_outputs(self, *_):
                pass

            def set_attributes(self, *_):
                pass
        yield _Noop()


@dataclass
class Citation:
    doc_id: str
    title: str
    section: str
    path: str


@dataclass
class AssistantResponse:
    answer: str
    status: str                          # answered | refused_input | no_context | refused_ungrounded
    citations: list[Citation] = field(default_factory=list)
    groundedness: float | None = None
    guardrail_events: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    standalone_question: str | None = None   # set when a follow-up was rewritten; store it in history

    def to_dict(self) -> dict:
        return asdict(self)


class KnowledgeAssistant:
    def __init__(self, retriever, reranker, llm, settings: Settings | None = None, context_expander=None,
                 query_rewriter=None):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.s = settings or Settings()
        self.context_expander = context_expander     # e.g. parent_child.ParentExpander
        self.query_rewriter = query_rewriter         # e.g. query_rewrite.LLMQueryRewriter

    def ask(self, question: str, user: UserContext, history: list[dict] | None = None) -> AssistantResponse:
        """Answer ``question``. ``history`` is the prior chat as ``[{"role", "content"}]``;
        with a ``query_rewriter`` set, follow-ups are rewritten into standalone questions."""
        t0 = time.perf_counter()
        events: list[str] = []
        with _span("ekb.ask", "CHAIN") as root:
            root.set_inputs({"question": question, "user_id": user.user_id})

            # [1] input filtration
            with _span("guardrail.input", "GUARDRAIL") as sp:
                check = check_input(question)
                sp.set_outputs({"allowed": check.allowed, "reasons": check.reasons})
            if not check.allowed:
                events += [f"input_blocked:{r}" for r in check.reasons]
                return self._done(AssistantResponse(
                    "I can't help with that request.", "refused_input", guardrail_events=events), t0, root)

            standalone = None
            if self.query_rewriter is not None and history:
                with _span("query.rewrite", "PARSER") as sp:
                    rw = self.query_rewriter.rewrite(question, history)
                    sp.set_outputs({"question": rw.question, "rewritten": rw.rewritten, "method": rw.method,
                                    "reasons": rw.reasons})
                if rw.rewritten:
                    events.append(f"query_rewritten:{rw.method}")
                    recheck = check_input(rw.question)      # history is user-controlled too
                    if not recheck.allowed:
                        events += [f"input_blocked_after_rewrite:{r}" for r in recheck.reasons]
                        return self._done(AssistantResponse(
                            "I can't help with that request.", "refused_input", guardrail_events=events), t0, root)
                    question = standalone = rw.question

            q_masked = mask_pii(question)
            if q_masked.found_pii:
                events.append("pii_masked:question")

            # [3] RBAC-filtered hybrid retrieval
            with _span("retrieve.hybrid", "RETRIEVER") as sp:
                candidates = self.retriever.search(q_masked.text, user=user, k=self.s.retrieve_k)
                sp.set_outputs({"n": len(candidates), "ids": [c["chunk_id"] for c in candidates]})
            with _span("rerank", "RERANKER") as sp:
                top = self.reranker.rerank(q_masked.text, candidates, top_n=self.s.rerank_top_n)
                sp.set_outputs({"ids": [c["chunk_id"] for c in top]})
            if not top:
                return self._done(AssistantResponse(self.s.refusal_text, "no_context", guardrail_events=events), t0, root, standalone)
            retrieved_ids = [c["chunk_id"] for c in top]
            if self.context_expander is not None:
                with _span("context.expand", "RETRIEVER") as sp:
                    top = self.context_expander.expand(top)
                    sp.set_outputs({"ids": [c.get("parent_id") or c["chunk_id"] for c in top]})

            # [2] PII masking of the context before it reaches the LLM
            safe_chunks = []
            for c in top:
                red = mask_pii(c["text"])
                if red.found_pii:
                    events.append(f"pii_masked:{c['chunk_id']}:{','.join(sorted(red.findings))}")
                safe_chunks.append(dict(c, text=red.text, content=mask_pii(c.get("content", "")).text))

            with _span("llm.generate", "LLM") as sp:
                raw = self.llm.chat(build_messages(q_masked.text, safe_chunks, self.s.refusal_text))
                sp.set_outputs({"raw": raw})

            if strip_citations(raw).strip().rstrip(".") == self.s.refusal_text.rstrip("."):
                return self._done(AssistantResponse(self.s.refusal_text, "no_context",
                                                    guardrail_events=events,
                                                    retrieved_chunk_ids=retrieved_ids), t0, root, standalone)

            # [4] output verification
            with _span("guardrail.output", "GUARDRAIL") as sp:
                v = verify_answer(raw, safe_chunks, self.s.min_groundedness)
                sp.set_outputs(asdict(v))
            if not v.passed:
                events.append(f"output_blocked:groundedness={v.groundedness}:invalid_citations={v.invalid_citations}")
                return self._done(AssistantResponse(self.s.refusal_text, "refused_ungrounded",
                                                    groundedness=v.groundedness, guardrail_events=events,
                                                    retrieved_chunk_ids=retrieved_ids), t0, root, standalone)

            answer = mask_pii(raw).text
            by_doc = {c["doc_id"]: c for c in safe_chunks}
            citations = [Citation(d, by_doc[d].get("title", ""), by_doc[d].get("section", ""),
                                  by_doc[d].get("relative_path", ""))
                         for d in dict.fromkeys(v.cited_ids) if d in by_doc]
            return self._done(AssistantResponse(answer, "answered", citations, v.groundedness, events,
                                                retrieved_ids), t0, root, standalone)

    @staticmethod
    def _done(resp: AssistantResponse, t0: float, root, standalone: str | None = None) -> AssistantResponse:
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        resp.standalone_question = standalone
        root.set_outputs({"status": resp.status, "answer": resp.answer, "groundedness": resp.groundedness})
        return resp
