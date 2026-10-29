# Changelog

All notable changes to this project. Newest first.

## 2026-10-29 — Metadata boosting (latest version, named category)
- `ekb.metadata_boost`: `MetadataAwareRetriever` wraps any retriever, oversamples, demotes (or drops) chunks of superseded `version`s using the candidates plus a catalog of latest versions, and filters to a category the question names explicitly (`detect_category`; bare "policy" is not a trigger) only when ≥2 candidates survive. Wired into notebook 04 with versions from `bronze_documents`.
- Why: hybrid search ranks on text alone, so a re-issued policy and its stale predecessor scored alike and "per the SOP" questions could surface a policy first. Category-named probe questions MRR@5 0.833 → 1.000; golden set unchanged (hit rate 0.95, MRR 0.95); 13 new tests (80 total).

## 2026-10-28 — Query rewriting for follow-up questions
- `ekb.query_rewrite`: lexical follow-up detector, `LLMQueryRewriter` (PII-masked history, validated one-line output, heuristic fallback on endpoint errors) and a deterministic `HeuristicRewriter`; `KnowledgeAssistant.ask(..., history=)` rewrites follow-ups in a `query.rewrite` span, re-runs the input guardrail on the rewrite and returns `standalone_question`. Notebook 04 shows a two-turn chat.
- Why: retrieval is stateless, so elliptical follow-ups ("What about international ones?") searched the wrong passage. On 6 fictional follow-up conversations MRR@5 of the answering chunk rose 0.806 → 0.917; 15 new tests (67 total).

## 2026-10-26 — Parent-child retrieval
- `ekb.parent_child`: chunks are grouped into per-document parent passages (`silver_parents`, ≤400 tokens, overlap removed); `ParentExpander` swaps each reranked chunk for its parent under a 1,500-token context budget, de-duplicates siblings and never widens RBAC. Wired into `KnowledgeAssistant` (optional `context_expander`, `context.expand` trace span) and notebooks 02/04.
- Why: small chunks retrieve precisely but often cut off the exception or definition in the neighbouring section; the LLM now sees the surrounding policy text while citations and retrieval metrics still use the child chunks. Golden set unchanged (answer correctness 0.90, citation accuracy 0.95, refusal accuracy 0.96); 8 new tests (52 total).

## 2026-10-26 — v0.1.0: initial release
- Synthetic corpus: 24 documents for the fictional "Meridian Trust & Co." across 7 business units, 4 access levels and 5 file formats (PDF, DOCX, HTML, Markdown, text), with document metadata and user entitlements.
- `ekb` package: multi-format parsing, hybrid chunking (structure-aware + semantic breakpoints + sliding window), BM25 + vector retrieval with Reciprocal Rank Fusion, lexical and LLM rerankers.
- Four guardrails: prompt-injection input filter, PII masking, RBAC as a retrieval filter (`acl_tag`), output groundedness and citation verification.
- `KnowledgeAssistant` orchestration with MLflow trace spans and source citations.
- Evaluation: hit rate, context precision/recall, MRR, answer correctness, citation and refusal accuracy, LLM-as-judge prompt; 25-question golden set.
- Databricks notebooks 01–05 (Bronze → Silver → Vector Search → assistant → evaluation), Asset Bundle job with an evaluation quality gate.
- GitHub Actions CI: ruff + 44 pytest tests including an offline end-to-end evaluation.
