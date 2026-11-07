# Changelog

All notable changes to this project. Newest first.

## 2026-11-07 — Fixed knowledge-graph schema (GraphRAG groundwork)
- `ekb.kg_schema`: closed vocabulary of 5 entity types and 4 relations with allowed (source, target) type pairs, deterministic `node_id`s for idempotent MERGE, mandatory `doc_id` / `version` / evidence span, `validate_graph` (splits valid vs rejected with reasons, de-duplicates), `schema_prompt()` for the extraction LLM and Delta DDL with CHECK constraints; new notebook `06_knowledge_graph` creates `silver_kg_nodes` / `silver_kg_edges`.
- Why: hybrid search cannot answer relationship questions ("which policies does the Bank Secrecy Act govern?"), and unconstrained LLM extraction produces label sprawl that cannot be validated or audited; off-schema triples are now rejected before they reach Delta. 11 new tests (99 total).

## 2026-11-07 — Golden set: 15 multi-document and paraphrased questions, per-slice metrics
- 7 `multi_doc` questions (answer spans 2–3 documents) and 8 `paraphrase` questions (no policy wording reused) added to the golden set (25 → 40); `ekb.evaluation.evaluate_retrieval_by_tag` / `slice_metrics` report retrieval metrics per slice, logged to MLflow from notebook 05. New tests check that every relevant doc is readable by the asking user.
- Why: the 25 original questions mostly echo the policy text, so the 0.95+ metrics overstated quality. Per slice, paraphrased questions drop to MRR@5 0.70 (core 0.95, multi_doc 1.00) — a measurable target for later retrieval work; 4 new tests (88 total).

## 2026-10-29 — RRF weight tuning on the golden set
- `ekb.tuning`: `grid_search_rrf` scores keyword/vector weights × `rrf_k` with the retrieval metrics (ties broken towards 1:1), `leave_one_out_rrf` checks whether the selection generalises; `scripts/tune_rrf.py` prints the table; notebook 05 reruns it with `gte-large` embeddings and logs the result to MLflow. Results recorded in the README.
- Why: the 1:1 RRF weights were an untested assumption. Best grid point (3:1) lifts MRR@5 0.967 → 0.975 on the full set, but leave-one-out MRR stays 0.967, so the balanced default is kept; 4 new tests (84 total).

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
