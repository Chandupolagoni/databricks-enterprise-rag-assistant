# Changelog

All notable changes to this project. Newest first.

## 2026-10-26 — v0.1.0: initial release
- Synthetic corpus: 24 documents for the fictional "Meridian Trust & Co." across 7 business units, 4 access levels and 5 file formats (PDF, DOCX, HTML, Markdown, text), with document metadata and user entitlements.
- `ekb` package: multi-format parsing, hybrid chunking (structure-aware + semantic breakpoints + sliding window), BM25 + vector retrieval with Reciprocal Rank Fusion, lexical and LLM rerankers.
- Four guardrails: prompt-injection input filter, PII masking, RBAC as a retrieval filter (`acl_tag`), output groundedness and citation verification.
- `KnowledgeAssistant` orchestration with MLflow trace spans and source citations.
- Evaluation: hit rate, context precision/recall, MRR, answer correctness, citation and refusal accuracy, LLM-as-judge prompt; 25-question golden set.
- Databricks notebooks 01–05 (Bronze → Silver → Vector Search → assistant → evaluation), Asset Bundle job with an evaluation quality gate.
- GitHub Actions CI: ruff + 44 pytest tests including an offline end-to-end evaluation.
