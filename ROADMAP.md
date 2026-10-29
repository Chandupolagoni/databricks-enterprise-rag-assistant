# Roadmap

Work is shipped in small daily increments, each through its own `feature/<slug>-<date>` branch and pull request.
Items are picked roughly top to bottom; tick an item (`[x]`) in the same PR that delivers it.

## Retrieval quality
- [x] Parent-child retrieval: search small chunks, send the larger parent section to the LLM
- [x] Query rewriting for follow-up questions (conversation history → standalone question)
- [x] Metadata boosting: prefer the latest `version` of a document; filter by `category` when the question names one
- [ ] Tune RRF weights (keyword vs vector) from the golden set and record the result in the README
- [ ] Add 15 more golden questions (multi-document and paraphrased questions)

## GraphRAG (policy knowledge graph)
- [ ] Fixed graph schema: entities `Policy`, `Regulation`, `Department`, `EmployeeRole`, `TransactionType`; relations `GOVERNED_BY`, `APPLIES_TO`, `OWNED_BY`, `REQUIRES`
- [ ] LLM entity + relationship extraction notebook writing `silver_kg_nodes` / `silver_kg_edges` Delta tables, each with source `doc_id`, `version` and evidence span
- [ ] Entity resolution: normalise names, de-duplicate entities, merge aliases
- [ ] Fact validation: keep an edge only when its evidence span is found in the source passage
- [ ] Graph retrieval step: expand the top chunks' entities by one hop and add linked policies to the context
- [ ] Optional Neo4j export: Cypher `MERGE` upsert script for nodes and relationships

## Guardrails and security
- [ ] Presidio-based PII detection as an optional second detector next to the regex rules
- [ ] LLM-based groundedness judge as a second opinion when the lexical score is borderline
- [ ] Red-team test set: 20 prompt-injection and jailbreak variants with expected verdicts
- [ ] Prompt-injection detection inside retrieved documents (indirect injection)
- [ ] Audit log Delta table: user, question hash, retrieved doc ids, guardrail events, status

## Evaluation and observability
- [ ] RAGAS metrics (context precision/recall, faithfulness) as an alternative scorer in `05_evaluation`
- [ ] `mlflow.genai.evaluate` with built-in scorers
- [ ] Latency and token-cost tracking per request in the trace attributes
- [ ] Evaluation trend dashboard query over `gold_eval_results` (Databricks SQL)

## Pipeline and operations
- [ ] Incremental ingestion: detect new / changed / deleted documents by `content_sha256` and re-chunk only those
- [ ] Auto Loader (`cloudFiles`) stream from the landing volume into `bronze_documents`
- [ ] Data quality expectations on Silver tables (no empty chunks, every chunk has an `acl_tag`)
- [ ] Unity Catalog row filter on `silver_chunks` mirroring the RBAC rules for SQL access

## Serving and agents
- [ ] Package the assistant as an MLflow `ChatModel` / pyfunc and register it in Unity Catalog
- [ ] Deploy to a Model Serving endpoint with AI Gateway rate limits and inference tables
- [ ] FastAPI wrapper with OAuth 2.0 / JWT bearer-token auth mapping the token's claims to `UserContext`
- [ ] MCP tool server exposing `search_policies`, `get_document`, `lookup_entitlements`
- [ ] Multi-agent workflow: router agent → policy-QA agent / graph agent / escalation agent
- [ ] Databricks App chat UI (Streamlit or Gradio) with citation links

## Done
- [x] v0.1.0 — multi-format parsing, hybrid chunking, hybrid retrieval + rerank, 4 guardrails, citations, evaluation, MLflow tracing, Asset Bundle job, CI
