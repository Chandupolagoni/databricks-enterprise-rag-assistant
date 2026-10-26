# Design decisions

## Why Databricks for the whole RAG stack

One governed platform for the documents, the chunk tables, the vector index, the models, the traces and the evaluation results — all under Unity Catalog permissions and lineage.

| Concern | Common AWS build | This project on Databricks |
|---|---|---|
| Document landing zone | S3 bucket | Unity Catalog Volume |
| Parsing / OCR | pdfplumber, python-docx, Textract | pdfplumber, python-docx, BeautifulSoup, `ai_parse_document` |
| Embeddings | Titan Text Embeddings (Bedrock) | `databricks-gte-large-en` (Foundation Model API) |
| Vector store | OpenSearch | Mosaic AI Vector Search (Delta Sync index) |
| Hybrid search | OpenSearch BM25 + k-NN | Vector Search `query_type="HYBRID"` |
| Reranker | Cohere Rerank (Bedrock) | LLM listwise reranker on a serving endpoint |
| LLM | Claude Sonnet (Bedrock) | Foundation Model API endpoint (Llama 3.3 70B default; Claude where available) |
| PII detection | Comprehend + regex | regex + Luhn rules (Presidio on the roadmap) |
| Tracing | LangSmith, OpenTelemetry, CloudWatch | MLflow Tracing (OpenTelemetry-compatible) |
| Evaluation | RAGAS | custom metrics + LLM judge, logged to MLflow (RAGAS on the roadmap) |
| Orchestration | Step Functions / Airflow | Databricks Jobs via Asset Bundles |

## Chunking
* **Structure first.** Policy sections are self-contained rules; mixing two sections in one chunk hurts precision. Sections that fit within `chunk_max_tokens` (220) stay whole.
* **Semantic split** for long sections: cut where the cosine distance between neighbouring sentence embeddings is in the top 20 %.
* **Sliding window** (40-token overlap) as the safety net for anything still too long.
* **Contextual header** (`Title > Section`) is prepended to the embedded / keyword-indexed text but not to the text sent to the LLM.

## Retrieval
* `retrieve_k = 20` candidates for recall, `rerank_top_n = 5` for precision.
* Reciprocal Rank Fusion (k = 60) avoids having to calibrate BM25 scores against cosine similarities.

## RBAC model
* Access levels are ordered: `public < internal < confidential < restricted`.
* A chunk's `acl_tag` is `public` or `<business_unit>:<access_level>`.
* A user with business units B and clearance C may read `public` plus every `b:l` for b ∈ B and l ≤ C.
* The tag list is a single `IN` filter on one column — cheap for the vector index and easy to mirror as a Unity Catalog row filter.

## Output verification
* Numbers are the riskiest hallucinations in policy answers, so any number in the answer that is not in the retrieved context fails that sentence outright.
* Citations must reference retrieved documents; an answer without citations fails.
* On failure the assistant returns the refusal text rather than a best effort.

## Testability
All logic lives in `src/ekb` with no Spark dependency, so CI runs 44 tests, including a full offline RAG evaluation with a BM25 + hashing-embedding retriever and a deterministic stand-in LLM. The notebooks only wire the same classes to Databricks services.
