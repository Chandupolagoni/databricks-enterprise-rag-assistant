"""Enterprise Knowledge Assistant (ekb) — a governed RAG pipeline on Databricks.

Pure-Python modules here hold the logic (parsing, chunking, retrieval, guardrails,
evaluation) so it can be unit-tested locally; the notebooks in /notebooks wire it to
Unity Catalog, Mosaic AI Vector Search, Foundation Model APIs and MLflow.
"""

__version__ = "0.1.0"
