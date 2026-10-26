"""Central configuration. Every value can be overridden with an EKB_* environment variable
(or a Databricks job/widget parameter passed into ``Settings.from_env``)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class Settings:
    # Unity Catalog locations
    catalog: str = "workspace"
    schema: str = "ekb"
    volume: str = "raw_docs"

    # Model serving endpoints (Databricks Foundation Model APIs).
    # Swap llm_endpoint to e.g. a Claude Sonnet endpoint where your workspace has it.
    embedding_endpoint: str = "databricks-gte-large-en"
    llm_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    judge_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"

    # Vector Search
    vs_endpoint: str = "ekb_vs_endpoint"

    # Chunking
    chunk_max_tokens: int = 220
    chunk_overlap_tokens: int = 40
    semantic_breakpoint_percentile: float = 80.0

    # Retrieval
    retrieve_k: int = 20          # candidates from hybrid search
    rerank_top_n: int = 5         # chunks sent to the LLM
    rrf_k: int = 60               # reciprocal rank fusion constant

    # Guardrails
    min_groundedness: float = 0.55
    refusal_text: str = "I don't know based on the documents you have access to."

    tags: dict = field(default_factory=lambda: {"project": "enterprise-knowledge-assistant"})

    # ---- derived names -------------------------------------------------------------
    @property
    def fq_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"

    def table(self, name: str) -> str:
        return f"{self.fq_schema}.{name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def chunks_table(self) -> str:
        return self.table("silver_chunks")

    @property
    def vs_index(self) -> str:
        return self.table("chunks_vs_index")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        kwargs = {}
        for f in fields(cls):
            if f.name == "tags":
                continue
            env = os.getenv(f"EKB_{f.name.upper()}")
            if env is not None:
                kwargs[f.name] = type(f.default)(env) if not isinstance(f.default, str) else env
        kwargs.update({k: v for k, v in overrides.items() if v not in (None, "")})
        return cls(**kwargs)
