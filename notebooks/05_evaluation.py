# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluation (Gold)
# MAGIC
# MAGIC Golden set: 25 questions (`data/eval/golden_questions.jsonl`) — answerable questions with known source documents, questions a user is **not entitled** to, out-of-scope questions and prompt-injection attempts.
# MAGIC
# MAGIC | Stage | Metrics |
# MAGIC |---|---|
# MAGIC | Retriever | hit rate@5, context precision@5, context recall@5, MRR — **keyword vs vector vs hybrid vs hybrid+rerank** |
# MAGIC | Generator | answer correctness (key facts), citation accuracy, refusal accuracy, lexical groundedness, **LLM-as-judge** faithfulness & relevance |
# MAGIC
# MAGIC Everything is logged to MLflow and written to `gold_eval_results` so quality can be tracked run over run.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow
import pandas as pd
from pyspark.sql import functions as F

from ekb.assistant import KnowledgeAssistant
from ekb.config import Settings
from ekb.evaluation import JUDGE_PROMPT, evaluate_assistant, evaluate_retrieval, load_golden, parse_judge
from ekb.llm import DatabricksLLM
from ekb.rbac import UserContext
from ekb.rerank import LLMReranker
from ekb.retrieval import DatabricksHybridRetriever

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
s = Settings.from_env(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
user_name = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user_name}/ekb-assistant")

golden = load_golden(os.path.abspath("../data/eval/golden_questions.jsonl"))
users = {r.user_id: UserContext(r.user_id, tuple(b for b in (r.business_units or "").split(";") if b),
                                r.clearance, tuple((r.roles or "").split(";")))
         for r in spark.table(s.table("user_entitlements")).collect()}

llm = DatabricksLLM(s.llm_endpoint)
judge = DatabricksLLM(s.judge_endpoint)
retriever = DatabricksHybridRetriever(s.vs_endpoint, s.vs_index)
reranker = LLMReranker(llm)

# COMMAND ----------

# MAGIC %md ### Retriever: vector-only vs hybrid vs hybrid + rerank

# COMMAND ----------

reports = [
    evaluate_retrieval(retriever, golden, users, k=5, mode="vector"),
    evaluate_retrieval(retriever, golden, users, k=5, mode="hybrid"),
    evaluate_retrieval(retriever, golden, users, k=5, mode="hybrid", reranker=reranker),
]
display(pd.DataFrame([r.__dict__ for r in reports]))

# COMMAND ----------

# MAGIC %md ### RRF weight tuning (keyword vs vector)
# MAGIC Vector Search `HYBRID` fuses server side with fixed weights, so the weights are tuned on a local
# MAGIC BM25 + `databricks-gte-large-en` replica of `silver_chunks`. The leave-one-out MRR shows whether the
# MAGIC best grid point generalises or just fits a few golden questions.

# COMMAND ----------

from ekb.llm import DatabricksEmbedder
from ekb.retrieval import LocalHybridRetriever
from ekb.tuning import as_markdown, grid_search_rrf, leave_one_out_rrf

chunk_cols = ["chunk_id", "doc_id", "content", "text", "acl_tag", "title", "section", "category", "version"]
local_chunks = [r.asDict() for r in spark.table(s.chunks_table).select(*chunk_cols).collect()]
local = LocalHybridRetriever(local_chunks, embed=DatabricksEmbedder(s.embedding_endpoint), rrf_k=s.rrf_k)
rrf_trials = grid_search_rrf(local, golden, users)
rrf_loo = leave_one_out_rrf(local, golden, users)
print(as_markdown(rrf_trials))
print(rrf_loo)

# COMMAND ----------

# MAGIC %md ### End-to-end assistant + LLM judge

# COMMAND ----------

assistant = KnowledgeAssistant(retriever, reranker, llm, s)
summary, details = evaluate_assistant(assistant, golden, users)

for d in details:
    if d["status"] == "answered":
        ctx = "\n".join(c["text"] for c in retriever.search(d["question"], users[d["user_id"]], k=5))
        verdict = parse_judge(judge.complete(JUDGE_PROMPT.format(context=ctx, question=d["question"], answer=d["answer"])))
        d["judge_faithfulness"], d["judge_relevance"] = verdict.get("faithfulness"), verdict.get("relevance")

answered = [d for d in details if d.get("judge_faithfulness") is not None]
summary["judge_faithfulness"] = round(sum(d["judge_faithfulness"] for d in answered) / len(answered), 3) if answered else None
summary["judge_relevance"] = round(sum(d["judge_relevance"] for d in answered) / len(answered), 3) if answered else None
summary

# COMMAND ----------

with mlflow.start_run(run_name="ekb-eval") as run:
    mlflow.log_params({"llm_endpoint": s.llm_endpoint, "embedding_endpoint": s.embedding_endpoint,
                       "chunk_max_tokens": s.chunk_max_tokens, "chunk_overlap_tokens": s.chunk_overlap_tokens,
                       "retrieve_k": s.retrieve_k, "rerank_top_n": s.rerank_top_n,
                       "min_groundedness": s.min_groundedness})
    for r in reports:
        mlflow.log_metrics(r.as_metrics())
    best_rrf = rrf_trials[0]
    mlflow.log_params({"rrf_best_keyword_weight": best_rrf.keyword_weight,
                       "rrf_best_vector_weight": best_rrf.vector_weight, "rrf_best_k": best_rrf.rrf_k})
    mlflow.log_metrics({"rrf_best_mrr_at5": best_rrf.mrr, "rrf_loo_mrr_at5": rrf_loo["loo_mrr"]})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    mlflow.log_table(pd.DataFrame(details), "eval_details.json")
    run_id = run.info.run_id

details_df = (spark.createDataFrame(pd.DataFrame(details).astype({"cited": str}))
              .withColumn("run_id", F.lit(run_id))
              .withColumn("evaluated_at", F.current_timestamp()))
details_df.write.mode("append").option("mergeSchema", True).saveAsTable(s.table("gold_eval_results"))
display(details_df.select("id", "user_id", "status", "answer", "cited", "groundedness", "refusal_ok", "citation_ok"))

# COMMAND ----------

# Quality gate for the scheduled job: fail the run if retrieval or refusals regress.
hybrid = reports[1]
assert hybrid.hit_rate >= 0.85, f"hybrid hit_rate dropped to {hybrid.hit_rate}"
assert summary["refusal_accuracy"] >= 0.85, f"refusal accuracy dropped to {summary['refusal_accuracy']}"
print(json.dumps(summary, indent=2))
