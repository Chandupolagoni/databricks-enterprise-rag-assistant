# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Policy knowledge graph (GraphRAG) — schema, LLM extraction and entity resolution
# MAGIC
# MAGIC Hybrid search answers "what does the wire SOP say?" but not "which policies does the Bank Secrecy Act
# MAGIC govern?" — that answer lives in relationships across documents. This notebook creates the graph tables with
# MAGIC a **fixed schema** (`ekb.kg_schema`): extraction may only produce these entity types and relations, and every
# MAGIC node and edge keeps the source `doc_id`, `version` and a verbatim evidence span.
# MAGIC
# MAGIC | Relation | Allowed (source → target) |
# MAGIC |---|---|
# MAGIC | `GOVERNED_BY` | Policy → Regulation, Policy → Policy |
# MAGIC | `APPLIES_TO` | Policy → Department / EmployeeRole / TransactionType |
# MAGIC | `OWNED_BY` | Policy → Department |
# MAGIC | `REQUIRES` | Policy → Policy, TransactionType → Policy |

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from ekb.config import Settings
import pandas as pd
from pyspark.sql import functions as F

from ekb.kg_extract import LLMGraphExtractor
from ekb.kg_resolve import EntityResolver
from ekb.kg_schema import KGEdge, KGNode, kg_table_ddl, node_id, schema_prompt, validate_graph
from ekb.llm import DatabricksLLM

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")
s = Settings.from_env(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"),
                      llm_endpoint=dbutils.widgets.get("llm_endpoint"))

# COMMAND ----------

# MAGIC %md ### Create `silver_kg_nodes` / `silver_kg_edges` (CHECK constraints enforce the vocabulary in Delta too)

# COMMAND ----------

for ddl in kg_table_ddl(s.table("silver_kg_nodes"), s.table("silver_kg_edges")):
    spark.sql(ddl)
print(schema_prompt())

# COMMAND ----------

# MAGIC %md ### Smoke test: a hand-written fragment validates; an off-schema edge is rejected before any write

# COMMAND ----------

ev = "The policy is owned by the Compliance department and is governed by the Bank Secrecy Act"
aml = node_id("Policy", "Anti-Money Laundering and KYC Policy")
check = validate_graph(
    [KGNode("Policy", "Anti-Money Laundering and KYC Policy", "cmp_aml_kyc_policy", 1, ev),
     KGNode("Regulation", "Bank Secrecy Act", "cmp_aml_kyc_policy", 1, ev)],
    [KGEdge(aml, "GOVERNED_BY", node_id("Regulation", "Bank Secrecy Act"), "cmp_aml_kyc_policy", 1, ev),
     KGEdge(node_id("Regulation", "Bank Secrecy Act"), "GOVERNED_BY", aml, "cmp_aml_kyc_policy", 1, ev)])
print(len(check.nodes), "nodes,", len(check.edges), "edges, rejected:", check.rejected[0][1])

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM extraction over parent passages
# MAGIC Each `silver_parents` passage (PII-masked, framed as data) goes to the LLM with the schema; replies are parsed
# MAGIC defensively, labels normalised and everything outside the schema is rejected with a reason. Only the latest
# MAGIC `version` of each document is extracted so superseded policies do not add stale relations.

# COMMAND ----------

parents = spark.sql(f"""
  SELECT p.doc_id, p.version, p.title, p.text
  FROM {s.table("silver_parents")} p
  JOIN (SELECT doc_id, max(version) AS version FROM {s.table("silver_parents")} GROUP BY doc_id) l
    USING (doc_id, version)
""").toPandas().to_dict("records")

extractor = LLMGraphExtractor(DatabricksLLM(s.llm_endpoint, max_tokens=1500))
result = extractor.extract(parents)
result.stats

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entity resolution
# MAGIC The LLM names one entity many ways ("AML policy", "AML & KYC Policy", "BSA" vs "Bank Secrecy Act"). The
# MAGIC resolver merges them per entity type — alias table, core-token signature, acronym, and (policies only) a unique
# MAGIC match against the document catalog — rewires edges to the canonical ids and logs every merge with the rule
# MAGIC that fired to `silver_kg_aliases`.

# COMMAND ----------

titles = [r.title for r in spark.table(s.table("bronze_documents")).select("title").distinct().collect()]
resolved = EntityResolver(catalog={"Policy": titles}).resolve(result.nodes, result.edges)
result.nodes, result.edges = resolved.nodes, resolved.edges
if resolved.aliases:
    (spark.createDataFrame(pd.DataFrame(resolved.alias_rows())).withColumn("resolved_at", F.current_timestamp())
     .write.mode("overwrite").saveAsTable(s.table("silver_kg_aliases")))
resolved.stats

# COMMAND ----------

# MAGIC %md ### Upsert into Delta (idempotent MERGE on the deterministic ids) and keep the reject log for review

# COMMAND ----------

def merge_rows(rows: list[dict], table: str, key: str) -> None:
    if not rows:
        return
    (spark.createDataFrame(pd.DataFrame(rows)).withColumn("extracted_at", F.current_timestamp())
     .createOrReplaceTempView("kg_updates"))
    spark.sql(f"""
      MERGE INTO {table} t USING kg_updates u ON t.{key} = u.{key}
      WHEN MATCHED THEN UPDATE SET *
      WHEN NOT MATCHED THEN INSERT *
    """)


merge_rows(result.node_rows(), s.table("silver_kg_nodes"), "node_id")
merge_rows(result.edge_rows(), s.table("silver_kg_edges"), "edge_id")
if result.rejected:
    (spark.createDataFrame(pd.DataFrame(result.reject_rows())).withColumn("extracted_at", F.current_timestamp())
     .write.mode("append").option("mergeSchema", True).saveAsTable(s.table("silver_kg_rejects")))

display(spark.sql(f"""
  SELECT relation, count(*) AS edges, count(DISTINCT doc_id) AS documents
  FROM {s.table("silver_kg_edges")} GROUP BY relation ORDER BY edges DESC
"""))
