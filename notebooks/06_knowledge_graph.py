# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Policy knowledge graph (GraphRAG) — schema, LLM extraction, fact validation, entity resolution and optional Neo4j export
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
from ekb.kg_resolve import EntityResolver, catalog_from_documents, resolve_graph
from ekb.kg_schema import KGEdge, KGNode, kg_table_ddl, node_id, schema_prompt, validate_graph
from ekb.kg_validate import sources_from_passages, validate_facts
from ekb.llm import DatabricksLLM

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("neo4j_uri", "")  # optional, e.g. neo4j+s://<id>.databases.neo4j.io; empty = skip export
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
# MAGIC ### Fact validation
# MAGIC An edge is kept only when its evidence span really occurs in the source document version (exact after
# MAGIC normalisation, or near-verbatim; source PII-masked like the extraction prompt) **and** the span mentions the
# MAGIC edge's target. Paraphrased or invented justifications ("governed by the Dodd-Frank Act") go to the reject log.
# MAGIC Runs before entity resolution so every duplicate edge is checked against its own evidence.

# COMMAND ----------

facts = validate_facts(result.edges, result.nodes, sources_from_passages(parents))
result.nodes, result.edges = facts.nodes, facts.edges
result.rejected += facts.rejected
facts.stats

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entity resolution
# MAGIC Passages name the same entity differently ("AML policy", "the AML and KYC Policy", "BSA", "Compliance team").
# MAGIC `ekb.kg_resolve` merges aliases into canonical names (document titles and business units from
# MAGIC `bronze_documents` are authoritative), remaps edges, drops duplicate edges and self loops the merge creates,
# MAGIC and leaves ambiguous aliases ("risk policy") unmerged. Every merge is written to `silver_kg_aliases` for review.

# COMMAND ----------

docs = spark.sql(f"SELECT DISTINCT title, business_unit FROM {s.table('bronze_documents')}").toPandas()
resolution = resolve_graph(result.nodes, result.edges, EntityResolver(catalog_from_documents(docs.to_dict("records"))))
result.nodes, result.edges = resolution.nodes, resolution.edges
if resolution.aliases:
    (spark.createDataFrame(pd.DataFrame(resolution.aliases)).withColumn("resolved_at", F.current_timestamp())
     .write.mode("overwrite").saveAsTable(s.table("silver_kg_aliases")))
resolution.stats

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: export to Neo4j
# MAGIC Delta stays the source of truth; Neo4j is a read replica for multi-hop Cypher and graph exploration.
# MAGIC `ekb.kg_neo4j` MERGEs on the deterministic ids (unique constraint per label), sends every value as a parameter
# MAGIC (labels and relationship types come only from the fixed schema) and prunes facts that left the Delta snapshot.
# MAGIC Credentials come from the `ekb` secret scope. Requires `%pip install neo4j` on the cluster.

# COMMAND ----------

neo4j_uri = dbutils.widgets.get("neo4j_uri")
if neo4j_uri:
    from neo4j import GraphDatabase

    from ekb.kg_neo4j import run_export

    kg_nodes = [KGNode(**{c: r[c] for c in ("entity_type", "name", "doc_id", "version", "evidence")})
                for r in spark.table(s.table("silver_kg_nodes")).collect()]
    kg_edges = [KGEdge(**{c: r[c] for c in ("src_id", "relation", "dst_id", "doc_id", "version", "evidence")})
                for r in spark.table(s.table("silver_kg_edges")).collect()]
    export_id = spark.sql("SELECT date_format(current_timestamp(), 'yyyyMMddHHmmss')").first()[0]
    auth = (dbutils.secrets.get("ekb", "neo4j_user"), dbutils.secrets.get("ekb", "neo4j_password"))
    with GraphDatabase.driver(neo4j_uri, auth=auth) as driver:
        print(run_export(driver, kg_nodes, kg_edges, export_id, prune=True))
else:
    print("neo4j_uri not set: skipping the Neo4j export")
