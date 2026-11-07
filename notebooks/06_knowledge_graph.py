# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Policy knowledge graph (GraphRAG) — schema
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
from ekb.kg_schema import KGEdge, KGNode, kg_table_ddl, node_id, schema_prompt, validate_graph

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
s = Settings.from_env(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

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
