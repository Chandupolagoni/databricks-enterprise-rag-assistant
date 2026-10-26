# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Embeddings and Vector Search index
# MAGIC
# MAGIC * Creates a **Mosaic AI Vector Search** endpoint.
# MAGIC * Creates a **Delta Sync index** on `silver_chunks` with **managed embeddings** (`databricks-gte-large-en` embeds the `content` column — title + section header + chunk text).
# MAGIC * Syncs only the columns needed for citations and RBAC filtering.
# MAGIC * Smoke-tests **hybrid** (keyword + vector) search with an RBAC filter.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.abspath("../src"))

from databricks.vector_search.client import VectorSearchClient

from ekb.config import Settings
from ekb.rbac import UserContext
from ekb.retrieval import DatabricksHybridRetriever

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
s = Settings.from_env(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if s.vs_endpoint not in existing:
    vsc.create_endpoint(name=s.vs_endpoint, endpoint_type="STANDARD")

while vsc.get_endpoint(s.vs_endpoint)["endpoint_status"]["state"] != "ONLINE":
    print("waiting for endpoint…")
    time.sleep(30)
print("endpoint online:", s.vs_endpoint)

# COMMAND ----------

def index_exists(name: str) -> bool:
    try:
        vsc.get_index(endpoint_name=s.vs_endpoint, index_name=name)
        return True
    except Exception:
        return False

if not index_exists(s.vs_index):
    vsc.create_delta_sync_index(
        endpoint_name=s.vs_endpoint,
        index_name=s.vs_index,
        source_table_name=s.chunks_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="content",
        embedding_model_endpoint_name=s.embedding_endpoint,
        columns_to_sync=DatabricksHybridRetriever.COLUMNS,
    )
else:
    vsc.get_index(endpoint_name=s.vs_endpoint, index_name=s.vs_index).sync()

index = vsc.get_index(endpoint_name=s.vs_endpoint, index_name=s.vs_index)
index.wait_until_ready(verbose=True, timeout=__import__("datetime").timedelta(minutes=30))
print(index.describe()["status"])

# COMMAND ----------

# MAGIC %md ### Smoke test: same question, two users

# COMMAND ----------

retriever = DatabricksHybridRetriever(s.vs_endpoint, s.vs_index, client=vsc)
question = "What is the recovery time objective for payments systems?"

for user in [UserContext("u_ops_manager", ("operations", "asset_servicing"), "confidential"),
             UserContext("u_payments_ops", ("operations",), "internal")]:
    hits = retriever.search(question, user=user, k=3)
    print(f"\n{user.user_id} ({user.clearance}):")
    for h in hits:
        print(f"  {h['doc_id']:<34} {h['acl_tag']:<26} {h['section']}")
# The ops manager sees the confidential BCP runbook; the payments specialist does not.
