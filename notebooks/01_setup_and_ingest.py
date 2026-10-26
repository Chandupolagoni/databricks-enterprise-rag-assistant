# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Setup and ingestion (Bronze)
# MAGIC
# MAGIC * Creates the Unity Catalog schema and the `raw_docs` volume (the landing zone, playing the role of an S3 bucket / ADLS container).
# MAGIC * Copies the sample corpus (PDF, Word, HTML, Markdown, text) from this Git folder into the volume.
# MAGIC * Loads every file as **binary** into `bronze_documents`, joined with its governance metadata (business unit, category, access level).
# MAGIC * Loads the user entitlement table used for RBAC.
# MAGIC
# MAGIC Run on serverless compute (Databricks Free Edition works).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import shutil
import sys

sys.path.append(os.path.abspath("../src"))

from ekb.config import Settings

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
s = Settings.from_env(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
print(s.fq_schema, s.volume_path)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s.fq_schema} COMMENT 'Enterprise Knowledge Assistant'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {s.fq_schema}.{s.volume} COMMENT 'Landing zone for raw policy documents'")

# COMMAND ----------

# MAGIC %md ### Land the documents in the volume
# MAGIC In production this zone is fed by the source systems (SharePoint / S3 / ADLS sync). Here we copy the repo's sample corpus.

# COMMAND ----------

repo_data = os.path.abspath("../data/raw")
for bu in sorted(os.listdir(repo_data)):
    src_dir, dst_dir = os.path.join(repo_data, bu), os.path.join(s.volume_path, bu)
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        shutil.copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))
display(dbutils.fs.ls(s.volume_path))

# COMMAND ----------

# MAGIC %md ### Bronze: raw bytes + governance metadata

# COMMAND ----------

from pyspark.sql import functions as F

meta = (spark.read.option("header", True).csv(f"file:{os.path.abspath('../data/metadata/documents.csv')}")
        .withColumn("version", F.col("version").cast("int")))

files = (spark.read.format("binaryFile")
         .option("recursiveFileLookup", "true")
         .load(s.volume_path)
         .withColumn("relative_path", F.regexp_replace("path", f"^(dbfs:)?{s.volume_path}/", "")))

bronze = (files.join(meta, "relative_path", "inner")
          .withColumn("content_sha256", F.sha2(F.col("content"), 256))
          .withColumn("ingested_at", F.current_timestamp())
          .select("doc_id", "title", "business_unit", "category", "access_level", "file_format", "version",
                  "relative_path", "path", "length", "modificationTime", "content_sha256", "content", "ingested_at"))

(bronze.write.mode("overwrite").option("overwriteSchema", True)
 .option("delta.enableChangeDataFeed", "true")
 .saveAsTable(s.table("bronze_documents")))

display(spark.table(s.table("bronze_documents")).drop("content")
        .groupBy("business_unit", "file_format").count().orderBy("business_unit"))

# COMMAND ----------

unmatched = files.join(meta, "relative_path", "left_anti").count()
assert unmatched == 0, f"{unmatched} files in the volume have no metadata row — they would bypass RBAC"

# COMMAND ----------

# MAGIC %md ### RBAC entitlements

# COMMAND ----------

ent = spark.read.option("header", True).csv(f"file:{os.path.abspath('../data/rbac/user_entitlements.csv')}")
ent.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(s.table("user_entitlements"))
display(spark.table(s.table("user_entitlements")))
