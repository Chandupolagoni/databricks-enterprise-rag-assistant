# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Parse and chunk (Silver)
# MAGIC
# MAGIC 1. **Parse** each binary document with the parser for its format (`pdfplumber`, `python-docx`, `BeautifulSoup`) → `silver_documents`.
# MAGIC 2. **Hybrid chunking**: structure-aware → semantic split on long sections (sentence-embedding breakpoints from `databricks-gte-large-en`) → sliding window with overlap → `silver_chunks`.
# MAGIC 3. Every chunk inherits its document's governance metadata and gets an `acl_tag` used by RBAC filtering.
# MAGIC
# MAGIC `silver_chunks` has Change Data Feed enabled — it is the source of the Vector Search Delta Sync index.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import pandas as pd
from pyspark.sql import functions as F

from ekb.chunking import hashing_embed, hybrid_chunk
from ekb.config import Settings
from ekb.parsing import parse_bytes
from ekb.pipeline import chunk_row_metadata

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "ekb")
dbutils.widgets.dropdown("semantic_embedder", "gte", ["gte", "hashing"])
s = Settings.from_env(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

# COMMAND ----------

# MAGIC %md ### Parse → `silver_documents`

# COMMAND ----------

bronze = spark.table(s.table("bronze_documents"))

def parse_partition(it):
    for pdf in it:
        out = []
        for r in pdf.itertuples():
            try:
                doc = parse_bytes(bytes(r.content), r.file_format, r.doc_id)
                sections = [{"heading": x.heading, "text": x.text} for x in doc.sections]
                out.append((r.doc_id, doc.title or r.title, sections, doc.text, len(doc.text), None))
            except Exception as e:  # keep bad files visible instead of failing the batch
                out.append((r.doc_id, r.title, [], "", 0, f"{type(e).__name__}: {e}"))
        yield pd.DataFrame(out, columns=["doc_id", "parsed_title", "sections", "full_text", "n_chars", "parse_error"])

schema = "doc_id string, parsed_title string, sections array<struct<heading:string,text:string>>, full_text string, n_chars int, parse_error string"
parsed = bronze.select("doc_id", "title", "file_format", "content").mapInPandas(parse_partition, schema)

silver_docs = (bronze.drop("content").join(parsed, "doc_id")
               .withColumn("parsed_at", F.current_timestamp()))
silver_docs.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(s.table("silver_documents"))

display(spark.table(s.table("silver_documents"))
        .select("doc_id", "file_format", "n_chars", F.size("sections").alias("n_sections"), "parse_error"))

# COMMAND ----------

errors = spark.table(s.table("silver_documents")).where("parse_error IS NOT NULL").count()
assert errors == 0, f"{errors} documents failed to parse — see parse_error"

# COMMAND ----------

# MAGIC %md
# MAGIC > **Scanned PDFs:** for image-only PDFs, `ai_parse_document` (Databricks AI Functions, the managed OCR / layout service) can replace `pdfplumber`:
# MAGIC > ```sql
# MAGIC > SELECT path, ai_parse_document(content) AS parsed FROM READ_FILES('/Volumes/.../raw_docs', format => 'binaryFile')
# MAGIC > ```

# COMMAND ----------

# MAGIC %md ### Hybrid chunking → `silver_chunks`

# COMMAND ----------

from ekb.parsing import ParsedDocument, Section

if dbutils.widgets.get("semantic_embedder") == "gte":
    from ekb.llm import DatabricksEmbedder
    embed = DatabricksEmbedder(s.embedding_endpoint)
else:
    embed = hashing_embed

docs = spark.table(s.table("silver_documents")).where("parse_error IS NULL").toPandas()
rows = []
for d in docs.itertuples():
    doc = ParsedDocument(d.doc_id, d.title, d.file_format,
                         [Section(x["heading"], x["text"]) for x in d.sections])
    meta = {"title": d.title, "business_unit": d.business_unit, "category": d.category,
            "access_level": d.access_level, "relative_path": d.relative_path, "version": d.version}
    for ch in hybrid_chunk(doc, embed, s.chunk_max_tokens, s.chunk_overlap_tokens,
                           s.semantic_breakpoint_percentile, chunk_row_metadata(meta)):
        rows.append(ch.to_row())

chunks_df = (spark.createDataFrame(pd.DataFrame(rows))
             .withColumn("chunked_at", F.current_timestamp()))

(chunks_df.write.mode("overwrite").option("overwriteSchema", True)
 .option("delta.enableChangeDataFeed", "true")
 .saveAsTable(s.chunks_table))

spark.sql(f"ALTER TABLE {s.chunks_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"{len(rows)} chunks from {len(docs)} documents")

# COMMAND ----------

display(spark.table(s.chunks_table)
        .groupBy("business_unit", "access_level")
        .agg(F.count("*").alias("chunks"), F.round(F.avg("n_tokens"), 1).alias("avg_tokens"))
        .orderBy("business_unit", "access_level"))

# COMMAND ----------

display(spark.table(s.chunks_table).select("chunk_id", "acl_tag", "section", "n_tokens", "content").limit(10))
