"""Local (non-Spark) pipeline helpers: raw files + metadata -> chunk rows.

The Databricks notebooks run the same functions inside Spark (``mapInPandas`` / a UDF);
keeping them here lets tests and CI build the exact same chunk table locally.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .chunking import EmbedFn, hashing_embed, hybrid_chunk
from .config import Settings
from .parsing import parse_file
from .rbac import acl_tag


def read_metadata(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chunk_row_metadata(meta: dict) -> dict:
    return {
        "title": meta["title"],
        "business_unit": meta["business_unit"],
        "category": meta["category"],
        "access_level": meta["access_level"],
        "acl_tag": acl_tag(meta["business_unit"], meta["access_level"]),
        "relative_path": meta["relative_path"],
        "version": int(meta.get("version", 1)),
    }


def build_chunks(raw_dir: str | Path, metadata_csv: str | Path, settings: Settings | None = None,
                 embed: EmbedFn = hashing_embed) -> list[dict]:
    s = settings or Settings()
    rows: list[dict] = []
    for meta in read_metadata(metadata_csv):
        doc = parse_file(Path(raw_dir) / meta["relative_path"], doc_id=meta["doc_id"])
        doc.title = meta["title"] or doc.title
        for ch in hybrid_chunk(doc, embed, s.chunk_max_tokens, s.chunk_overlap_tokens,
                               s.semantic_breakpoint_percentile, chunk_row_metadata(meta)):
            rows.append(ch.to_row())
    return rows
