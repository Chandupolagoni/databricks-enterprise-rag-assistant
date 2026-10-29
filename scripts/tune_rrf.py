"""Grid-search the RRF keyword/vector weights on the golden set and print a Markdown table.

    python scripts/tune_rrf.py            # local hashing embedder, k=5
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ekb.evaluation import load_golden  # noqa: E402
from ekb.pipeline import build_chunks  # noqa: E402
from ekb.rbac import load_entitlements  # noqa: E402
from ekb.retrieval import LocalHybridRetriever  # noqa: E402
from ekb.tuning import as_markdown, grid_search_rrf, leave_one_out_rrf  # noqa: E402


def main() -> None:
    chunks = build_chunks(ROOT / "data" / "raw", ROOT / "data" / "metadata" / "documents.csv")
    users = load_entitlements(ROOT / "data" / "rbac" / "user_entitlements.csv")
    golden = load_golden(ROOT / "data" / "eval" / "golden_questions.jsonl")
    retriever = LocalHybridRetriever(chunks)
    trials = grid_search_rrf(retriever, golden, users)
    print(as_markdown(trials))
    best = trials[0]
    print(f"\nbest: keyword={best.keyword_weight} vector={best.vector_weight} rrf_k={best.rrf_k}")
    print("leave-one-out:", leave_one_out_rrf(retriever, golden, users))


if __name__ == "__main__":
    main()
