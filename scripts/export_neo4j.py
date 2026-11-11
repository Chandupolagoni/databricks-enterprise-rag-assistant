"""Write the policy knowledge graph as a Cypher MERGE script for ``cypher-shell``.

    python scripts/export_neo4j.py > policy_graph.cypher          # offline fixture graph
    python scripts/export_neo4j.py --prune --export-id 2026-11-11 > policy_graph.cypher
    cypher-shell -a neo4j+s://<host> -u neo4j -f policy_graph.cypher

On Databricks, notebook 06 exports the Delta tables directly with ``ekb.kg_neo4j.run_export``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ekb.graph_retrieval import edges_from_fixture  # noqa: E402
from ekb.kg_neo4j import to_cypher_script  # noqa: E402
from ekb.kg_schema import KGNode  # noqa: E402


def fixture_graph(path: Path) -> tuple[list[KGNode], list]:
    rows = json.loads(path.read_text())["edges"]
    nodes = [KGNode(r[f"{side}_type"], r[side], r["doc_id"], int(r.get("version", 1)), r["evidence"])
             for r in rows for side in ("source", "target")]
    return nodes, edges_from_fixture(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", type=Path, default=ROOT / "data" / "kg" / "policy_graph_fixture.json")
    ap.add_argument("--export-id", default=date.today().isoformat())
    ap.add_argument("--prune", action="store_true", help="delete nodes/relationships missing from this export")
    args = ap.parse_args()
    nodes, edges = fixture_graph(args.graph)
    sys.stdout.write(to_cypher_script(nodes, edges, args.export_id, prune=args.prune))


if __name__ == "__main__":
    main()
