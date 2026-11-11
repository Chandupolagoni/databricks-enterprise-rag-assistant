import importlib.util

import pytest

from ekb.kg_neo4j import cypher_literal, export_statements, run_export, to_cypher_script
from ekb.kg_schema import KGEdge, KGNode, node_id

AML = "Anti-Money Laundering and KYC Policy"
EV = "The policy is governed by the Bank Secrecy Act"
NASTY = "x'}) DETACH DELETE n //"   # extracted text must never become Cypher

NODES = [KGNode("Policy", AML, "aml", 1, EV), KGNode("Regulation", "Bank Secrecy Act", "aml", 1, NASTY),
         KGNode("Department", "Compliance", "aml", 1, "owned by Compliance")]
EDGES = [KGEdge(node_id("Policy", AML), "GOVERNED_BY", node_id("Regulation", "Bank Secrecy Act"), "aml", 1, EV),
         KGEdge(node_id("Policy", AML), "OWNED_BY", node_id("Department", "Compliance"), "aml", 1, "owned by Compliance")]


def test_statements_merge_on_ids_with_values_as_parameters():
    stmts = export_statements(NODES, EDGES, "run-1")
    assert sum(c.startswith("CREATE CONSTRAINT") for c, _ in stmts) == 5
    node_stmts = [(c, p) for c, p in stmts if "MERGE (n:" in c]
    assert {c.split("MERGE (n:")[1].split(" ")[0] for c, _ in node_stmts} == {"Policy", "Regulation", "Department"}
    edge_stmts = [c for c, _ in stmts if "MERGE (a)-" in c]
    assert any("(a:Policy" in c and "[r:GOVERNED_BY]" in c and "(b:Regulation" in c for c in edge_stmts)
    assert all(NASTY not in c for c, _ in stmts)                       # only in params
    reg = next(p for c, p in node_stmts if "n:Regulation" in c)["rows"][0]
    assert reg["props"]["evidence"] == NASTY and reg["props"]["export_id"] == "run-1"


def test_batching_and_prune():
    many = [KGNode("Department", f"Desk {i}", "d", 1, "ev") for i in range(5)]
    stmts = export_statements(many, [], "run-2", batch_size=2, prune=True)
    assert [len(p["rows"]) for c, p in stmts if "MERGE (n:" in c] == [2, 2, 1]
    prune = [c for c, p in stmts if "DELETE" in c]
    assert len(prune) == 2 and all(p == {"export_id": "run-2"} for c, p in stmts if "DELETE" in c)
    assert "DETACH DELETE n" in prune[1] and "n:Policy OR" in prune[1]


def test_invalid_graph_is_refused():
    bad = [KGEdge(node_id("Regulation", "Bank Secrecy Act"), "GOVERNED_BY", node_id("Policy", AML), "aml", 1, EV)]
    with pytest.raises(ValueError, match="violate the graph schema"):
        export_statements(NODES, bad, "run-3")
    with pytest.raises(ValueError):
        export_statements([KGNode("Customer", "Jane", "d", 1, "ev")], [], "run-3")


def test_cypher_script_escapes_literals():
    assert cypher_literal("it's a \\ path\n") == "'it\\'s a \\\\ path\\n'"
    assert cypher_literal({"a": [1, None, True]}) == "{`a`: [1, null, true]}"
    script = to_cypher_script(NODES, EDGES, "$rows")                  # placeholder-looking value is not re-expanded
    assert "$" not in script.replace("'$rows'", "")
    assert "x\\'}) DETACH DELETE n //" in script and script.count(";\n") == 5 + 3 + 2


class FakeTx:
    def __init__(self, log):
        self.log = log

    def run(self, cypher, **params):
        self.log.append(("tx", cypher, params))


class FakeSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        self.log.append(("auto", cypher, params))

    def execute_write(self, fn):
        fn(FakeTx(self.log))


class FakeDriver:
    def __init__(self):
        self.log, self.database = [], "unset"

    def session(self, database=None):
        self.database = database
        return FakeSession(self.log)


def test_run_export_ddl_outside_the_write_transaction():
    drv = FakeDriver()
    stats = run_export(drv, NODES, EDGES, "run-4", database="kg", prune=True)
    assert stats == {"statements": 5 + 3 + 2 + 2, "export_id": "run-4", "nodes": 3, "edges": 2}
    assert drv.database == "kg"
    assert all(kind == "auto" for kind, c, _ in drv.log if c.startswith("CREATE CONSTRAINT"))
    assert [kind for kind, _, _ in drv.log].count("tx") == 7


def test_export_script_runs_on_fixture(root):
    spec = importlib.util.spec_from_file_location("export_neo4j", root / "scripts" / "export_neo4j.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nodes, edges = mod.fixture_graph(root / "data" / "kg" / "policy_graph_fixture.json")
    stmts = export_statements(nodes, edges, "fixture")
    assert sum(len(p["rows"]) for c, p in stmts if "MERGE (a)-" in c) == len(edges) == 12
