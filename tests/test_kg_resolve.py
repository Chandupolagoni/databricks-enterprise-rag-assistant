import csv

from ekb.kg_resolve import EntityResolver, catalog_from_documents, resolution_key, resolve_graph
from ekb.kg_schema import KGEdge, KGNode, node_id


def _catalog(root):
    with open(root / "data" / "metadata" / "documents.csv", newline="") as f:
        return catalog_from_documents(csv.DictReader(f))


AML = "Anti-Money Laundering and KYC Policy"
EV = "evidence"


def _n(t, name, doc="d1"):
    return KGNode(t, name, doc, 1, EV)


def _e(st, s, rel, tt, t, doc="d1"):
    return KGEdge(node_id(st, s), rel, node_id(tt, t), doc, 1, EV)


def test_resolution_key_drops_generic_words_and_expands_acronyms():
    assert resolution_key("the AML Policy") == ("anti", "money", "laundering")
    assert resolution_key("Compliance department") == resolution_key("Compliance team") == ("compliance",)
    assert resolution_key("SOP: Outgoing Wire Transfer Processing") == ("outgoing", "wire", "transfer", "processing")


def test_aliases_resolve_to_catalog_titles(root):
    r = EntityResolver(_catalog(root))
    pol = r.catalog["Policy"]
    assert r.match("AML policy", pol) == (AML, "containment")
    assert r.match("Wire Transfer SOP", pol) == ("SOP: Outgoing Wire Transfer Processing", "containment")
    assert r.match("the Model Risk Management policy", pol) == ("Model Risk Management Policy", "key")
    assert r.match("ORMF", pol) == ("Operational Risk Management Framework", "acronym")
    assert r.match("Investment Suitablity Policy", pol) == ("Investment Suitability Policy", "fuzzy")  # typo
    assert r.match("HR", r.catalog["Department"])[0] == "Human Resources"


def test_ambiguous_or_unknown_aliases_are_not_merged(root):
    r = EntityResolver(_catalog(root))
    pol = r.catalog["Policy"]
    target, why = r.match("risk policy", pol)        # operational / model / vendor risk
    assert target is None and why.startswith("ambiguous")
    assert r.match("Travel Expense Policy", pol) == (None, "no match")


def test_resolve_graph_merges_nodes_and_remaps_edges(root):
    nodes = [_n("Policy", AML), _n("Policy", "AML policy", "d2"), _n("Policy", "the AML and KYC Policy", "d3"),
             _n("Regulation", "Bank Secrecy Act"), _n("Regulation", "BSA", "d2"),
             _n("Department", "Compliance department"), _n("Department", "Compliance", "d2"),
             _n("Policy", "SOP: Outgoing Wire Transfer Processing", "d3")]
    edges = [_e("Policy", AML, "GOVERNED_BY", "Regulation", "Bank Secrecy Act"),
             _e("Policy", "AML policy", "GOVERNED_BY", "Regulation", "BSA", "d2"),       # duplicate after merge
             _e("Policy", "AML policy", "OWNED_BY", "Department", "Compliance department", "d2"),
             _e("Policy", "SOP: Outgoing Wire Transfer Processing", "REQUIRES", "Policy", "the AML and KYC Policy", "d3"),
             _e("Policy", "AML policy", "REQUIRES", "Policy", AML, "d2")]                  # becomes a self loop
    res = resolve_graph(nodes, edges, EntityResolver(_catalog(root)))
    aml, bsa = node_id("Policy", AML), node_id("Regulation", "Bank Secrecy Act")
    comp = node_id("Department", "Compliance")
    assert {n.node_id for n in res.nodes} == {aml, bsa, comp, node_id("Policy", "SOP: Outgoing Wire Transfer Processing")}
    keys = {(e.src_id, e.relation, e.dst_id) for e in res.edges}
    assert keys == {(aml, "GOVERNED_BY", bsa), (aml, "OWNED_BY", comp),
                    (node_id("Policy", "SOP: Outgoing Wire Transfer Processing"), "REQUIRES", aml)}
    assert res.stats == {"nodes_in": 8, "nodes_out": 4, "aliases_merged": 4, "edges_in": 5, "edges_out": 3,
                         "self_loops_dropped": 1}
    rules = {a["alias_name"]: a["rule"] for a in res.aliases}
    assert rules == {"AML policy": "containment", "the AML and KYC Policy": "key", "BSA": "key",
                     "Compliance department": "key"}


def test_without_catalog_fuller_extracted_name_becomes_canonical():
    nodes = [_n("Regulation", "BSA"), _n("Regulation", "BSA", "d2"), _n("Regulation", "Bank Secrecy Act", "d3")]
    res = resolve_graph(nodes, [], EntityResolver())
    assert [n.name for n in res.nodes] == ["Bank Secrecy Act"]
    assert res.aliases[0]["canonical_id"] == "regulation:bank_secrecy_act"
