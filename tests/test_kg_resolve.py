from ekb.kg_resolve import EntityResolver, acronym, core_tokens, expanded_tokens, signature
from ekb.kg_schema import KGEdge, KGNode, node_id

TITLES = ["Anti-Money Laundering and KYC Policy", "SOP: Outgoing Wire Transfer Processing",
          "Personal Trading Policy", "Model Risk Management Policy", "Operational Risk Management Framework"]
AML = "Anti-Money Laundering and KYC Policy"
EV = "verbatim evidence"


def n(t, name, doc="d1"):
    return KGNode(t, name, doc, 1, EV)


def e(st, s, rel, dt, d, doc="d1"):
    return KGEdge(node_id(st, s), rel, node_id(dt, d), doc, 1, EV)


def test_token_helpers():
    assert core_tokens("Policy", "SOP: Outgoing Wire Transfer Processing") == ["outgoing", "wire", "transfer",
                                                                              "processing"]
    assert signature("Department", "Compliance Department") == signature("Department", "compliance")
    assert signature("EmployeeRole", "Relationship Managers") == signature("EmployeeRole", "relationship manager")
    assert acronym(["bank", "secrecy", "act"]) == "bsa"
    assert "aml" in expanded_tokens(core_tokens("Policy", AML))


def test_resolve_name_rules():
    r = EntityResolver(catalog={"Policy": TITLES}, aliases={("Regulation", "Reg SHO"): "Regulation SHO"})
    assert r.resolve_name("Policy", "Anti-Money Laundering & KYC Policy") == (AML, "signature")
    assert r.resolve_name("Policy", "AML policy") == (AML, "catalog")
    assert r.resolve_name("Policy", "AML and KYC Policy") == (AML, "catalog")
    assert r.resolve_name("Policy", "Wire Transfer SOP") == ("SOP: Outgoing Wire Transfer Processing", "catalog")
    assert r.resolve_name("Regulation", "reg sho") == ("Regulation SHO", "alias")
    assert r.resolve_name("Regulation", "BSA", ["Bank Secrecy Act"]) == ("Bank Secrecy Act", "acronym")


def test_ambiguous_or_partial_names_are_not_merged():
    r = EntityResolver(catalog={"Policy": TITLES})
    # "risk management" is in two titles -> ambiguous, left alone rather than guessed
    assert r.resolve_name("Policy", "Risk Management Policy")[1] == "none"
    # subset matching is catalog-only and policy-only: departments never merge by containment
    assert r.resolve_name("Department", "Operations", ["Payments Operations"])[1] == "none"


def test_resolve_graph_merges_nodes_and_rewires_edges():
    nodes = [n("Policy", AML, "cmp_aml_kyc_policy"), n("Policy", "AML policy", "ops_wire_transfer_sop"),
             n("Policy", "Wire Transfer SOP", "ops_wire_transfer_sop"),
             n("Regulation", "Bank Secrecy Act"), n("Regulation", "BSA"),
             n("Department", "Compliance"), n("Department", "Compliance Department"), n("Department", "Compliance"),
             n("Department", "Payments Operations"), n("Department", "Operations")]
    edges = [e("Policy", AML, "GOVERNED_BY", "Regulation", "Bank Secrecy Act"),
             e("Policy", "AML policy", "GOVERNED_BY", "Regulation", "BSA"),  # duplicate after merge
             e("Policy", AML, "OWNED_BY", "Department", "Compliance Department"),
             e("Policy", "Wire Transfer SOP", "REQUIRES", "Policy", "AML policy"),
             e("Policy", "AML policy", "REQUIRES", "Policy", AML),  # becomes a self loop -> dropped
             e("Policy", "Wire Transfer SOP", "APPLIES_TO", "Department", "Payments Operations")]
    res = EntityResolver(catalog={"Policy": TITLES}).resolve(nodes, edges)

    ids = {x.node_id for x in res.nodes}
    aml, wire = node_id("Policy", AML), node_id("Policy", "SOP: Outgoing Wire Transfer Processing")
    assert {aml, wire, node_id("Regulation", "Bank Secrecy Act"), node_id("Department", "Compliance"),
            node_id("Department", "Payments Operations"), node_id("Department", "Operations")} == ids
    edge_ids = {x.edge_id for x in res.edges}
    assert f"{aml}|GOVERNED_BY|{node_id('Regulation', 'Bank Secrecy Act')}" in edge_ids
    assert f"{wire}|REQUIRES|{aml}" in edge_ids
    assert f"{aml}|OWNED_BY|{node_id('Department', 'Compliance')}" in edge_ids
    assert len(res.edges) == 4 and not res.rejected
    assert res.stats["nodes_in"] == 9 and res.stats["nodes_out"] == 6
    assert res.stats["by_method"] == {"catalog": 2, "acronym": 1, "signature": 1}
    rows = {r["alias_name"]: r for r in res.alias_rows()}
    assert rows["BSA"]["canonical_name"] == "Bank Secrecy Act" and rows["BSA"]["method"] == "acronym"


def test_resolution_is_idempotent():
    nodes = [n("Policy", "AML policy"), n("Regulation", "Bank Secrecy Act"), n("Regulation", "BSA")]
    edges = [e("Policy", "AML policy", "GOVERNED_BY", "Regulation", "BSA")]
    r = EntityResolver(catalog={"Policy": TITLES})
    first = r.resolve(nodes, edges)
    second = r.resolve(first.nodes, first.edges)
    assert {x.node_id for x in second.nodes} == {x.node_id for x in first.nodes}
    assert {x.edge_id for x in second.edges} == {x.edge_id for x in first.edges}
    assert second.aliases == []
