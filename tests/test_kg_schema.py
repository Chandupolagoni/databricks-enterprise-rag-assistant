import pytest

from ekb.kg_schema import (
    ENTITY_TYPES,
    RELATIONS,
    KGEdge,
    KGNode,
    entity_type_of,
    kg_table_ddl,
    node_id,
    normalize_name,
    schema_prompt,
    validate_edge,
    validate_graph,
)

AML = "Anti-Money Laundering and KYC Policy"
EV_AML = "The policy is owned by the Compliance department and is governed by the Bank Secrecy Act"
EV_WIRE = "This SOP requires the AML and KYC Policy."


def _n(t, name, doc="cmp_aml_kyc_policy", ev=EV_AML):
    return KGNode(t, name, doc, 1, ev)


def _e(src, rel, dst, doc="cmp_aml_kyc_policy", ev=EV_AML):
    return KGEdge(src, rel, dst, doc, 1, ev)


def test_node_ids_are_deterministic_and_typed():
    assert normalize_name("  Anti-Money  Laundering & KYC Policy ") == "anti money laundering and kyc policy"
    assert node_id("Policy", AML) == "policy:anti_money_laundering_and_kyc_policy"
    assert node_id("Regulation", "SEC Rule 204A-1") == "regulation:sec_rule_204a_1"
    assert entity_type_of(node_id("EmployeeRole", "Access persons")) == "EmployeeRole"
    with pytest.raises(ValueError):
        node_id("Rule", "x")


def test_every_relation_uses_known_entity_types():
    for pairs in RELATIONS.values():
        assert pairs and all(s in ENTITY_TYPES and d in ENTITY_TYPES for s, d in pairs)


def test_valid_edges_pass():
    pol, bsa = node_id("Policy", AML), node_id("Regulation", "Bank Secrecy Act")
    comp = node_id("Department", "Compliance")
    assert validate_edge(_e(pol, "GOVERNED_BY", bsa), {pol, bsa}) == []
    assert validate_edge(_e(pol, "OWNED_BY", comp)) == []
    wire = node_id("TransactionType", "Outgoing wire transfer")
    assert validate_edge(_e(wire, "REQUIRES", pol, "ops_wire_transfer_sop", EV_WIRE)) == []


@pytest.mark.parametrize("src,rel,dst,reason", [
    ("regulation:bank_secrecy_act", "GOVERNED_BY", "policy:aml", "not allowed from Regulation to Policy"),
    ("policy:aml", "OWNED_BY", "employeerole:fiu_analyst", "not allowed from Policy to EmployeeRole"),
    ("policy:aml", "IS_SUBJECT_TO", "regulation:bsa", "unknown relation"),
    ("policy:aml", "REQUIRES", "policy:aml", "self loop"),
    ("rule:x", "REQUIRES", "policy:aml", "not allowed from None to Policy"),
])
def test_schema_violations_are_rejected(src, rel, dst, reason):
    assert any(reason in r for r in validate_edge(_e(src, rel, dst)))


def test_provenance_is_mandatory():
    errs = validate_edge(KGEdge("policy:a", "REQUIRES", "policy:b", "", 1, "  "))
    assert "missing doc_id" in errs and "missing evidence span" in errs
    assert "evidence span too long" in validate_edge(_e("policy:a", "REQUIRES", "policy:b", ev="x" * 501))


def test_validate_graph_splits_and_dedupes():
    nodes = [_n("Policy", AML), _n("Policy", "anti-money laundering & KYC policy"),  # same id after normalising
             _n("Regulation", "Bank Secrecy Act"), _n("Department", "Compliance"),
             _n("Regulator", "FinCEN")]                                               # unknown type
    pol, bsa = node_id("Policy", AML), node_id("Regulation", "Bank Secrecy Act")
    edges = [_e(pol, "GOVERNED_BY", bsa), _e(pol, "GOVERNED_BY", bsa),                # duplicate
             _e(pol, "OWNED_BY", node_id("Department", "Compliance")),
             _e(pol, "GOVERNED_BY", "regulation:usa_patriot_act")]                    # endpoint never extracted
    g = validate_graph(nodes, edges)
    assert len(g.nodes) == 3 and len(g.edges) == 2
    assert [r for _, rs in g.rejected for r in rs] == ["unknown entity type 'Regulator'",
                                                       "missing node regulation:usa_patriot_act"]
    assert g.reject_rate == round(2 / 7, 4)
    assert g.edges[0].as_row()["edge_id"] == f"{pol}|GOVERNED_BY|{bsa}"


def test_prompt_and_ddl_cover_the_whole_schema():
    prompt = schema_prompt()
    assert all(t in prompt for t in ENTITY_TYPES) and all(r in prompt for r in RELATIONS)
    nodes_ddl, edges_ddl = kg_table_ddl("workspace.ekb.silver_kg_nodes", "workspace.ekb.silver_kg_edges")
    assert "silver_kg_nodes" in nodes_ddl and "'TransactionType'" in nodes_ddl and "evidence STRING NOT NULL" in nodes_ddl
    assert "silver_kg_edges" in edges_ddl and "'REQUIRES'" in edges_ddl and "doc_id STRING NOT NULL" in edges_ddl
