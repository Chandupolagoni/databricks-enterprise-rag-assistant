from ekb.kg_schema import KGEdge, KGNode, node_id
from ekb.kg_validate import find_evidence, mentions_entity, sources_from_passages, validate_facts

AML = "Anti-Money Laundering and KYC Policy"
AML_TEXT = ("## Purpose\nThis policy is owned by the **Compliance** department and is governed by the "
            "Bank Secrecy Act (BSA).\n\n## Scope\nIt applies to all relationship managers. Escalations go to "
            "aml.desk@meridian.example with the case reference.")
WIRE_TEXT = "## Controls\nBefore release, every outgoing wire requires the AML and KYC Policy screening checks."
SOURCES = sources_from_passages([
    {"doc_id": "aml", "version": 1, "text": AML_TEXT.split("## Scope")[0]},
    {"doc_id": "aml", "version": 1, "text": "## Scope" + AML_TEXT.split("## Scope")[1]},
    {"doc_id": "wire", "version": 1, "text": WIRE_TEXT}])


def _edge(src, rel, dst, ev, doc="aml", version=1):
    return KGEdge(node_id(*src), rel, node_id(*dst), doc, version, ev)


def test_find_evidence_exact_fuzzy_and_missing():
    assert find_evidence("owned by the Compliance department", AML_TEXT)[2] == "exact"   # markdown ignored
    assert find_evidence("and is governed by Bank Secrecy Act (BSA)", AML_TEXT)[2] == "fuzzy"  # dropped "the"
    found, score, method = find_evidence("This policy is governed by the Dodd-Frank Act.", AML_TEXT)
    assert not found and method == "not found" and score < 0.9
    # two sentence fragments stitched together are not a verbatim quote
    assert not find_evidence("This policy is governed by the Bank Secrecy Act (BSA)", AML_TEXT)[0]
    assert not find_evidence("the BSA", "governed by the B.S.A.")[0]                 # short spans: exact only


def test_masked_evidence_matches_raw_source():
    # the extractor only ever saw the PII-masked passage
    assert find_evidence("Escalations go to [EMAIL] with the case reference", AML_TEXT)[0]


def test_target_must_be_mentioned_in_the_span():
    assert mentions_entity("governed by the BSA", "Bank Secrecy Act")
    assert mentions_entity("requires the AML and KYC Policy screening", AML)
    assert not mentions_entity("It applies to all relationship managers", "Wealth Management")


def test_validate_facts_keeps_only_grounded_edges():
    pol, wire = ("Policy", AML), ("Policy", "SOP: Outgoing Wire Transfer Processing")
    edges = [
        _edge(pol, "OWNED_BY", ("Department", "Compliance"), "owned by the Compliance department"),
        _edge(pol, "GOVERNED_BY", ("Regulation", "Bank Secrecy Act"), "is governed by the Bank Secrecy Act (BSA)"),
        _edge(pol, "GOVERNED_BY", ("Regulation", "Dodd-Frank Act"), "is governed by the Dodd-Frank Act"),  # invented
        _edge(pol, "APPLIES_TO", ("Department", "Wealth Management"), "It applies to all relationship managers"),
        _edge(wire, "REQUIRES", pol, "every outgoing wire requires the AML and KYC Policy screening checks", "wire"),
        _edge(pol, "OWNED_BY", ("Department", "Compliance"), "owned by the Compliance department", version=2),
    ]
    nodes = [KGNode("Policy", AML, "aml", 1, "This policy is owned by the Compliance department"),
             KGNode("Department", "Compliance", "aml", 1, "owned by the Compliance department"),
             KGNode("Regulation", "Dodd-Frank Act", "aml", 1, "governed by the Dodd-Frank Act"),
             KGNode("Department", "Wealth Management", "aml", 1, "Wealth Management")]
    res = validate_facts(edges, nodes, SOURCES)
    assert [e.dst_id for e in res.edges] == [node_id("Department", "Compliance"),
                                            node_id("Regulation", "Bank Secrecy Act"), node_id(*pol)]
    assert res.stats["edges_in"] == 6 and res.stats["edges_kept"] == 3 and res.stats["edge_reject_rate"] == 0.5
    assert {n.name for n in res.nodes} == {AML, "Compliance"}
    reasons = " | ".join(r["reasons"] for r in res.reject_rows())
    assert "evidence not in source" in reasons and "does not mention target department:wealth_management" in reasons
    assert "source text not found for aml v2" in reasons


def test_target_check_can_be_disabled():
    e = _edge(("Policy", AML), "APPLIES_TO", ("EmployeeRole", "Relationship Manager"),
              "It applies to all relationship managers")
    assert validate_facts([e], [], SOURCES, check_target=False).stats["edges_kept"] == 1
