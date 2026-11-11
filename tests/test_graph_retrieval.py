import json
import re

import pytest

from ekb.assistant import KnowledgeAssistant
from ekb.evaluation import context_recall, load_golden, question_tags
from ekb.graph_retrieval import GraphExpander, edges_from_fixture
from ekb.kg_validate import find_evidence, sources_from_passages
from ekb.llm import EchoLLM
from ekb.rerank import LexicalReranker
from ekb.retrieval import LocalHybridRetriever


@pytest.fixture(scope="module")
def fixture_rows(root):
    return json.loads((root / "data" / "kg" / "policy_graph_fixture.json").read_text())["edges"]


@pytest.fixture(scope="module")
def expander(chunks, fixture_rows):
    return GraphExpander(edges_from_fixture(fixture_rows), chunks, {c["doc_id"]: c["title"] for c in chunks})


def test_fixture_edges_are_grounded_in_the_corpus(chunks, fixture_rows):
    sources = sources_from_passages(chunks)
    for r in fixture_rows:
        assert find_evidence(r["evidence"], sources[(r["doc_id"], 1)])[0], r


def test_one_hop_links_follow_rank_order(expander):
    links = expander.linked_docs(["wm_client_onboarding_sop", "tech_genai_usage_policy", "cmp_aml_kyc_policy"])
    assert links == [("wm_client_onboarding_sop", "REQUIRES", "wm_suitability_policy"),
                     ("tech_genai_usage_policy", "GOVERNED_BY", "risk_model_risk_policy")]   # AML already a seed
    assert expander.linked_docs(["hr_leave_policy"]) == []


def test_expand_adds_linked_policy_passage(expander, chunks, users):
    seed = [c for c in chunks if c["doc_id"] == "wm_client_onboarding_sop"][:2]
    out = expander.expand("Which suitability checks apply before opening an account?", seed, users["u_wm_advisor"])
    added = out[len(seed):]
    assert {c["doc_id"] for c in added} == {"cmp_aml_kyc_policy", "wm_suitability_policy"}
    assert all(c["graph_link"].startswith("SOP: Wealth Client Onboarding REQUIRES ") for c in added)
    assert out[:2] == seed                                         # retrieved chunks keep their order
    capped = GraphExpander(edges_from_fixture([]), chunks, {}, max_added=0)
    assert capped.expand("q", seed, users["u_wm_advisor"]) == seed


def test_expansion_never_widens_rbac(expander, chunks, users):
    ops = users["u_payments_ops"]                                  # operations only: cannot read compliance:internal
    seed = [c for c in chunks if c["doc_id"] == "ops_wire_transfer_sop"][:1]
    assert ops.is_authorized(seed[0])
    assert expander.expand("What does the wire SOP require?", seed, ops) == seed


def test_token_budget_limits_additions(fixture_rows, chunks, users):
    tight = GraphExpander(edges_from_fixture(fixture_rows), chunks, {c["doc_id"]: c["title"] for c in chunks},
                          max_added_tokens=1)
    seed = [c for c in chunks if c["doc_id"] == "wm_client_onboarding_sop"][:1]
    assert tight.expand("account opening", seed, users["u_wm_advisor"]) == seed


def test_graph_expansion_lifts_context_recall(root, chunks, users, expander):
    golden = [g for g in load_golden(root / "data" / "eval" / "golden_questions.jsonl") if g.get("relevant_doc_ids")]
    retriever, reranker = LocalHybridRetriever(chunks), LexicalReranker()
    before, after, multi = [], [], []
    for g in golden:
        user = users[g["user_id"]]
        top = reranker.rerank(g["question"], retriever.search(g["question"], user=user, k=20), top_n=5)
        rel = set(g["relevant_doc_ids"])
        before.append(context_recall([c["doc_id"] for c in top], rel))
        after.append(context_recall([c["doc_id"] for c in expander.expand(g["question"], top, user)], rel))
        if "multi_doc" in question_tags(g):
            multi.append(after[-1])
    assert all(a >= b for a, b in zip(after, before))
    assert sum(after) / len(after) > sum(before) / len(before)     # 0.962 -> 1.000 at top-5
    assert sum(multi) / len(multi) == 1.0


def test_assistant_sends_linked_policies_to_the_llm(chunks, expander, users):
    q = "What does the wealth client onboarding SOP require before an account is opened?"
    plain, graph = EchoLLM(), EchoLLM()
    KnowledgeAssistant(LocalHybridRetriever(chunks), LexicalReranker(), plain).ask(q, users["u_wm_advisor"])
    resp = KnowledgeAssistant(LocalHybridRetriever(chunks), LexicalReranker(), graph,
                              graph_expander=expander).ask(q, users["u_wm_advisor"])
    sent = lambda llm: set(re.findall(r'<source id="([^"]+)"', llm.last_messages[-1]["content"]))  # noqa: E731
    assert sent(graph) > sent(plain)                     # strictly more documents reach the LLM
    assert resp.status == "answered"
