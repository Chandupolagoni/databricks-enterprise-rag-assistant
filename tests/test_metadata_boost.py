import pytest

from ekb.evaluation import evaluate_retrieval, load_golden, reciprocal_rank
from ekb.metadata_boost import MetadataAwareRetriever, detect_category, latest_versions
from ekb.rerank import LexicalReranker
from ekb.retrieval import LocalHybridRetriever


class _Static:
    """Retriever stub returning fixed rows in a fixed order."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, question, user=None, k=20, mode="hybrid"):
        self.calls.append(k)
        return [dict(r) for r in self.rows[:k]]


def _row(cid, doc, version=1, category="sop"):
    return {"chunk_id": cid, "doc_id": doc, "version": version, "category": category}


@pytest.mark.parametrize("q,cat", [
    ("According to the wire transfer SOP, who approves a recall?", "sop"),
    ("Is there a KB article on failed DVP settlements?", "knowledge_base"),
    ("What does the complaints handling manual say about escalation?", "complaints_manual"),
    ("Which policy document covers gifts?", "financial_policy"),
    ("What does our policy say about gifts?", None),            # bare "policy" is not a trigger
    ("Does the SOP or the knowledge base cover this?", None),   # ambiguous -> no filter
])
def test_detect_category(q, cat):
    assert detect_category(q) == cat


def test_latest_versions():
    assert latest_versions([_row("a", "d1", 1), _row("b", "d1", 3), _row("c", "d2", "2")]) == {"d1": 3, "d2": 2}


def test_superseded_versions_are_demoted_not_lost():
    base = _Static([_row("old-1", "wire", 1), _row("new-1", "wire", 2), _row("other", "kb", 1)])
    r = MetadataAwareRetriever(base)
    out = r.search("cut-off time?", k=3)
    assert [c["chunk_id"] for c in out] == ["new-1", "other", "old-1"]
    assert out[-1]["superseded"] and r.last_report.superseded == ["old-1"]
    assert base.calls == [6]                                     # oversampled


def test_drop_superseded_and_catalog_versions():
    # version 2 exists in the catalog but was not retrieved: version 1 is still stale.
    base = _Static([_row("old-1", "wire", 1), _row("other", "kb", 1)])
    r = MetadataAwareRetriever(base, catalog_versions={"wire": 2}, drop_superseded=True)
    assert [c["chunk_id"] for c in r.search("q", k=5)] == ["other"]


def test_category_filter_with_safe_fallback():
    rows = [_row("p1", "pol", category="financial_policy"), _row("s1", "sop1"), _row("s2", "sop2")]
    r = MetadataAwareRetriever(_Static(rows))
    assert [c["chunk_id"] for c in r.search("per the SOP, what next?", k=5)] == ["s1", "s2"]
    assert r.last_report.category_applied
    # only one KB hit (< min_category_hits): keep the unfiltered ranking
    out = r.search("is there a knowledge base entry?", k=5)
    assert len(out) == 3 and not r.last_report.category_applied


def test_rbac_is_never_widened(chunks, users):
    user = users["u_new_joiner"]
    allowed = set(user.allowed_tags())
    r = MetadataAwareRetriever(LocalHybridRetriever(chunks))
    out = r.search("What does the SOP say about wire transfers?", user=user, k=10)
    assert all(c["acl_tag"] in allowed for c in out)


def test_category_named_questions_rank_better(chunks, users):
    base = LocalHybridRetriever(chunks)
    boosted = MetadataAwareRetriever(base)
    cases = [  # (user, question naming a category, doc that should come first)
        ("u_payments_ops", "In the SOP, what checks apply before releasing a wire?", "ops_wire_transfer_sop"),
        ("u_fiu_analyst", "Per the SOP, how is a suspicious activity report filed?", "cmp_sar_filing_sop"),
        ("u_new_joiner", "What does the complaints handling manual say about acknowledging complaints?",
         "cmp_complaints_manual"),
    ]

    def mrr(ret):
        return sum(reciprocal_rank([c["doc_id"] for c in ret.search(q, user=users[u], k=5)], {d})
                   for u, q, d in cases) / len(cases)

    assert mrr(boosted) >= mrr(base)
    assert mrr(boosted) >= 0.8


def test_no_regression_on_golden_set(chunks, users, root):
    golden = load_golden(root / "data" / "eval" / "golden_questions.jsonl")
    base = LocalHybridRetriever(chunks)
    before = evaluate_retrieval(base, golden, users, k=5, reranker=LexicalReranker())
    after = evaluate_retrieval(MetadataAwareRetriever(base), golden, users, k=5, reranker=LexicalReranker())
    assert after.hit_rate >= before.hit_rate and after.mrr >= before.mrr - 1e-9
