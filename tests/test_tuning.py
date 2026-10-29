import pytest

from ekb.evaluation import load_golden
from ekb.retrieval import LocalHybridRetriever
from ekb.tuning import RRFTrial, as_markdown, grid_search_rrf, leave_one_out_rrf


@pytest.fixture(scope="module")
def setup(chunks, users, root):
    return LocalHybridRetriever(chunks), load_golden(root / "data" / "eval" / "golden_questions.jsonl"), users


def test_tie_break_prefers_balanced_weights():
    a = RRFTrial(3.0, 1.0, 60, 1.0, 0.5, 0.9)
    b = RRFTrial(1.0, 1.0, 60, 1.0, 0.5, 0.9)
    c = RRFTrial(1.0, 1.0, 20, 1.0, 0.5, 0.9)
    assert sorted([a, c, b], key=RRFTrial.sort_key) == [b, c, a]
    assert sorted([b, RRFTrial(3.0, 1.0, 60, 1.0, 0.5, 0.95)], key=RRFTrial.sort_key)[0].keyword_weight == 3.0


def test_grid_search_ranks_and_restores_settings(setup):
    retriever, golden, users = setup
    trials = grid_search_rrf(retriever, golden, users, weight_grid=[(1.0, 1.0), (3.0, 1.0)], rrf_ks=[60])
    assert len(trials) == 2 and trials[0].mrr >= trials[1].mrr
    assert retriever.weights == (1.0, 1.0) and retriever.rrf_k == 60
    balanced = next(t for t in trials if t.keyword_weight == 1.0)
    assert balanced.hit_rate == 1.0                        # matches the README's hybrid row
    assert "| w_keyword |" in as_markdown(trials)


def test_extreme_weights_converge_to_single_retriever(setup):
    retriever, golden, users = setup
    kw_heavy = grid_search_rrf(retriever, golden, users, weight_grid=[(100.0, 1.0)], rrf_ks=[60])[0]
    vec_heavy = grid_search_rrf(retriever, golden, users, weight_grid=[(1.0, 100.0)], rrf_ks=[60])[0]
    assert kw_heavy.mrr > vec_heavy.mrr                     # the golden set is token-heavy


def test_leave_one_out_reports_generalisation(setup):
    retriever, golden, users = setup
    out = leave_one_out_rrf(retriever, golden, users, weight_grid=[(1.0, 1.0), (3.0, 1.0)], rrf_ks=[60])
    assert 0.0 < out["loo_mrr"] <= 1.0
    assert sum(out["selected"].values()) == len([g for g in golden if g.get("relevant_doc_ids")])
