import pytest

from ekb.evaluation import (
    evaluate_retrieval_by_type,
    filter_golden,
    load_document_acl,
    load_golden,
    question_type,
    validate_golden,
)
from ekb.retrieval import LocalHybridRetriever


@pytest.fixture(scope="module")
def golden(root):
    return load_golden(root / "data" / "eval" / "golden_questions.jsonl")


@pytest.fixture(scope="module")
def doc_acl(root):
    return load_document_acl(root / "data" / "metadata" / "documents.csv")


def test_golden_set_is_consistent(golden, users, doc_acl):
    assert validate_golden(golden, users, doc_acl) == []


def test_golden_set_covers_all_slices(golden):
    assert len(golden) == 40
    assert len(filter_golden(golden, "paraphrase")) >= 8
    multi = filter_golden(golden, "multi_doc")
    assert len(multi) >= 7 and all(len(set(g["relevant_doc_ids"])) >= 2 for g in multi)
    assert question_type({"id": "x"}) == "single_doc"
    assert len(filter_golden(golden, ("paraphrase", "multi_doc"))) == 15


def test_validate_golden_flags_labelling_mistakes(users, doc_acl):
    bad = [
        {"id": "a", "user_id": "u_engineer", "question": "q", "relevant_doc_ids": ["hr_compensation_bands"],
         "expected_facts": ["3-year"], "expect": "answer"},
        {"id": "a", "user_id": "u_ghost", "question": "q", "relevant_doc_ids": ["no_such_doc"],
         "expected_facts": ["x"], "expect": "answer"},
        {"id": "b", "user_id": "u_new_joiner", "question": "q", "relevant_doc_ids": ["hr_leave_policy"],
         "expected_facts": [], "expect": "answer", "type": "multi_doc"},
        {"id": "c", "user_id": "u_new_joiner", "question": "q", "relevant_doc_ids": ["hr_leave_policy"],
         "expected_facts": [], "expect": "refuse", "type": "typo"},
        {"id": "d", "user_id": "u_new_joiner", "question": "q", "relevant_doc_ids": [], "expect": "maybe"},
    ]
    problems = "\n".join(validate_golden(bad, users, doc_acl))
    assert "u_engineer is not entitled to hr_compensation_bands" in problems
    assert "a: duplicate id" in problems
    assert "unknown user 'u_ghost'" in problems and "unknown document 'no_such_doc'" in problems
    assert "b: multi_doc question needs at least two" in problems
    assert "b: answerable question without expected_facts" in problems
    assert "c: unknown type 'typo'" in problems and "c: refuse question must not list" in problems
    assert "d: unknown expect 'maybe'" in problems


def test_retrieval_by_type_reports_each_slice(chunks, users, golden):
    reports = evaluate_retrieval_by_type(LocalHybridRetriever(chunks), golden, users, k=5)
    assert set(reports) == {"all", "single_doc", "paraphrase", "multi_doc"}
    assert reports["single_doc"].n == 20 and reports["paraphrase"].n == 8 and reports["multi_doc"].n == 7
    assert reports["all"].n == sum(reports[t].n for t in ("single_doc", "paraphrase", "multi_doc"))
    # Both source documents of every multi-doc question are in the top 5.
    assert reports["multi_doc"].context_recall == 1.0
    # Hybrid recovers paraphrases that keyword search alone misses.
    keyword = evaluate_retrieval_by_type(LocalHybridRetriever(chunks), golden, users, k=5, mode="keyword")
    assert reports["paraphrase"].hit_rate == 1.0 > keyword["paraphrase"].hit_rate
