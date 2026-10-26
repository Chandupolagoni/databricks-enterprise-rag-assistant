import pytest

from ekb.assistant import KnowledgeAssistant
from ekb.llm import EchoLLM
from ekb.parent_child import ParentExpander, assign_parents
from ekb.rerank import LexicalReranker
from ekb.retrieval import LocalHybridRetriever


def _row(doc, idx, section, text, acl="public"):
    return {"chunk_id": f"{doc}::{idx:03d}", "doc_id": doc, "chunk_index": idx, "section": section,
            "text": text, "content": f"{doc} > {section}\n{text}", "n_tokens": len(text.split()),
            "title": doc, "acl_tag": acl}


@pytest.fixture(scope="module")
def corpus_parents(chunks):
    rows = [dict(c) for c in chunks]
    return rows, assign_parents(rows, max_parent_tokens=400)


def test_every_chunk_gets_a_parent_within_its_document(corpus_parents):
    rows, parents = corpus_parents
    by_id = {p["parent_id"]: p for p in parents}
    for r in rows:
        p = by_id[r["parent_id"]]
        assert p["doc_id"] == r["doc_id"] and r["chunk_id"] in p["chunk_ids"]
        assert p["acl_tag"] == r["acl_tag"]          # one access level per parent


def test_parents_respect_token_budget():
    rows = [_row("d", i, f"S{i}", " ".join(["word"] * 40) + ".") for i in range(6)]
    parents = assign_parents(rows, max_parent_tokens=120)
    assert len(parents) > 1
    assert all(p["n_tokens"] <= 120 or len(p["chunk_ids"]) == 1 for p in parents)
    assert [r["parent_id"] for r in rows][:2] == ["d::p00", "d::p00"]


def test_sliding_window_overlap_is_not_duplicated():
    rows = [_row("d", 0, "Limits", "Wires above 1M need two approvers. Callbacks are mandatory."),
            _row("d", 1, "Limits", "Callbacks are mandatory. Use the number on file.")]
    (parent,) = assign_parents(rows)
    assert parent["text"].count("Callbacks are mandatory.") == 1
    assert parent["text"].startswith("## Limits\n")


def test_expander_dedupes_siblings_and_keeps_rank():
    rows = [_row("a", 0, "S1", "Alpha one."), _row("a", 1, "S2", "Alpha two."), _row("b", 0, "S1", "Beta.")]
    exp = ParentExpander(assign_parents(rows))
    out = exp.expand([rows[1], rows[2], rows[0]])
    assert [c["doc_id"] for c in out] == ["a", "b"]
    assert out[0]["matched_chunk_ids"] == ["a::001", "a::000"]
    assert "Alpha one." in out[0]["text"] and out[0]["child_text"] == "Alpha two."


def test_expander_falls_back_to_child_when_parent_exceeds_budget():
    rows = [_row("a", i, f"S{i}", " ".join(["policy"] * 50) + ".") for i in range(4)]
    exp = ParentExpander(assign_parents(rows, max_parent_tokens=1000), max_context_tokens=100)
    (only,) = exp.expand([rows[0]])
    assert only["text"] == rows[0]["text"] and "child_text" not in only


def test_expander_never_widens_access():
    rows = [_row("a", 0, "S1", "Public part.")]
    parents = assign_parents(rows)
    parents[0]["acl_tag"] = "human_resources:restricted"     # tampered / mismatched parent
    (out,) = ParentExpander(parents).expand([rows[0]])
    assert out["text"] == "Public part."


def test_parent_brings_in_neighbouring_section():
    rows = [_row("w", 0, "Cut-off", "Same-day Fedwire payments must be released before the cut-off."),
            _row("w", 1, "Cut-off time", "The Fedwire cut-off is 4:30 PM Eastern.")]
    (parent,) = assign_parents(rows)
    (out,) = ParentExpander([parent]).expand([rows[0]])
    assert "4:30 PM" in out["text"] and "4:30 PM" not in rows[0]["text"]


def test_assistant_with_parent_expansion(chunks, users):
    rows = [dict(c) for c in chunks]
    llm = EchoLLM()
    assistant = KnowledgeAssistant(LocalHybridRetriever(rows), LexicalReranker(), llm,
                                   context_expander=ParentExpander(assign_parents(rows)))
    r = assistant.ask("What is the cut-off time for same-day Fedwire payments?", users["u_payments_ops"])
    assert r.status == "answered" and "4:30 PM" in r.answer
    assert [c.doc_id for c in r.citations] == ["ops_wire_transfer_sop"]
    assert all("::p" not in cid for cid in r.retrieved_chunk_ids)   # children are still reported
    assert "## " in llm.last_messages[-1]["content"]                # parent passage was sent
