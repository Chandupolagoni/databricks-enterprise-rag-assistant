from ekb.chunking import count_tokens, hashing_embed, hybrid_chunk, semantic_segments, sliding_window, split_sentences
from ekb.parsing import ParsedDocument, Section, clean_text, parse_file
from ekb.pipeline import read_metadata


def test_every_format_parses_with_sections(root):
    meta = read_metadata(root / "data" / "metadata" / "documents.csv")
    assert {m["file_format"] for m in meta} == {"pdf", "docx", "html", "md", "txt"}
    for m in meta:
        doc = parse_file(root / "data" / "raw" / m["relative_path"], m["doc_id"])
        assert doc.title, m["doc_id"]
        assert len(doc.sections) >= 3, m["doc_id"]
        assert all(s.text for s in doc.sections)


def test_html_boilerplate_removed(root):
    doc = parse_file(root / "data" / "raw" / "compliance" / "cmp_complaints_manual.html")
    assert "intranet" not in doc.text and "Generated sample" not in doc.text


def test_clean_text_dehyphenates_and_collapses_space():
    assert clean_text("recon-\nciliation  of   breaks") == "reconciliation of breaks"


def test_sentence_split():
    assert split_sentences("Wires settle T+1. Fails older than 5 days escalate. Done!") == [
        "Wires settle T+1.", "Fails older than 5 days escalate.", "Done!"]


def test_sliding_window_respects_size_and_overlaps():
    sents = [f"Sentence number {i} talks about wire cut off times and approvals." for i in range(30)]
    windows = sliding_window(sents, max_tokens=60, overlap_tokens=15)
    assert len(windows) > 1
    assert all(sum(count_tokens(s) for s in w) <= 60 + 15 for w in windows)
    assert windows[0][-1] == windows[1][0]          # overlap carried forward


def test_semantic_segments_break_on_topic_shift():
    sents = ["Wire transfers need callback verification."] * 3 + ["Parental leave is sixteen weeks for caregivers."] * 3
    segs = semantic_segments(sents, hashing_embed, breakpoint_percentile=50)
    assert any(all("Wire" in s for s in seg) for seg in segs)
    assert len(segs) >= 2


def test_hybrid_chunk_keeps_sections_and_header():
    doc = ParsedDocument("d1", "Policy", "md", [Section("Limits", "Gifts up to 100 USD."), Section("Approval", "Ask Compliance.")])
    chunks = hybrid_chunk(doc, metadata={"acl_tag": "public"})
    assert [c.section for c in chunks] == ["Limits", "Approval"]
    assert chunks[0].content.startswith("Policy > Limits")
    assert chunks[0].chunk_id == "d1::000" and chunks[0].metadata["acl_tag"] == "public"


def test_long_section_is_split_into_multiple_chunks():
    long = " ".join(f"Rule {i} says reconciliation breaks older than {i} days escalate to risk." for i in range(60))
    doc = ParsedDocument("d2", "Recon", "md", [Section("Ageing", long)])
    chunks = hybrid_chunk(doc, max_tokens=120, overlap_tokens=20)
    assert len(chunks) > 3 and all(c.n_tokens <= 140 for c in chunks)


def test_corpus_chunk_table(chunks):
    assert len(chunks) >= 70
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    required = {"chunk_id", "doc_id", "title", "section", "text", "content", "acl_tag", "access_level", "relative_path"}
    assert required <= set(chunks[0])
