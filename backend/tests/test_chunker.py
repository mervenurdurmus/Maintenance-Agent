from app.services.chunker import chunk_text


def test_chunk_text_returns_stable_chunk_ids() -> None:
    chunks = chunk_text("A" * 1200, document_id="doc_test", chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "doc_test_c1"
