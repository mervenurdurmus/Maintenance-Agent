import logging
from unittest.mock import MagicMock

from app.services.chunker import Chunk
from app.services.vector_store import VectorStore


def test_search_returns_results_without_score_filtering(caplog):
    store = VectorStore.__new__(VectorStore)

    store.embedding_model = MagicMock()
    store.collection = MagicMock()
    store.model_name = "test-model"

    store.embedding_model.embed_query.return_value = [0.1, 0.2]
    store.collection.count.return_value = 2

    store.collection.query.return_value = {
        "documents": [["ilgili chunk", "alakasız chunk"]],
        "metadatas": [[
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
        ]],
        "distances": [[0.2, 0.7]],
    }

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        matches = store.search(
            query="P204 alarmı",
            top_k=2,
        )

    assert len(matches) == 2
    assert matches[0]["metadata"]["chunk_id"] == "c1"
    assert "embedding_search.started" in caplog.text
    assert "query='P204 alarmı'" in caplog.text
    assert "chunk_id" in caplog.text
def test_search_returns_empty_when_collection_is_empty():
    store = VectorStore.__new__(VectorStore)

    store.embedding_model = MagicMock()
    store.collection = MagicMock()
    store.collection.count.return_value = 0

    matches = store.search(
        query="P204 alarmı",
        top_k=3,
    )

    assert matches == []
    store.embedding_model.embed_query.assert_not_called()
def test_add_chunks_preserves_chunk_metadata():
    store = VectorStore.__new__(VectorStore)

    store.model_name = "test-model"
    store.embedding_model = MagicMock()
    store.collection = MagicMock()

    store.embedding_model.embed_documents.return_value = [[0.1, 0.2]]

    chunk = Chunk(
        chunk_id="doc_pdf_c4",
        text="Sayfa metni",
        metadata={
            "page_number": 2,
            "document_title": "Bakım Kılavuzu",
        },
    )

    store.add_chunks(
        document_id="doc_pdf",
        document_name="bakim.pdf",
        chunks=[chunk],
    )

    saved_metadata = store.collection.upsert.call_args.kwargs["metadatas"][0]

    assert saved_metadata["page_number"] == 2
    assert saved_metadata["document_title"] == "Bakım Kılavuzu"
    assert saved_metadata["document_id"] == "doc_pdf"
    assert saved_metadata["document_name"] == "bakim.pdf"
    assert saved_metadata["chunk_id"] == "doc_pdf_c4"

def test_delete_document_deletes_chunks_by_document_id():
    store = VectorStore.__new__(VectorStore)

    store.collection = MagicMock()

    store.delete_document("doc_123")

    store.collection.delete.assert_called_once_with(
        where={
            "document_id": "doc_123",
        }
    )   
