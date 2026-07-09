import logging
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.services import conversation_store
from app.api.routes import router
from app.core.config import get_settings
from app.main import app
from app.models.schemas import ChatResponse, ToolCall


def test_conversation_endpoints_create_list_and_load_history(monkeypatch, tmp_path):
    monkeypatch.setattr(conversation_store, "DB_PATH", tmp_path / "history.db")
    client = TestClient(app)

    create_response = client.post("/api/conversations")

    assert create_response.status_code == 200
    conversation_id = create_response.json()["conversation_id"]
    assert conversation_id.startswith("chat_")

    history = conversation_store.get_chat_history(conversation_id)
    history.add_messages(
        [
            HumanMessage(content="Selam"),
            AIMessage(content="Merhaba!"),
        ]
    )
    conversation_store.save_tool_calls(
        conversation_id,
        [
            ToolCall(
                name="semantic_search",
                input={"query": "P204"},
                output={"matches": [{"chunk_id": "doc_1_c3"}]},
            )
        ],
    )

    list_response = client.get("/api/conversations")
    messages_response = client.get(
        f"/api/conversations/{conversation_id}/messages"
    )
    tool_calls_response = client.get(
        f"/api/conversations/{conversation_id}/tool-calls"
    )

    assert list_response.status_code == 200
    assert list_response.json()[0]["conversation_id"] == conversation_id
    assert messages_response.status_code == 200
    assert [item["content"] for item in messages_response.json()] == [
        "Selam",
        "Merhaba!",
    ]
    assert tool_calls_response.status_code == 200
    assert tool_calls_response.json()[0]["name"] == "semantic_search"
    assert tool_calls_response.json()[0]["input"] == {"query": "P204"}

    delete_response = client.delete(f"/api/conversations/{conversation_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "conversation_id": conversation_id,
        "status": "deleted",
    }
    assert client.get(
        f"/api/conversations/{conversation_id}/messages"
    ).status_code == 404
    assert conversation_store.list_history_messages(conversation_id) == []
    assert conversation_store.list_tool_calls(conversation_id) == []


def test_chat_requires_a_conversation_id():
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "Selam"})

    assert response.status_code == 422


def test_upload_removes_file_when_embedding_service_fails(
    monkeypatch,
    tmp_path,
):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    vector_store = MagicMock()
    vector_store.add_chunks.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(
        "app.api.routes.get_vector_store",
        lambda: vector_store,
    )
    client = TestClient(app)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("bakim.txt", b"P204 alarm bilgisi", "text/plain")},
    )

    assert response.status_code == 503
    assert "Embedding servisi kullanılamıyor" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_chat_logs_the_user_query(monkeypatch, caplog):
    monkeypatch.setattr(
        "app.api.routes.answer_message",
        lambda _message, conversation_id: ChatResponse(
            answer="Yanıt",
            sources=[],
            tool_calls=[],
        ),
    )

    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = client.post(
            "/api/chat",
            json={
                "conversation_id": "test-session",
                "message": "P204 alarm kodu",
            },
        )

    assert response.status_code == 200
    assert "chat.query" in caplog.text
    assert "conversation_id=test-session" in caplog.text
    assert "query='P204 alarm kodu'" in caplog.text


def test_chat_with_image_saves_attachment(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "chat_image_dir", tmp_path)

    def fake_answer_message(
        message,
        conversation_id,
        attachments=None,
        image_descriptions=None,
    ):
        return ChatResponse(
            answer="Görsel alındı.",
            sources=[],
            tool_calls=[],
            attachments=attachments or [],
        )

    monkeypatch.setattr("app.api.routes.answer_message", fake_answer_message)
    monkeypatch.setattr("app.api.routes.describe_image", lambda *_args: "Görsel özeti")
    client = TestClient(app)

    response = client.post(
        "/api/chat/image",
        data={
            "message": "Bu görseli ekle",
            "conversation_id": "test-session",
        },
        files={
            "image": ("makine.png", b"fake-png-content", "image/png"),
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["attachments"][0]["filename"] == "makine.png"
    assert payload["attachments"][0]["content_type"] == "image/png"
    assert payload["attachments"][0]["url"].startswith("/chat-images/img_")
    assert len(list(tmp_path.iterdir())) == 1


def test_delete_document_removes_file_and_vectors(monkeypatch, tmp_path):
    settings = get_settings()
    settings.upload_dir = tmp_path

    document_id = "doc_test123"
    uploaded_file = tmp_path / f"{document_id}_bakim.pdf"
    uploaded_file.write_text("test content", encoding="utf-8")

    vector_store = MagicMock()

    monkeypatch.setattr(
        "app.api.routes.get_vector_store",
        lambda: vector_store,
    )

    client = TestClient(app)

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "status": "deleted",
    }
    assert not uploaded_file.exists()
    vector_store.delete_document.assert_called_once_with(document_id)

    
def test_list_documents_returns_document_id_and_name(tmp_path):
    settings = get_settings()
    settings.upload_dir = tmp_path

    uploaded_file = tmp_path / "doc_abc123_bakim.pdf"
    uploaded_file.write_text("test content", encoding="utf-8")

    client = TestClient(app)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "document_id": "doc_abc123",
            "document_name": "bakim.pdf",
            "url": "/uploaded-documents/doc_abc123_bakim.pdf",
        }
    ]    
