from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.services import conversation_store
from app.models.schemas import Source, ToolCall


def test_conversations_have_unique_ids_and_isolated_histories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(conversation_store, "DB_PATH", tmp_path / "history.db")

    first = conversation_store.create_conversation()
    second = conversation_store.create_conversation()

    assert first["conversation_id"] != second["conversation_id"]

    first_history = conversation_store.get_chat_history(first["conversation_id"])
    second_history = conversation_store.get_chat_history(second["conversation_id"])

    first_history.add_messages(
        [
            HumanMessage(content="P204 nedir?"),
            AIMessage(content="Bir hidrolik alarmıdır."),
        ]
    )
    second_history.add_messages([HumanMessage(content="Selam")])

    assert [message.content for message in first_history.messages] == [
        "P204 nedir?",
        "Bir hidrolik alarmıdır.",
    ]
    assert [message.content for message in second_history.messages] == ["Selam"]


def test_first_message_becomes_the_conversation_title(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(conversation_store, "DB_PATH", tmp_path / "history.db")
    conversation = conversation_store.create_conversation()

    conversation_store.ensure_conversation(
        conversation["conversation_id"],
        title_candidate="  P204 alarm kodu   ne demek?  ",
    )

    saved = conversation_store.list_conversations()[0]
    assert saved["title"] == "P204 alarm kodu ne demek?"


def test_tool_calls_are_persisted_per_conversation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(conversation_store, "DB_PATH", tmp_path / "history.db")
    conversation = conversation_store.create_conversation()

    conversation_store.save_tool_calls(
        conversation["conversation_id"],
        [
            ToolCall(
                name="semantic_search",
                input={"query": "P204 alarmı"},
                output={"matches": []},
            )
        ],
    )

    saved = conversation_store.list_tool_calls(conversation["conversation_id"])
    assert saved[0]["name"] == "semantic_search"
    assert saved[0]["input"] == {"query": "P204 alarmı"}
    assert saved[0]["output"] == {"matches": []}


def test_sources_are_persisted_with_assistant_turn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(conversation_store, "DB_PATH", tmp_path / "history.db")
    conversation = conversation_store.create_conversation()

    conversation_store.save_conversation_turn(
        conversation_id=conversation["conversation_id"],
        turn_id="turn_1",
        user_message="P204 ne?",
        assistant_message="P204 hidrolik basınç alarmıdır.",
        sources=[
            Source(
                document_name="bakim.txt",
                chunk_id="doc_1_c3",
                score=0.82,
            )
        ],
    )

    messages = conversation_store.list_history_messages(conversation["conversation_id"])

    assert messages[0]["sources"] == []
    assert messages[1]["sources"] == [
        {
            "document_name": "bakim.txt",
            "chunk_id": "doc_1_c3",
            "score": 0.82,
            "page_number": None,
        }
    ]
