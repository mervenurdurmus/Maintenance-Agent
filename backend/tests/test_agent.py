from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from app.models.schemas import ImageAttachment
from app.services.agent import (
    _filter_relevant_contexts,
    answer_message,
)


def test_llm_can_answer_without_selecting_a_tool() -> None:
    model = MagicMock()
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "Merhaba!",
        "intermediate_steps": [],
    }

    with (
        patch("app.services.agent.create_chat_model", return_value=model),
        patch("app.services.agent.create_tool_calling_agent", return_value=MagicMock()),
        patch("app.services.agent.AgentExecutor", return_value=executor),
    ):
        response = answer_message("Merhaba")

    assert response.answer == "Merhaba!"
    assert response.tool_calls == []
    assert response.sources == []


def test_empty_model_output_returns_fallback_for_image() -> None:
    model = MagicMock()
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "",
        "intermediate_steps": [],
    }
    attachment = ImageAttachment(
        filename="soru.png",
        content_type="image/png",
        url="/chat-images/soru.png",
    )

    with (
        patch("app.services.agent.create_chat_model", return_value=model),
        patch("app.services.agent.create_tool_calling_agent", return_value=MagicMock()),
        patch("app.services.agent.AgentExecutor", return_value=executor),
    ):
        response = answer_message(
            "Görsel eklendi.",
            attachments=[attachment],
            image_descriptions=["Görselde integral sorusu var."],
        )

    assert response.answer == (
        "Görsel için cevap oluşturulamadı. Görseli tekrar gönderip neyi çözmemi "
        "istediğini kısaca yazabilir misin?"
    )


def test_absence_answer_mentions_requested_alarm_code() -> None:
    model = MagicMock()
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "Elimde bu konuda doküman/kaynak yok.",
        "intermediate_steps": [],
    }

    with (
        patch("app.services.agent.create_chat_model", return_value=model),
        patch("app.services.agent.create_tool_calling_agent", return_value=MagicMock()),
        patch("app.services.agent.AgentExecutor", return_value=executor),
    ):
        response = answer_message("Dokümanda E999 alarm kodu için bir çözüm var mı?")

    assert response.answer == (
        "Verilen bakım dokümanında E999 alarm kodu için bilgi veya çözüm adımı "
        "bulunmuyor. Bu nedenle kaynakta olmayan bir çözüm uyduramam."
    )


def test_agent_iteration_stop_message_is_localized_to_turkish() -> None:
    model = MagicMock()
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "Agent stopped due to max iterations.",
        "intermediate_steps": [],
    }

    with (
        patch("app.services.agent.create_chat_model", return_value=model),
        patch("app.services.agent.create_tool_calling_agent", return_value=MagicMock()),
        patch("app.services.agent.AgentExecutor", return_value=executor),
    ):
        response = answer_message("hangi fabrika makinaları hakkında bilgin var")

    assert response.answer == (
        "Bu soruda araçları kullanırken karar döngüsünü tamamlayamadım. "
        "Soruyu biraz daha netleştirir misin?"
    )


def test_llm_can_select_semantic_search() -> None:
    model = MagicMock()
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "P204 hidrolik basınç alarmıdır.",
        "intermediate_steps": [
            (
                SimpleNamespace(tool="semantic_search", tool_input={"query": "P204 alarmı"}),
                {
                    "matches": [
                        {
                            "text": "P204 hidrolik hat basıncının düşük olduğunu gösterir.",
                            "metadata": {
                                "document_name": "bakim.txt",
                                "chunk_id": "doc_1_c2",
                            },
                            "score": 0.8,
                        }
                    ]
                },
            )
        ],
    }

    with (
        patch("app.services.agent.create_chat_model", return_value=model),
        patch("app.services.agent.create_tool_calling_agent", return_value=MagicMock()),
        patch("app.services.agent.AgentExecutor", return_value=executor),
    ):
        response = answer_message("P204 alarmı nedir?")

    assert response.answer == "P204 hidrolik basınç alarmıdır."
    assert response.tool_calls[0].name == "semantic_search"
    assert response.sources[0].chunk_id == "doc_1_c2"


def test_answer_uses_and_updates_langchain_chat_history() -> None:
    history = InMemoryChatMessageHistory(
        messages=[
            HumanMessage(content="P204 nedir?"),
            AIMessage(content="Hidrolik basınç alarmıdır."),
        ]
    )
    model = MagicMock()
    executor = MagicMock()
    executor.invoke.return_value = {
        "output": "Filtreyi de kontrol et.",
        "intermediate_steps": [],
    }

    with (
        patch("app.services.agent.create_chat_model", return_value=model),
        patch("app.services.agent.create_tool_calling_agent", return_value=MagicMock()),
        patch("app.services.agent.AgentExecutor", return_value=executor),
        patch("app.services.agent.ensure_conversation"),
        patch("app.services.agent.get_chat_history", return_value=history),
    ):
        response = answer_message("Başka ne yapmalıyım?", conversation_id="session-1")

    invoke_payload = executor.invoke.call_args.args[0]
    assert [message.content for message in invoke_payload["chat_history"]] == [
        "P204 nedir?",
        "Hidrolik basınç alarmıdır.",
    ]
    assert invoke_payload["input"] == "Başka ne yapmalıyım?"
    assert response.answer == "Filtreyi de kontrol et."
    assert [message.content for message in history.messages[-2:]] == [
        "Başka ne yapmalıyım?",
        "Filtreyi de kontrol et.",
    ]


def test_relevance_judge_filters_unrelated_chunks() -> None:
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content='{"relevant_chunk_ids": ["doc_1_c3"]}',
    )
    contexts = [
        {
            "text": "Alarm: P204 Hidrolik hat basıncı düşük.",
            "metadata": {
                "document_name": "bakim.txt",
                "chunk_id": "doc_1_c3",
            },
            "score": 0.8,
        },
        {
            "text": "Alarm: E106 Motor aşırı ısınıyor.",
            "metadata": {
                "document_name": "bakim.txt",
                "chunk_id": "doc_1_c2",
            },
            "score": 0.7,
        },
    ]

    with patch("app.services.agent.create_chat_model", return_value=model):
        filtered = _filter_relevant_contexts("P204 alarmı nedir?", contexts)

    assert [item["metadata"]["chunk_id"] for item in filtered] == ["doc_1_c3"]


def test_relevance_judge_can_return_no_chunks_for_unknown_alarm() -> None:
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content='{"relevant_chunk_ids": []}',
    )
    contexts = [
        {
            "text": "Alarm: P204 Hidrolik hat basıncı düşük.",
            "metadata": {
                "document_name": "bakim.txt",
                "chunk_id": "doc_1_c3",
            },
            "score": 0.8,
        },
    ]

    with patch("app.services.agent.create_chat_model", return_value=model):
        filtered = _filter_relevant_contexts("X999 alarmı nedir?", contexts)

    assert filtered == []


def test_absence_question_keeps_evidence_context_when_judge_returns_empty() -> None:
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content='{"relevant_chunk_ids": []}',
    )
    contexts = [
        {
            "text": "Alarm Kodlari\nAlarm: E106 Motor asiri isiniyor.\nAlarm: P204 Hidrolik basinc dusuktur.",
            "metadata": {
                "document_name": "bakim.txt",
                "chunk_id": "doc_1_c1",
            },
            "score": 0.7,
        },
        {
            "text": "Genel dokuman girisi.",
            "metadata": {
                "document_name": "bakim.txt",
                "chunk_id": "doc_1_c0",
            },
            "score": 0.6,
        },
    ]

    with patch("app.services.agent.create_chat_model", return_value=model):
        filtered = _filter_relevant_contexts(
            "Dokümanda E999 alarm kodu için bir çözüm var mı?",
            contexts,
        )

    assert [item["metadata"]["chunk_id"] for item in filtered] == ["doc_1_c1"]
