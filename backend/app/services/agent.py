import json
import re
from typing import Any
from uuid import uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_settings
from app.models.schemas import ChatResponse, ImageAttachment, Source, ToolCall
from app.services.conversation_store import (
    SQLiteChatMessageHistory,
    ensure_conversation,
    get_chat_history,
    save_conversation_turn,
    save_tool_calls,
)
from app.services.llm import SYSTEM_PROMPT, create_chat_model
from app.services.reranker import rerank_contexts
from app.services.vector_store import get_vector_store
from app.tools.deterministic_tools import (
    calculate_next_maintenance as calculate_next_maintenance_value,
)
from app.tools.deterministic_tools import date_info as date_info_value
from app.tools.deterministic_tools import get_today as get_today_value

MAX_TOOL_ROUNDS = 4
MAX_JUDGE_TEXT_CHARS = 1200
RELEVANCE_JUDGE_PROMPT = """Sen bir retrieval kalite denetleyicisisin.

Görevin, kullanıcı sorusu için getirilen chunk'ların gerçekten ilgili olup olmadığını seçmek.

Kurallar:
- Sadece soruyu doğrudan cevaplamaya yarayan chunk_id'leri seç.
- Benzer konu ama farklı alarm kodu, farklı ekipman veya farklı bakım işlemi ise ilgili sayma.
- Kullanıcı belirli bir alarm kodu soruyorsa chunk içinde aynı alarm kodu açıkça geçmiyorsa ilgili sayma.
- Genel başlık, doküman giriş kısmı veya sadece kategori adı tek başına yeterli değildir.
- Emin değilsen chunk'ı seçme.
- Sadece geçerli JSON döndür. Açıklama yazma.

JSON formatı:
{"relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"]}
"""
ABSENCE_EVIDENCE_TERMS = (
    "alarm kodlari",
    "alarm kodları",
    "alarm:",
    "periyodik bakim",
    "periyodik bakım",
    "bakim proseduru",
    "bakım prosedürü",
    "is guvenligi",
    "iş güvenliği",
)
AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


@tool
def semantic_search(query: str) -> dict[str, Any]:
    """Bakım, alarm ve iş güvenliği soruları için yüklenen dokümanlarda anlamsal arama yapar."""
    settings = get_settings()
    # Retrieve a wider candidate set, then compress it to the most relevant chunks.
    matches = get_vector_store().search(query, top_k=settings.top_k)
    contexts = rerank_contexts(query, matches, top_n=settings.rerank_top_n)
    contexts = _filter_relevant_contexts(query, contexts)
    return {"matches": contexts}


@tool("get_today")
def get_today_tool() -> dict[str, Any]:
    """Bugünün tarihini ve haftanın gününü kesin sistem verisiyle döndürür."""
    return get_today_value()


@tool("date_info")
def date_info_tool(
    date_value: str | None = None,
    offset_days: int | None = None,
) -> dict[str, Any]:
    """Belirli bir tarihin veya bugüne göre göreli bir tarihin gün/ay/yıl ve haftanın günü bilgisini kesin hesaplar.

    Args:
        date_value: YYYY-MM-DD biçiminde tarih. Örnek: 2026-06-22.
        offset_days: Bugüne göre gün farkı. Dün için -1, yarın için 1.
    """
    return date_info_value(date_value=date_value, offset_days=offset_days)


@tool("calculate_next_maintenance")
def calculate_next_maintenance_tool(last_date: str, interval_days: int) -> dict[str, Any]:
    """Son bakım tarihi ve gün cinsinden periyodu kullanarak sonraki bakım tarihini hesaplar."""
    return calculate_next_maintenance_value(last_date=last_date, interval_days=interval_days)


TOOLS: list[BaseTool] = [
    semantic_search,
    get_today_tool,
    date_info_tool,
    calculate_next_maintenance_tool,
]


def answer_message(
    message: str,
    conversation_id: str | None = None,
    attachments: list[ImageAttachment] | None = None,
    image_descriptions: list[str] | None = None,
) -> ChatResponse:
    if conversation_id:
        ensure_conversation(conversation_id, title_candidate=message)
    history = get_chat_history(conversation_id) if conversation_id else None

    try:
        model = create_chat_model()
        agent = create_tool_calling_agent(model, TOOLS, AGENT_PROMPT)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            return_intermediate_steps=True,
            max_iterations=MAX_TOOL_ROUNDS,
        )
    except Exception as exc:
        response = _error_response(exc)
        _record_turn(history, message, response.answer, conversation_id)
        return response

    chat_history = list(history.messages) if history else []
    agent_message = _message_with_attachment_context(
        message,
        attachments or [],
        image_descriptions or [],
    )
    recorded_calls: list[ToolCall] = []
    sources: list[Source] = []
    turn_id = f"turn_{uuid4().hex}"

    try:
        result = agent_executor.invoke(
            {
                "input": agent_message,
                "chat_history": chat_history,
            }
        )

        for step in result.get("intermediate_steps", []):
            action, output = step
            tool_output = output if isinstance(output, dict) else {"result": output}

            recorded_calls.append(
                ToolCall(
                    name=action.tool,
                    input=action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input},
                    output=_public_tool_output(action.tool, tool_output),
                    turn_id=turn_id,
                )
            )

            if action.tool == "semantic_search":
                sources = _sources_from_search(tool_output)

        answer = _final_answer_text(
            result.get("output", ""),
            message,
            has_attachments=bool(attachments),
        )
        chat_response = ChatResponse(
            answer=answer,
            sources=sources,
            tool_calls=recorded_calls,
            turn_id=turn_id,
            attachments=attachments or [],
        )
        _record_turn(
            history,
            message,
            chat_response.answer,
            conversation_id,
            recorded_calls,
            turn_id,
            sources,
            attachments,
        )
        return chat_response
    except Exception as exc:
        response = _error_response(
            exc,
            tool_calls=recorded_calls,
            sources=sources,
            turn_id=turn_id,
            attachments=attachments or [],
        )
        _record_turn(
            history,
            message,
            response.answer,
            conversation_id,
            recorded_calls,
            turn_id,
            response.sources,
            attachments,
        )
        return response


def _record_turn(
    history: BaseChatMessageHistory | None,
    user_message: str,
    assistant_message: str,
    conversation_id: str | None,
    tool_calls: list[ToolCall] | None = None,
    turn_id: str | None = None,
    sources: list[Source] | None = None,
    user_attachments: list[ImageAttachment] | None = None,
) -> None:
    if history is None:
        return

    if conversation_id and turn_id and isinstance(history, SQLiteChatMessageHistory):
        save_conversation_turn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message=user_message,
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            sources=sources,
            user_attachments=user_attachments,
        )
        return

    history.add_messages(
        [
            HumanMessage(content=user_message),
            AIMessage(content=assistant_message),
        ]
    )
    if conversation_id and tool_calls:
        save_tool_calls(conversation_id, tool_calls, turn_id=turn_id)


def _filter_relevant_contexts(query: str, contexts: list[dict]) -> list[dict]:
    if not contexts:
        return []

    candidates = [
        {
            "chunk_id": context.get("metadata", {}).get("chunk_id"),
            "document_name": context.get("metadata", {}).get("document_name"),
            "score": context.get("score"),
            "text": str(context.get("text", ""))[:MAX_JUDGE_TEXT_CHARS],
        }
        for context in contexts
        if context.get("metadata", {}).get("chunk_id")
    ]

    if not candidates:
        return []

    try:
        model = create_chat_model()
        response = model.invoke(
            [
                SystemMessage(content=RELEVANCE_JUDGE_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": query,
                            "chunks": candidates,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        relevant_chunk_ids = _parse_relevance_judge_response(_message_text(response.content))
    except Exception:
        return contexts

    if relevant_chunk_ids is None:
        return contexts

    filtered_contexts = [
        context
        for context in contexts
        if context.get("metadata", {}).get("chunk_id") in relevant_chunk_ids
    ]
    if filtered_contexts:
        return filtered_contexts

    if _is_absence_question(query):
        return _absence_evidence_contexts(contexts)

    return filtered_contexts


def _is_absence_question(query: str) -> bool:
    normalized = query.casefold()
    asks_about_document = any(
        marker in normalized
        for marker in ("dokümanda", "dokumanda", "kaynakta", "verilmiş", "verilmis")
    )
    asks_presence = any(
        marker in normalized
        for marker in ("var mı", "var mi", "açıklanıyor mu", "aciklaniyor mu", "kaç", "kac")
    )

    return asks_about_document and asks_presence


def _absence_evidence_contexts(contexts: list[dict]) -> list[dict]:
    evidence_contexts = [
        context
        for context in contexts
        if any(term in str(context.get("text", "")).casefold() for term in ABSENCE_EVIDENCE_TERMS)
    ]

    return (evidence_contexts or contexts)[:1]


def _parse_relevance_judge_response(content: str) -> set[str] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    relevant_chunk_ids = payload.get("relevant_chunk_ids")
    if not isinstance(relevant_chunk_ids, list):
        return None

    return {str(chunk_id) for chunk_id in relevant_chunk_ids}


def _sources_from_search(output: dict[str, Any]) -> list[Source]:
    sources: list[Source] = []
    seen_chunk_ids: set[str] = set()

    for match in output.get("matches", []):
        metadata = match.get("metadata", {})
        chunk_id = metadata.get("chunk_id")
        document_name = metadata.get("document_name")
        if not chunk_id or not document_name or chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        sources.append(
            Source(
                document_name=metadata.get("document_name", "unknown"),
                chunk_id=metadata.get("chunk_id", ""),
                score=match.get("score", 0.0),
                page_number=metadata.get("page_number"),
            )
        )

    return sources


def _public_tool_output(name: str, output: dict[str, Any]) -> dict[str, Any]:
    if name != "semantic_search" or "matches" not in output:
        return output

    return {
        "matches": [
            {
                "document_name": match.get("metadata", {}).get("document_name"),
                "chunk_id": match.get("metadata", {}).get("chunk_id"),
                "score": match.get("score", 0.0),
            }
            for match in output["matches"]
        ]
    }


def _message_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _message_with_attachment_context(
    message: str,
    attachments: list[ImageAttachment],
    image_descriptions: list[str],
) -> str:
    if not attachments:
        return message

    attachment_names = ", ".join(attachment.filename for attachment in attachments)
    if image_descriptions:
        image_context = "\n\n".join(
            f"Görsel {index}: {description}"
            for index, description in enumerate(image_descriptions, start=1)
        )
        return (
            f"{message}\n\n"
            f"Kullanıcı şu görsel eklerini gönderdi: {attachment_names}.\n"
            "Görsel yükleme eylemi inceleme izni sayılır; kullanıcıdan tekrar izin isteme.\n"
            "Eğer mesaj boş veya genel ise görseldeki soruyu/konuyu doğrudan çöz veya açıkla.\n"
            "Görselde çözülmesi gereken bir soru varsa çözümü adım adım anlat; sadece nihai cevabı verme.\n"
            f"Görselden okunan bilgi:\n{image_context}"
        )

    return (
        f"{message}\n\n"
        f"Kullanıcı şu görsel eklerini gönderdi: {attachment_names}. "
        "Bu çalışma modunda görsel dosya sohbete eklenir ve geçmişte saklanır; "
        "görselin piksel içeriğini analiz etmek için vision destekli model gerekir. "
        "Eğer kullanıcı görselin içeriğini analiz etmeni isterse bunu açıkça belirt."
    )


def _final_answer_text(
    raw_output: Any,
    user_message: str = "",
    has_attachments: bool = False,
) -> str:
    answer = str(raw_output or "").strip()
    if answer:
        if _is_agent_iteration_stop(answer):
            return _agent_iteration_stop_answer(user_message)
        return _specific_absence_answer(user_message, answer)

    if has_attachments:
        return (
            "Görsel için cevap oluşturulamadı. Görseli tekrar gönderip neyi çözmemi "
            "istediğini kısaca yazabilir misin?"
        )

    return "Cevap oluşturulamadı. Soruyu tekrar yazar mısın?"


def _is_agent_iteration_stop(answer: str) -> bool:
    return "agent stopped due to max iterations" in answer.casefold()


def _agent_iteration_stop_answer(user_message: str) -> str:
    if _looks_turkish(user_message):
        return (
            "Bu soruda araçları kullanırken karar döngüsünü tamamlayamadım. "
            "Soruyu biraz daha netleştirir misin?"
        )

    return (
        "I could not finish the tool-selection loop for this question. "
        "Could you make the question a little more specific?"
    )


def _looks_turkish(text: str) -> bool:
    normalized = text.casefold()
    turkish_markers = (
        "ı",
        "ğ",
        "ü",
        "ş",
        "ö",
        "ç",
        "hangi",
        "nedir",
        "mı",
        "mi",
        "musun",
        "mısın",
        "bilgin",
    )
    return any(marker in normalized for marker in turkish_markers)


def _specific_absence_answer(user_message: str, answer: str) -> str:
    normalized_answer = answer.casefold()
    if "doküman/kaynak yok" not in normalized_answer and "dokuman/kaynak yok" not in normalized_answer:
        return answer

    if not _is_absence_question(user_message):
        return answer

    alarm_match = re.search(r"\b([A-ZÇĞİÖŞÜ]{1,4}\d{2,4})\b", user_message, flags=re.IGNORECASE)
    if alarm_match:
        code = alarm_match.group(1).upper()
        return (
            f"Verilen bakım dokümanında {code} alarm kodu için bilgi veya çözüm adımı "
            "bulunmuyor. Bu nedenle kaynakta olmayan bir çözüm uyduramam."
        )

    return (
        "Verilen bakım dokümanında bu konu için bilgi bulunmuyor. "
        "Bu nedenle kaynakta olmayan bir cevap uyduramam."
    )


def _error_response(
    exc: Exception,
    tool_calls: list[ToolCall] | None = None,
    sources: list[Source] | None = None,
    turn_id: str | None = None,
    attachments: list[ImageAttachment] | None = None,
) -> ChatResponse:
    message = str(exc)
    if "api key" in message.lower() or "groq_api_key" in message.lower():
        answer = "Groq API anahtarı tanımlı değil. backend/.env içinde GROQ_API_KEY girilmelidir."
    else:
        answer = f"Groq API hatası: {message}"

    return ChatResponse(
        answer=answer,
        sources=sources or [],
        tool_calls=tool_calls or [],
        turn_id=turn_id,
        attachments=attachments or [],
    )
