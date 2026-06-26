import re
from typing import Any, TypedDict

from langchain_core.runnables import RunnableLambda

from app.core.config import get_settings
from app.models.schemas import RouteName
from app.models.schemas import ChatResponse, Source, ToolCall
from app.services.llm import generate_answer
from app.services.reranker import rerank_contexts
from app.services.router import classify_route, should_use_rag, should_use_tools
from app.services.vector_store import get_vector_store
from app.tools.deterministic_tools import calculate_next_maintenance, get_today


class AgentState(TypedDict, total=False):
    message: str
    route: RouteName
    tool_calls: list[ToolCall]
    contexts: list[dict[str, Any]]
    answer: str


def _route_question(state: AgentState) -> AgentState:
    return {
        **state,
        "route": classify_route(state["message"]),
        "tool_calls": [],
        "contexts": [],
    }


def _run_deterministic_tools(state: AgentState) -> AgentState:
    route = state["route"]
    tool_calls: list[ToolCall] = []

    if should_use_tools(route):
        tool_calls = _run_tools(state["message"], route)

        if tool_calls:
            return {
                **state,
                "tool_calls": tool_calls,
                "contexts": [
                    {
                        "text": f"Tool sonucu: {tool_calls[0].output}",
                        "metadata": {
                            "document_name": "tool_result",
                            "chunk_id": tool_calls[0].name,
                            **tool_calls[0].output,
                        },
                        "score": 1.0,
                    }
                ],
            }

    return {**state, "tool_calls": tool_calls}


def _retrieve_contexts(state: AgentState) -> AgentState:
    if state.get("contexts"):
        return state

    route = state["route"]
    if not should_use_rag(route):
        return state

    settings = get_settings()
    matches = get_vector_store().search(state["message"], top_k=settings.top_k)
    contexts = rerank_contexts(state["message"], matches, top_n=settings.rerank_top_n)
    return {**state, "contexts": contexts}


def _generate_final_answer(state: AgentState) -> AgentState:
    answer = generate_answer(
        question=state["message"],
        contexts=state.get("contexts", []),
        route=state["route"],
    )

    return {**state, "answer": answer}


maintenance_agent_chain = (
    RunnableLambda(_route_question)
    | RunnableLambda(_run_deterministic_tools)
    | RunnableLambda(_retrieve_contexts)
    | RunnableLambda(_generate_final_answer)
)


def answer_message(message: str) -> ChatResponse:
    state = maintenance_agent_chain.invoke({"message": message})
    contexts = state.get("contexts", [])

    sources = [
        Source(
            document_name=item["metadata"]["document_name"],
            chunk_id=item["metadata"]["chunk_id"],
            score=item["score"],
        )
        for item in contexts
        if item["metadata"]["document_name"] != "tool_result"
    ]

    return ChatResponse(
        answer=state["answer"],
        sources=sources,
        tool_calls=state.get("tool_calls", []),
        route=state["route"],
    )


def _run_tools(message: str, route: str) -> list[ToolCall]:
    if route == "date_question":
        return [
            ToolCall(
                name="get_today",
                input={},
                output=get_today(),
            )
        ]

    match = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d+)\s*g[üu]n", message.lower())

    if route == "period_calculation" and match:
        last_date, interval_days = match.groups()
        tool_input = {
            "last_date": last_date,
            "interval_days": int(interval_days),
        }

        return [
            ToolCall(
                name="calculate_next_maintenance",
                input=tool_input,
                output=calculate_next_maintenance(**tool_input),
            )
        ]

    return []
