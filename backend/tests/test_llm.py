from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.llm import create_chat_model


def test_create_chat_model_uses_openrouter_when_selected() -> None:
    settings = SimpleNamespace(
        chat_llm_provider="openrouter",
        chat_llm_model="google/gemma-4-31b-it:free",
        openrouter_model="google/gemma-4-31b-it:free",
        openrouter_api_key="test-openrouter-key",
        groq_model="openai/gpt-oss-120b",
        groq_api_key="test-groq-key",
    )

    with (
        patch("app.services.llm.get_settings", return_value=settings),
        patch("app.services.llm.ChatOpenAI", return_value=MagicMock()) as chat_openai,
    ):
        create_chat_model()

    chat_openai.assert_called_once_with(
        model="google/gemma-4-31b-it:free",
        api_key="test-openrouter-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=4000,
    )


def test_create_chat_model_uses_groq_when_selected() -> None:
    settings = SimpleNamespace(
        chat_llm_provider="groq",
        chat_llm_model="openai/gpt-oss-20b",
        openrouter_model="google/gemma-4-31b-it:free",
        openrouter_api_key="test-openrouter-key",
        groq_model="openai/gpt-oss-120b",
        groq_api_key="test-groq-key",
    )

    with (
        patch("app.services.llm.get_settings", return_value=settings),
        patch("app.services.llm.ChatGroq", return_value=MagicMock()) as chat_groq,
    ):
        create_chat_model()

    chat_groq.assert_called_once_with(
        model="openai/gpt-oss-20b",
        api_key="test-groq-key",
        temperature=0.1,
        max_tokens=4000,
    )
