import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.local_embeddings import LlamaServerEmbeddings, _extract_embedding


def test_embed_query_calls_llama_server_embedding_endpoint() -> None:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    ).encode("utf-8")
    embeddings = LlamaServerEmbeddings("http://127.0.0.1:8080/embedding")

    with patch("app.services.local_embeddings.urlopen", return_value=response) as request:
        result = embeddings.embed_query("P204 alarmı")

    assert result == [0.1, 0.2, 0.3]
    sent_request = request.call_args.args[0]
    assert sent_request.full_url == "http://127.0.0.1:8080/embedding"
    assert json.loads(sent_request.data) == {"input": "P204 alarmı"}


def test_extract_embedding_supports_openai_compatible_response() -> None:
    result = _extract_embedding(
        {
            "data": [
                {
                    "index": 0,
                    "embedding": [0.4, 0.5],
                }
            ]
        }
    )

    assert result == [0.4, 0.5]


def test_extract_embedding_rejects_unpooled_token_embeddings() -> None:
    with pytest.raises(RuntimeError, match="pooled embedding"):
        _extract_embedding(
            [
                {
                    "index": 0,
                    "embedding": [[0.1, 0.2], [0.3, 0.4]],
                }
            ]
        )
