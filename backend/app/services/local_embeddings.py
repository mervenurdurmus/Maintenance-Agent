import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.embeddings import Embeddings


class LlamaServerEmbeddings(Embeddings):
    """LangChain embeddings adapter for llama.cpp's `/embedding` endpoint."""

    def __init__(self, endpoint: str, timeout_seconds: float = 60.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        request = Request(
            self.endpoint,
            data=json.dumps({"input": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding sunucusu HTTP {exc.code} döndürdü: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Embedding sunucusuna bağlanılamadı: {self.endpoint}"
            ) from exc

        embedding = _extract_embedding(payload)
        if not embedding:
            raise RuntimeError("Embedding sunucusu boş bir vektör döndürdü.")

        return [float(value) for value in embedding]


def _extract_embedding(payload: object) -> list[float]:
    if isinstance(payload, dict):
        if "embedding" in payload:
            return _single_embedding(payload["embedding"])

        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return _single_embedding(data[0].get("embedding"))

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return _single_embedding(payload[0].get("embedding"))

    raise RuntimeError("Embedding sunucusunun yanıt biçimi tanınmadı.")


def _single_embedding(value: object) -> list[float]:
    if not isinstance(value, list):
        raise RuntimeError("Embedding yanıtında vektör bulunamadı.")

    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError(
                "Sunucu token embeddingleri döndürdü; pooled embedding etkinleştirilmeli."
            )
        value = value[0]

    if not all(isinstance(item, (int, float)) for item in value):
        raise RuntimeError("Embedding vektörü sayısal olmayan değerler içeriyor.")

    return value
