from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.config import get_settings
from app.services.chunker import Chunk

COLLECTION_NAME = "maintenance_documents"


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        model_name = settings.embedding_model.replace("sentence-transformers/", "")
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
        )

    def add_chunks(self, document_id: str, document_name: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"document_id": document_id, "document_name": document_name, "chunk_id": chunk.chunk_id}
                for chunk in chunks
            ],
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: list[dict] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            matches.append({"text": text, "metadata": metadata, "score": score})
        return matches


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
