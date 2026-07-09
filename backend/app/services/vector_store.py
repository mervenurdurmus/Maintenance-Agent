from functools import lru_cache
import logging

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.services.chunker import Chunk
from app.services.local_embeddings import LlamaServerEmbeddings

logger = logging.getLogger("uvicorn.error")


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model

        if settings.embedding_provider == "llama_server":
            self.embedding_model = LlamaServerEmbeddings(
                endpoint=settings.embedding_endpoint,
                timeout_seconds=settings.embedding_timeout_seconds,
            )
        elif settings.embedding_provider == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )
        else:
            raise ValueError(
                f"Desteklenmeyen embedding provider: {settings.embedding_provider}"
            )

        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.embedding_collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    def add_chunks(
        self,
        document_id: str,
        document_name: str,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_model.embed_documents(texts)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    **chunk.metadata,
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_id": chunk.chunk_id,
                    "embedding_model": self.model_name,
                }
                for chunk in chunks
            ],
        )
    def delete_document(self, document_id: str) -> None:
        self.collection.delete(
            where={
                "document_id": document_id,
            }
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        collection_count = self.collection.count()

        logger.info(
            "embedding_search.started query=%r model=%s top_k=%d collection_count=%d",
            query,
            getattr(self, "model_name", "unknown"),
            top_k,
            collection_count,
        )

        if collection_count == 0:
            logger.info(
                "embedding_search.completed query=%r result_count=0 results=[]",
                query,
            )
            return []

        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: list[dict] = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            score = max(
                0.0,
                min(1.0, 1.0 - float(distance)),
            )

            matches.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "score": score,
                }
            )

        logger.info(
            "embedding_search.completed query=%r result_count=%d results=%s",
            query,
            len(matches),
            [
                {
                    "document_name": match["metadata"].get("document_name"),
                    "chunk_id": match["metadata"].get("chunk_id"),
                    "score": round(match["score"], 4),
                }
                for match in matches
            ],
        )

        return matches

@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
