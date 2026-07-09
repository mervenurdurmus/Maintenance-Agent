from app.services.vector_store import VectorStore

store = VectorStore()

result = store.collection.get(
    limit=1,
    include=["documents", "embeddings", "metadatas"],
)

text = result["documents"][0]
embedding = result["embeddings"][0]

print("Chunk metni:", text)
print("Embedding boyutu:", len(embedding))
print("İlk 10 sayı:", embedding[:10])