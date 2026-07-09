from app.services.chunker import chunk_text

text = "A" * 3000

chunks_1 = chunk_text(
    text=text,
    document_id="doc_buyuk",
    chunk_size=900,
    overlap=150,
)

chunks_2 = chunk_text(
    text=text,
    document_id="doc_kucuk",
    chunk_size=500,
    overlap=100,
)

print("900/150 sonucu:", len(chunks_1), "chunk")
print("500/100 sonucu:", len(chunks_2), "chunk")