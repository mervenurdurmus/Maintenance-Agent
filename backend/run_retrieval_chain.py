from langchain.globals import set_debug
from app.services.retrieval_chain import create_rag_chain

set_debug(True)

chain = create_rag_chain()

result = chain.invoke(
    {
        "input": "Aylık bakım sırasında hangi kontroller yapılmalıdır?"
    }
)

print("\nCEVAP:")
print(result["answer"])

print("\nGETİRİLEN CHUNK'LAR:")
for document in result["context"]:
    print(
        document.metadata["chunk_id"],
        "-",
        document.metadata["document_name"],
    )
