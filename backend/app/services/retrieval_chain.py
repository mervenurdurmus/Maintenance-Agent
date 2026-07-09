"""create_retrieval_chain öğrenme deneyi.

Web uygulamasının ana RAG akışı bu dosyayı kullanmaz.
Ana akış agent.py içindeki LLM kontrollü semantic_search aracıdır.
"""
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.config import get_settings
from app.services.llm import create_chat_model
from app.services.vector_store import get_vector_store


def retrieve_documents(inputs: dict) -> list[Document]:
    question = inputs["input"]
    settings = get_settings()

    matches = get_vector_store().search(
        query=question,
        top_k=settings.top_k,
    )

    return [
        Document(
            page_content=match["text"],
            metadata=match["metadata"],
        )
        for match in matches
    ]


def create_rag_chain():
    retriever = RunnableLambda(retrieve_documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Soruyu yalnızca aşağıdaki kaynaklara dayanarak cevapla.
Kaynaklarda cevap yoksa bunu açıkça söyle.
Kullandığın chunk_id bilgisini belirt.

Kaynaklar:
{context}""",
            ),
            ("human", "{input}"),
        ]
    )

    document_prompt = PromptTemplate.from_template(
        """Dosya: {document_name}
Chunk: {chunk_id}
İçerik: {page_content}"""
    )

    document_chain = create_stuff_documents_chain(
        llm=create_chat_model(),
        prompt=prompt,
        document_prompt=document_prompt,
    )

    return create_retrieval_chain(
        retriever,
        document_chain,
    )
