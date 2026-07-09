from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


document_path = Path("../heading-test.md")
text = document_path.read_text(encoding="utf-8")


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# Strateji 1: Doğrudan recursive splitting
recursive_documents = recursive_splitter.create_documents(
    texts=[text],
    metadatas=[{"strategy": "recursive"}],
)


# Strateji 2: Önce başlık, sonra recursive splitting
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "title"),
        ("##", "section"),
        ("###", "subsection"),
    ],
    strip_headers=False,
)

header_documents = header_splitter.split_text(text)

heading_recursive_documents = recursive_splitter.split_documents(
    header_documents
)


def show_documents(name: str, documents: list) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print("Chunk sayısı:", len(documents))

    for index, document in enumerate(documents, start=1):
        print(f"\nChunk {index}")
        print("Uzunluk:", len(document.page_content))
        print("Metadata:", document.metadata)
        print("İçerik:", document.page_content[:200])


show_documents(
    "SAF RECURSIVE",
    recursive_documents,
)

show_documents(
    "BAŞLIK + RECURSIVE",
    heading_recursive_documents,
)