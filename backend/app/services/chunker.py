from dataclasses import dataclass, field
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 350,
    overlap: int = 50,
    metadata: dict[str, Any] | None = None,
    start_index: int = 1,
) -> list[Chunk]:
    normalized = text.strip()

    if not normalized:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    pieces = splitter.split_text(normalized)

    return [
        Chunk(
            chunk_id=f"{document_id}_c{index}",
            text=piece,
            metadata={
                **(metadata or {}),
                "chunk_index": index,
            },
        )
        for index, piece in enumerate(pieces, start=start_index)
    ]
