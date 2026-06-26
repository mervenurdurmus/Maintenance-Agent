from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str


def chunk_text(text: str, document_id: str, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 1

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(Chunk(chunk_id=f"{document_id}_c{index}", text=chunk))
        if end == len(normalized):
            break
        start = max(0, end - overlap)
        index += 1

    return chunks
