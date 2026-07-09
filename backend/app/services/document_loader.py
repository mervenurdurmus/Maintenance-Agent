from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader  
@dataclass(frozen=True)
class DocumentSection:  
    text: str
    metadata: dict[str, str | int]

def load_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if path.suffix.lower() in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported document type: {path.suffix}")   
def load_document_sections(path: Path) -> list[DocumentSection]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        pdf_title = path.stem
        if reader.metadata and reader.metadata.title:
            pdf_title = str(reader.metadata.title)

        sections = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()

            if page_text:
                sections.append(
                    DocumentSection(
                        text=page_text,
                        metadata={
                            "page_number": page_number,
                            "document_title": pdf_title,
                        },
                    )
                )

        return sections

    text = load_document_text(path).strip()

    if not text:
        return []

    return [
        DocumentSection(
            text=text,
            metadata={
                "document_title": path.stem,
            },
        )
    ]
