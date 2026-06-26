from pathlib import Path

from pypdf import PdfReader


def load_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if path.suffix.lower() in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported document type: {path.suffix}")
