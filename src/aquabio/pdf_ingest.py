from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import fitz


def _clean_text(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunks(text: str, size: int = 1400, overlap: int = 220) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def ingest_pdf(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    records: list[dict] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            text = _clean_text(page.get_text("text"))
            for chunk_index, content in enumerate(_chunks(text)):
                digest = hashlib.sha1(
                    f"{path.name}:{page_index}:{chunk_index}:{content[:80]}".encode("utf-8")
                ).hexdigest()[:16]
                records.append(
                    {
                        "id": f"pdf_{digest}",
                        "source_type": "pdf_chunk",
                        "source": path.name,
                        "page": page_index + 1,
                        "content": content,
                        "keywords": [],
                    }
                )
    return records


def ingest_directory(input_path: str | Path, output_path: str | Path) -> int:
    source = Path(input_path)
    pdfs = [source] if source.is_file() else sorted(source.glob("*.pdf"))
    records = [record for pdf in pdfs for record in ingest_pdf(pdf)]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)

