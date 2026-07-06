from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import MRAGPaths
from .io_utils import write_jsonl
from .models import PDFRegistryEntry, RAGDocument


def load_registry(paths: MRAGPaths) -> list[PDFRegistryEntry]:
    config_path = paths.root / "configs" / "pdf_sources.json"
    return [
        PDFRegistryEntry.model_validate(item)
        for item in json.loads(config_path.read_text(encoding="utf-8"))
    ]


def download_registered_pdfs(paths: MRAGPaths) -> dict:
    paths.ensure()
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "AquaBio-MRAG/0.2 educational research project"}
    )
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=2,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
        ),
    )
    registry = load_registry(paths)
    results = []
    for item in registry:
        target = paths.root / item.local_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes()[:4] == b"%PDF":
            status = "existing"
        else:
            response = session.get(item.source_url, timeout=180)
            response.raise_for_status()
            if response.content[:4] != b"%PDF":
                raise ValueError(f"下载内容不是 PDF：{item.source_url}")
            target.write_bytes(response.content)
            status = "downloaded"
        results.append(
            {
                **item.model_dump(),
                "download_status": status,
                "file_size": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        print(f"pdf: {item.doc_id} {status}")
    count = write_jsonl(paths.knowledge_dir / "pdf_registry.jsonl", results)
    return {"registered": count, "items": results}


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, size: int = 1800, overlap: int = 250) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            candidates = [
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
            ]
            boundary = max(candidates)
            if boundary > start + size // 2:
                end = boundary + 1
        content = text[start:end].strip()
        if content:
            chunks.append(content)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _section(text: str) -> str:
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if 4 <= len(stripped) <= 100 and (
            stripped.isupper()
            or re.match(r"^(\d+(\.\d+)*)?\s*[A-Z][A-Za-z \-:]+$", stripped)
        ):
            return stripped
    return ""


def _related_species(text: str, species: list[dict]) -> list[str]:
    lowered = text.lower()
    related = []
    for item in species:
        terms = [
            item["english_name"].lower(),
            item["scientific_name"].split("/")[0].strip().lower(),
            *[str(value).lower() for value in item.get("keywords", [])],
        ]
        if any(term and term in lowered for term in terms):
            related.append(item["species_id"])
    return sorted(set(related))


def parse_registered_pdfs(
    paths: MRAGPaths,
    extract_figures: bool = False,
) -> dict:
    registry = load_registry(paths)
    species = json.loads(paths.species_list.read_text(encoding="utf-8"))
    chunk_records: list[dict] = []
    figure_records: list[dict] = []

    for entry in registry:
        local_path = paths.root / entry.local_path
        if not local_path.exists():
            raise FileNotFoundError(
                f"缺少登记 PDF：{local_path}，请先执行下载命令。"
            )
        with fitz.open(local_path) as document:
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                page_text = _clean(page.get_text("text"))
                page_section = _section(page_text)
                for chunk_index, content in enumerate(_split(page_text)):
                    related = _related_species(content, species)
                    chunk_type = (
                        "species_description"
                        if related and entry.doc_type == "species_identification_guide"
                        else "ecology_context"
                        if entry.doc_type
                        in {
                            "ecology_monitoring_manual",
                            "education_material",
                            "conservation_report",
                        }
                        else "document_text"
                    )
                    chunk_id = (
                        f"{entry.doc_id}_p{page_number:04d}_c{chunk_index + 1:03d}"
                    )
                    record = RAGDocument(
                        id=chunk_id,
                        source_type="pdf_chunk",
                        species_id=related[0] if len(related) == 1 else "",
                        modality="pdf_text",
                        content=content,
                        embedding_text=" ".join(
                            [
                                entry.title,
                                entry.source_org,
                                entry.doc_type,
                                page_section,
                                chunk_type,
                                " ".join(related),
                                content,
                            ]
                        ),
                        metadata={
                            "doc_id": entry.doc_id,
                            "doc_title": entry.title,
                            "doc_type": entry.doc_type,
                            "source_org": entry.source_org,
                            "source_url": entry.source_url,
                            "page": page_number,
                            "section": page_section,
                            "chunk_type": chunk_type,
                            "related_species": related,
                        },
                    )
                    chunk_records.append(record.model_dump())

                if extract_figures:
                    for figure_index, image in enumerate(
                        page.get_images(full=True), start=1
                    ):
                        xref = image[0]
                        extracted = document.extract_image(xref)
                        extension = extracted["ext"]
                        figure_id = (
                            f"fig_{entry.doc_id}_p{page_number:04d}_{figure_index:03d}"
                        )
                        figure_path = (
                            paths.pdf_figures_dir / f"{figure_id}.{extension}"
                        )
                        figure_path.write_bytes(extracted["image"])
                        nearby = page_text[:600]
                        related = _related_species(nearby, species)
                        figure_records.append(
                            RAGDocument(
                                id=figure_id,
                                source_type="pdf_figure",
                                species_id=related[0] if len(related) == 1 else "",
                                modality="pdf_figure",
                                content=(
                                    f"Figure extracted from {entry.title}, "
                                    f"page {page_number}. Nearby text: {nearby}"
                                ),
                                embedding_text=" ".join(
                                    [entry.title, " ".join(related), nearby]
                                ),
                                metadata={
                                    "doc_id": entry.doc_id,
                                    "doc_title": entry.title,
                                    "doc_type": entry.doc_type,
                                    "source_url": entry.source_url,
                                    "page": page_number,
                                    "image_path": str(
                                        figure_path.relative_to(paths.root)
                                    ).replace("\\", "/"),
                                    "related_species": related,
                                },
                            ).model_dump()
                        )
        print(f"parsed pdf: {entry.doc_id}")

    chunks = write_jsonl(
        paths.knowledge_dir / "pdf_chunks.jsonl", chunk_records
    )
    figures = write_jsonl(
        paths.knowledge_dir / "pdf_figures.jsonl", figure_records
    )
    return {"pdf_chunks": chunks, "pdf_figures": figures}

