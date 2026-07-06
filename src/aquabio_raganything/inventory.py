from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz

from .config import BOOKS, RAGAnythingPaths, RAGAnythingSettings


UPPER_TAXA = {
    "Echinodermata",
    "Asteroidea",
    "Echinoidea",
    "Holothuroidea",
    "Mollusca",
    "Bivalvia",
    "Cephalopoda",
    "Octopoda",
    "Teuthida",
    "Crustacea",
    "Caridea",
    "Brachyura",
    "Nephropidae",
    "Palinuridae",
    "Cnidaria",
    "Anthozoa",
    "Actiniaria",
    "Scleractinia",
    "Chondrichthyes",
    "Myliobatiformes",
    "Selachimorpha",
    "Mobulidae",
    "Actinopterygii",
    "Anguilliformes",
    "Hippocampus",
    "Amphiprioninae",
    "Chelonioidea",
    "Delphinidae",
}


@dataclass(frozen=True)
class Segment:
    segment_id: str
    doc_id: str
    book_id: str
    source_file: str
    start_page: int
    end_page: int
    matched_species: list[str]
    scope: str
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_terms(species_path: Path) -> dict[str, list[str]]:
    records = json.loads(species_path.read_text(encoding="utf-8"))
    terms: dict[str, list[str]] = {}
    for row in records:
        values = {
            row.get("english_name", ""),
            row.get("scientific_name", ""),
            row.get("worms_name", ""),
            row.get("wiki_title", "").replace("_", " "),
        }
        values.update(
            item
            for item in row.get("keywords", [])
            if item and item.isascii()
        )
        terms[row["species_id"]] = sorted(
            {value.strip() for value in values if value.strip()},
            key=len,
            reverse=True,
        )
    return terms


def _matches(text: str, terms: Iterable[str]) -> list[str]:
    found = []
    for term in terms:
        if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text, re.I):
            found.append(term)
    return found


def _split_range(
    start: int,
    end: int,
    max_pages: int,
    overlap: int,
) -> list[tuple[int, int]]:
    result = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + max_pages - 1)
        result.append((cursor, stop))
        if stop == end:
            break
        cursor = stop - overlap + 1
    return result


def _merge_candidates(
    pages: list[dict],
    page_count: int,
    max_pages: int,
    overlap: int,
) -> list[tuple[int, int, list[str]]]:
    candidates = [row for row in pages if row["candidate"]]
    if not candidates:
        return []
    groups: list[tuple[int, int, set[str]]] = []
    for row in candidates:
        page = row["page"]
        species = set(row["matched_species"])
        if groups and page - groups[-1][1] <= 3:
            start, _, previous = groups[-1]
            groups[-1] = (start, page, previous | species)
        else:
            groups.append((page, page, species))

    expanded: list[tuple[int, int, set[str]]] = []
    for start, end, species in groups:
        start = max(1, start - 1)
        end = min(page_count, end + 1)
        if expanded and start <= expanded[-1][1] + 1:
            old_start, old_end, old_species = expanded[-1]
            expanded[-1] = (
                old_start,
                max(old_end, end),
                old_species | species,
            )
        else:
            expanded.append((start, end, species))

    result = []
    for start, end, species in expanded:
        for part_start, part_end in _split_range(
            start, end, max_pages, overlap
        ):
            result.append((part_start, part_end, sorted(species)))
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_inventory(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
) -> dict:
    paths.ensure()
    species_terms = _load_terms(paths.species_list)
    page_rows: list[dict] = []
    books = []
    relevant_segments: list[Segment] = []
    full_segments: list[Segment] = []

    for book_id, filename in BOOKS.items():
        pdf_path = paths.pdf_dir / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Required PDF not found: {pdf_path}")
        source_hash = _sha256(pdf_path)
        document = fitz.open(pdf_path)
        try:
            page_count = document.page_count
            image_total = 0
            book_pages = []
            for page_index in range(page_count):
                page = document.load_page(page_index)
                text = page.get_text("text")
                images = page.get_images(full=True)
                image_total += len(images)
                matched_species = []
                matched_terms = []
                for species_id, terms in species_terms.items():
                    hits = _matches(text, terms)
                    if hits:
                        matched_species.append(species_id)
                        matched_terms.extend(hits)
                matched_terms.extend(_matches(text, UPPER_TAXA))
                row = {
                    "book_id": book_id,
                    "source_file": filename,
                    "page": page_index + 1,
                    "page_index": page_index,
                    "text_chars": len(text.strip()),
                    "image_count": len(images),
                    "matched_species": sorted(set(matched_species)),
                    "matched_terms": sorted(set(matched_terms)),
                    "needs_ocr": len(text.strip()) < 120 and bool(images),
                    "candidate": bool(matched_species),
                }
                page_rows.append(row)
                book_pages.append(row)
            books.append(
                {
                    "book_id": book_id,
                    "source_file": filename,
                    "path": str(pdf_path),
                    "pages": page_count,
                    "image_objects": image_total,
                    "sha256": source_hash,
                }
            )
            for start, end, species in _merge_candidates(
                book_pages,
                page_count,
                settings.max_segment_pages,
                settings.segment_overlap,
            ):
                segment_id = f"{book_id}_p{start:04d}_{end:04d}"
                relevant_segments.append(
                    Segment(
                        segment_id=segment_id,
                        doc_id=f"doc_{segment_id}",
                        book_id=book_id,
                        source_file=filename,
                        start_page=start,
                        end_page=end,
                        matched_species=species,
                        scope="relevant",
                        source_sha256=source_hash,
                    )
                )
            for start, end in _split_range(
                1,
                page_count,
                settings.max_segment_pages,
                settings.segment_overlap,
            ):
                segment_id = f"{book_id}_p{start:04d}_{end:04d}"
                full_segments.append(
                    Segment(
                        segment_id=segment_id,
                        doc_id=f"doc_{segment_id}",
                        book_id=book_id,
                        source_file=filename,
                        start_page=start,
                        end_page=end,
                        matched_species=[],
                        scope="full",
                        source_sha256=source_hash,
                    )
                )
        finally:
            document.close()

    _write_json(paths.inventory_dir / "books.json", books)
    with (paths.inventory_dir / "page_inventory.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in page_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(
        paths.inventory_dir / "relevant_ranges.json",
        [asdict(item) for item in relevant_segments],
    )
    _write_json(
        paths.inventory_dir / "full_ranges.json",
        [asdict(item) for item in full_segments],
    )
    return {
        "books": books,
        "pages": len(page_rows),
        "relevant_segments": len(relevant_segments),
        "full_segments": len(full_segments),
        "candidate_pages": sum(row["candidate"] for row in page_rows),
    }


def load_segments(paths: RAGAnythingPaths, scope: str) -> list[dict]:
    source = paths.inventory_dir / f"{scope}_ranges.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Inventory is missing: {source}. Run inventory first."
        )
    return json.loads(source.read_text(encoding="utf-8"))


def extract_segment_pdf(
    paths: RAGAnythingPaths,
    segment: dict,
) -> Path:
    target_dir = paths.segment_pdf_dir / segment["book_id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{segment['segment_id']}.pdf"
    if target.is_file():
        return target
    source = fitz.open(paths.pdf_dir / segment["source_file"])
    output = fitz.open()
    try:
        output.insert_pdf(
            source,
            from_page=segment["start_page"] - 1,
            to_page=segment["end_page"] - 1,
        )
        output.save(target, garbage=4, deflate=True)
    finally:
        output.close()
        source.close()
    return target
