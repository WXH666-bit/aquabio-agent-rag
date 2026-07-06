from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import fitz

from .config import RAGAnythingPaths
from .inventory import load_segments


PROVENANCE_RE = re.compile(
    r"\[DOC_ID=(?P<doc_id>[^\]]+)\]"
    r"\[SOURCE=(?P<source>[^\]]+)\]"
    r"\[PAGE=(?P<page>\d+)\]"
)


def _normalize(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return normalized.replace("- ", "").replace(" -", "-")


def inspect_segment(
    paths: RAGAnythingPaths,
    segment_id: str,
    scope: str = "relevant",
) -> dict:
    segments = load_segments(paths, scope)
    segment = next(
        (item for item in segments if item["segment_id"] == segment_id),
        None,
    )
    if segment is None:
        raise ValueError(f"Unknown segment ID: {segment_id}")

    parser_dir = (
        paths.parser_output_dir
        / segment["book_id"]
        / segment["segment_id"]
    )
    filtered_path = parser_dir / "filtered_content_list.json"
    if not filtered_path.is_file():
        raise FileNotFoundError(
            f"Filtered content_list does not exist: {filtered_path}"
        )
    items = json.loads(filtered_path.read_text(encoding="utf-8"))
    pdf = fitz.open(paths.pdf_dir / segment["source_file"])
    page_results = []
    errors = []
    try:
        for page_number in range(
            segment["start_page"], segment["end_page"] + 1
        ):
            page_index = page_number - 1
            original_text = _normalize(
                pdf.load_page(page_index).get_text("text")
            )
            blocks = [
                item
                for item in items
                if int(item.get("page_idx", -1)) == page_index
            ]
            text_blocks = [
                item for item in blocks if item.get("type") == "text"
            ]
            matched_blocks = 0
            for block in text_blocks:
                text = str(block.get("text", ""))
                match = PROVENANCE_RE.search(text)
                if not match:
                    errors.append(
                        f"Missing provenance on page {page_number}"
                    )
                    continue
                if (
                    match.group("doc_id") != segment["doc_id"]
                    or match.group("source") != segment["source_file"]
                    or int(match.group("page")) != page_number
                ):
                    errors.append(
                        f"Incorrect provenance on page {page_number}: "
                        f"{match.group(0)}"
                    )
                body = PROVENANCE_RE.sub("", text, count=1).strip()
                normalized_body = _normalize(body)
                if normalized_body and normalized_body in original_text:
                    matched_blocks += 1
                elif normalized_body:
                    errors.append(
                        f"Text not found verbatim on page {page_number}: "
                        f"{body[:160]}"
                    )
            page_results.append(
                {
                    "page": page_number,
                    "text_blocks": len(text_blocks),
                    "exact_text_blocks": matched_blocks,
                    "images": sum(
                        item.get("type") == "image" for item in blocks
                    ),
                    "tables": sum(
                        item.get("type") == "table" for item in blocks
                    ),
                }
            )
    finally:
        pdf.close()

    missing_images = [
        item.get("img_path")
        for item in items
        if item.get("type") == "image"
        and not Path(str(item.get("img_path", ""))).is_file()
    ]
    if missing_images:
        errors.append(f"Missing image files: {len(missing_images)}")
    counts = Counter(item.get("type", "unknown") for item in items)
    total_text = sum(row["text_blocks"] for row in page_results)
    exact_text = sum(row["exact_text_blocks"] for row in page_results)
    return {
        "segment": segment,
        "content_list": str(filtered_path),
        "counts": dict(counts),
        "pages": page_results,
        "text_exact_match_rate": (
            round(exact_text / total_text, 4) if total_text else 0.0
        ),
        "missing_images": missing_images,
        "valid": not errors and exact_text == total_text,
        "errors": errors,
    }
