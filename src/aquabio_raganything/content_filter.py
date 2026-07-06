from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def prepare_content_list(
    content_list: list[dict[str, Any]],
    segment: dict,
    min_image_size: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prepared = []
    seen_images: set[str] = set()
    stats = {
        "text": 0,
        "image": 0,
        "table": 0,
        "equation": 0,
        "chart": 0,
        "dropped_metadata": 0,
        "rejected_images": 0,
    }
    prefix = (
        f"[DOC_ID={segment['doc_id']}]"
        f"[SOURCE={segment['source_file']}]"
    )

    for original in content_list:
        if not isinstance(original, dict):
            continue
        item = dict(original)
        kind = str(item.get("type", "text")).lower()
        if kind == "page_number":
            stats["dropped_metadata"] += 1
            continue
        if kind == "header":
            kind = "text"
            item["type"] = "text"
            item["text"] = item.get("text") or item.get("content") or ""
        if kind == "chart":
            kind = "image"
            item["type"] = "image"
            item["image_caption"] = _as_list(
                item.get("chart_caption") or item.get("caption")
            )
            item["image_footnote"] = _as_list(
                item.get("chart_footnote")
            )
            stats["chart"] += 1
        local_page = int(item.get("page_idx", 0) or 0)
        global_index = segment["start_page"] - 1 + local_page
        global_page = global_index + 1
        provenance = (
            f"doc_id={segment['doc_id']}; page={global_page}; "
            f"source={segment['source_file']}"
        )
        item["page_idx"] = global_index

        if kind == "text":
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            item["text"] = (
                f"{prefix}[PAGE={global_page}]\n{text}"
            )
            stats["text"] += 1
        elif kind == "image":
            raw_path = item.get("img_path") or item.get("image_path")
            if not raw_path:
                stats["rejected_images"] += 1
                continue
            image_path = Path(raw_path).resolve()
            try:
                with Image.open(image_path) as image:
                    if (
                        image.width < min_image_size
                        or image.height < min_image_size
                    ):
                        stats["rejected_images"] += 1
                        continue
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except (OSError, ValueError):
                stats["rejected_images"] += 1
                continue
            if digest in seen_images:
                stats["rejected_images"] += 1
                continue
            seen_images.add(digest)
            item["img_path"] = str(image_path)
            item["image_footnote"] = [
                *_as_list(item.get("image_footnote")),
                provenance,
            ]
            stats["image"] += 1
        elif kind == "table":
            body = str(
                item.get("table_body")
                or item.get("table_content")
                or ""
            ).strip()
            if not body:
                continue
            item["table_body"] = body
            item["table_footnote"] = [
                *_as_list(item.get("table_footnote")),
                provenance,
            ]
            stats["table"] += 1
        elif kind == "equation":
            if not (item.get("latex") or item.get("text")):
                continue
            item["text"] = (
                f"{prefix}[PAGE={global_page}]\n"
                f"{str(item.get('text', '')).strip()}"
            )
            stats["equation"] += 1
        else:
            item["content"] = (
                f"{prefix}[PAGE={global_page}]\n"
                f"{str(item.get('content', '')).strip()}"
            )
        prepared.append(item)

    if not stats["text"]:
        raise ValueError("Filtered content_list contains no text blocks.")
    return prepared, stats
