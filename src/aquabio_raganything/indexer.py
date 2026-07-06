from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from .config import BOOKS, RAGAnythingPaths, RAGAnythingSettings
from .content_filter import prepare_content_list
from .inventory import extract_segment_pdf, load_segments
from .manifest import SegmentManifest
from .runtime import create_rag, ensure_initialized
from .storage_audit import audit_persistent_storages


def _load_mineru_content(parser_dir: Path) -> list[dict] | None:
    candidates = sorted(
        parser_dir.rglob("*_content_list.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for content_path in candidates:
        content = json.loads(content_path.read_text(encoding="utf-8"))
        if not isinstance(content, list) or not content:
            continue
        valid = True
        normalized = []
        for original in content:
            item = dict(original)
            raw_image = item.get("img_path") or item.get("image_path")
            if raw_image:
                image_path = Path(raw_image)
                if not image_path.is_absolute():
                    image_path = content_path.parent / image_path
                if not image_path.is_file():
                    valid = False
                    break
                item["img_path"] = str(image_path.resolve())
            normalized.append(item)
        if valid:
            return normalized
    return None


def _run_mineru_windows_fallback(
    segment_pdf: Path,
    parser_dir: Path,
    settings: RAGAnythingSettings,
) -> list[dict]:
    recovery_dir = parser_dir / "windows_cli_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "mineru",
        "-p",
        str(segment_pdf),
        "-o",
        str(recovery_dir),
        "-m",
        "txt" if settings.parse_method == "auto" else settings.parse_method,
        "-b",
        settings.parser_backend,
        "-l",
        "en",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    log_path = recovery_dir / "mineru_windows.log"
    log_path.write_text(
        f"return_code={completed.returncode}\n"
        f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    recovered = _load_mineru_content(recovery_dir)
    if recovered is None:
        raise RuntimeError(
            "MinerU did not produce a complete content_list. "
            f"See {log_path}"
        )
    return recovered


async def index_segments(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
    scope: str,
    resume: bool = False,
    limit: int | None = None,
    segment_id: str | None = None,
) -> dict:
    scripts_dir = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if scripts_dir.lower() not in {item.lower() for item in path_entries}:
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get(
            "PATH", ""
        )
    segments = load_segments(paths, scope)
    if segment_id:
        segments = [
            item for item in segments if item["segment_id"] == segment_id
        ]
        if not segments:
            raise ValueError(f"Unknown segment ID: {segment_id}")
    if limit is not None:
        segments = segments[:limit]

    manifest = SegmentManifest(
        paths.manifests_dir / "segment_status.jsonl"
    )
    rag = create_rag(paths, settings)
    summary = {
        "requested": len(segments),
        "fully_processed": 0,
        "failed": 0,
        "skipped": 0,
    }
    try:
        for segment in segments:
            previous = manifest.get(segment["segment_id"])
            unchanged = (
                previous
                and previous.get("source_sha256")
                == segment["source_sha256"]
            )
            if (
                resume
                and unchanged
                and previous.get("index_status") == "fully_processed"
            ):
                summary["skipped"] += 1
                continue
            manifest.update(segment, "indexing", error="")
            try:
                segment_pdf = extract_segment_pdf(paths, segment)
                parser_dir = (
                    paths.parser_output_dir
                    / segment["book_id"]
                    / segment["segment_id"]
                )
                parser_dir.mkdir(parents=True, exist_ok=True)
                cache_path = parser_dir / "filtered_content_list.json"
                if cache_path.is_file():
                    prepared = json.loads(
                        cache_path.read_text(encoding="utf-8")
                    )
                    stats = {
                        "text": sum(
                            item.get("type") == "text" for item in prepared
                        ),
                        "image": sum(
                            item.get("type") == "image" for item in prepared
                        ),
                        "table": sum(
                            item.get("type") == "table" for item in prepared
                        ),
                        "equation": sum(
                            item.get("type") == "equation"
                            for item in prepared
                        ),
                        "chart": 0,
                        "dropped_metadata": 0,
                        "rejected_images": 0,
                    }
                else:
                    try:
                        content_list, _ = await rag.parse_document(
                            file_path=str(segment_pdf),
                            output_dir=str(parser_dir),
                            parse_method=settings.parse_method,
                            display_stats=True,
                            backend=settings.parser_backend,
                            lang="en",
                        )
                    except Exception:
                        content_list = _load_mineru_content(parser_dir)
                        if content_list is None:
                            content_list = _run_mineru_windows_fallback(
                                segment_pdf,
                                parser_dir,
                                settings,
                            )
                    prepared, stats = prepare_content_list(
                        content_list,
                        segment,
                        min_image_size=settings.min_image_size,
                    )
                    cache_path.write_text(
                        json.dumps(
                            prepared,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                await rag.insert_content_list(
                    content_list=prepared,
                    file_path=segment["source_file"],
                    doc_id=segment["doc_id"],
                    display_stats=True,
                )
                # RAG-Anything can overwrite a failed text-extraction status
                # after multimodal processing. Persist every storage and verify
                # the actual files instead of trusting doc_status alone.
                await rag.lightrag._insert_done()
                processing = await rag.get_document_processing_status(
                    segment["doc_id"]
                )
                storage_audit = audit_persistent_storages(
                    paths, segment["doc_id"]
                )
                fully_processed = bool(
                    processing.get("fully_processed")
                    and storage_audit["valid"]
                )
                status = "fully_processed" if fully_processed else "failed"
                manifest.update(
                    segment,
                    status,
                    parse_status="completed",
                    text_items=stats["text"],
                    image_items=stats["image"],
                    table_items=stats["table"],
                    equation_items=stats["equation"],
                    chart_items=stats["chart"],
                    dropped_metadata=stats["dropped_metadata"],
                    rejected_images=stats["rejected_images"],
                    chunks_count=processing.get("chunks_count", 0),
                    storage_counts=storage_audit["counts"],
                    storage_missing=storage_audit["missing"],
                    traceback="",
                    error=""
                    if fully_processed
                    else (
                        "LightRAG storage validation failed: "
                        + ", ".join(storage_audit["missing"])
                    ),
                )
                summary[status] += 1
            except Exception as exc:
                manifest.update(
                    segment,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                summary["failed"] += 1
    finally:
        await rag.finalize_storages()
    return summary


async def index_book_native_units(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
    book_id: str,
    resume: bool = False,
    limit_units: int | None = None,
) -> dict:
    source = (
        paths.book_native_dir
        / book_id
        / "raganything_content_list.jsonl"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing. Run raganything_cli.py book-native first."
        )
    grouped: dict[str, list[dict]] = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            grouped.setdefault(item["unit_id"], []).append(item)
    unit_ids = list(grouped)
    if limit_units is not None:
        unit_ids = unit_ids[:limit_units]
    manifest = SegmentManifest(
        paths.manifests_dir / "book_native_status.jsonl"
    )
    rag = create_rag(paths, settings)
    await ensure_initialized(rag)
    summary = {
        "book_id": book_id,
        "requested": len(unit_ids),
        "fully_processed": 0,
        "failed": 0,
        "skipped": 0,
    }
    try:
        for unit_id in unit_ids:
            items = grouped[unit_id]
            doc_id = items[0]["doc_id"]
            record = {
                "segment_id": unit_id,
                "doc_id": doc_id,
                "book_id": book_id,
                "source_file": BOOKS[book_id],
                "start_page": items[0]["page_idx"] + 1,
                "end_page": items[0]["page_idx"] + 1,
                "scope": "book_native",
                "source_sha256": "",
                "matched_species": [],
            }
            previous = manifest.get(unit_id)
            if (
                resume
                and previous
                and previous.get("index_status") == "fully_processed"
            ):
                summary["skipped"] += 1
                continue
            manifest.update(record, "indexing", error="")
            try:
                content_list = [
                    {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "unit_id",
                            "doc_id",
                            "chunk_type",
                        }
                    }
                    for item in items
                ]
                await rag.insert_content_list(
                    content_list=content_list,
                    file_path=BOOKS[book_id],
                    doc_id=doc_id,
                    display_stats=False,
                )
                await rag.lightrag._insert_done()
                status = await rag.get_document_processing_status(doc_id)
                fully_processed = bool(status.get("fully_processed"))
                manifest.update(
                    record,
                    "fully_processed" if fully_processed else "failed",
                    parse_status="book_native",
                    text_items=len(content_list),
                    chunks_count=status.get("chunks_count", 0),
                    error="" if fully_processed else str(status),
                )
                summary[
                    "fully_processed" if fully_processed else "failed"
                ] += 1
            except Exception as error:
                manifest.update(
                    record,
                    "failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
                summary["failed"] += 1
    finally:
        await rag.finalize_storages()
    return summary
