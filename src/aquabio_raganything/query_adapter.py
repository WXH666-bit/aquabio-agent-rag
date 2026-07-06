from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import networkx as nx

from .config import RAGAnythingPaths, RAGAnythingSettings
from .manifest import SegmentManifest
from .runtime import create_rag, ensure_initialized
from .storage_audit import audit_persistent_storages
from .image_rag import entity_image_records


PROVENANCE_RE = re.compile(
    r"\[DOC_ID=(?P<doc_id>[^\]]+)\]"
    r"\[SOURCE=(?P<source_file>[^\]]+)\]"
    r"\[PAGE=(?P<page>\d+)\]"
)
ENTITY_SECTION_RE = re.compile(
    r"Knowledge Graph Data \(Entity\):\s*```json\s*(.*?)```",
    re.DOTALL,
)
RELATION_SECTION_RE = re.compile(
    r"Knowledge Graph Data \(Relationship\):\s*```json\s*(.*?)```",
    re.DOTALL,
)

_QUERY_RAGS: dict[str, Any] = {}
_QUERY_RAG_READY: set[str] = set()
_QUERY_RAG_LOCK: asyncio.Lock | None = None


async def _query_rag(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
):
    global _QUERY_RAG_LOCK
    key = str(paths.working_dir.resolve())
    if key in _QUERY_RAG_READY:
        return _QUERY_RAGS[key]
    if _QUERY_RAG_LOCK is None:
        _QUERY_RAG_LOCK = asyncio.Lock()
    async with _QUERY_RAG_LOCK:
        if key not in _QUERY_RAGS:
            _QUERY_RAGS[key] = create_rag(paths, settings)
        if key not in _QUERY_RAG_READY:
            await ensure_initialized(_QUERY_RAGS[key])
            _QUERY_RAG_READY.add(key)
    return _QUERY_RAGS[key]


def _parse_json_lines(section: str) -> list[dict]:
    values = []
    decoder = json.JSONDecoder()
    position = 0
    while position < len(section):
        while position < len(section) and (
            section[position].isspace() or section[position] == ","
        ):
            position += 1
        if position >= len(section):
            break
        try:
            value, position = decoder.raw_decode(section, position)
        except json.JSONDecodeError:
            next_line = section.find("\n", position)
            position = len(section) if next_line < 0 else next_line + 1
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _parse_graph_context(content: str) -> tuple[list[dict], list[dict]]:
    entity_match = ENTITY_SECTION_RE.search(content)
    relation_match = RELATION_SECTION_RE.search(content)
    entities = (
        _parse_json_lines(entity_match.group(1)) if entity_match else []
    )
    relations = (
        _parse_json_lines(relation_match.group(1)) if relation_match else []
    )
    return entities, relations


def _build_evidence(
    content: str,
    entities: list[dict],
    relations: list[dict],
    top_k: int,
) -> list[dict]:
    matches = list(PROVENANCE_RE.finditer(content))
    pages: dict[tuple[str, str, int], list[str]] = {}
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )
        body = content[match.end() : end].strip()
        if not body or body.startswith("```"):
            continue
        key = (
            match.group("doc_id"),
            match.group("source_file"),
            int(match.group("page")),
        )
        if body not in pages.setdefault(key, []):
            pages[key].append(body)

    evidence = []
    entity_names = [
        str(item.get("entity", ""))
        for item in entities
        if item.get("entity")
    ]
    relation_paths = [
        [
            str(item.get("entity1", "")),
            str(item.get("description", "")),
            str(item.get("entity2", "")),
        ]
        for item in relations
        if item.get("entity1") and item.get("entity2")
    ]
    for rank, ((doc_id, source_file, page), bodies) in enumerate(
        pages.items(), start=1
    ):
        excerpt = "\n".join(bodies)
        mentioned = [
            name for name in entity_names if name.casefold() in excerpt.casefold()
        ][:8]
        paths = [
            path
            for path in relation_paths
            if path[0] in mentioned or path[2] in mentioned
        ]
        evidence.append(
            {
                "id": "graph_"
                + hashlib.sha256(excerpt.encode("utf-8")).hexdigest()[:12],
                "source": "lightrag_graph",
                "doc_id": doc_id,
                "page": page,
                "source_file": source_file,
                "modality": "mixed",
                "entity_names": mentioned,
                "relation_path": paths[0] if paths else [],
                "content": excerpt,
                "rank": rank,
            }
        )
        if len(evidence) >= top_k:
            break
    return evidence


def _graph_path(paths: RAGAnythingPaths) -> Path | None:
    candidates = sorted(paths.working_dir.glob("*.graphml"))
    if not candidates:
        candidates = sorted(paths.working_dir.rglob("*.graphml"))
    return candidates[0] if candidates else None


def graph_neighbors(
    paths: RAGAnythingPaths,
    entity: str,
    depth: int = 1,
) -> dict:
    depth = max(1, min(2, depth))
    fallback_detail = source_detail(paths, entity=entity, top_k=24)
    fallback_relations = [
        {
            "source": row.get("source", ""),
            "target": row.get("target", ""),
            "relation": row.get("relation", ""),
            "description": row.get("evidence", "")
            or row.get("relation", ""),
            "doc_id": row.get("doc_id", ""),
            "page": row.get("page"),
            "unit_id": row.get("unit_id", ""),
        }
        for row in fallback_detail.get("relations", [])
    ]
    fallback_nodes = []
    seen_nodes = set()
    for relation in fallback_relations:
        for key in ("source", "target"):
            value = str(relation.get(key, "")).strip()
            if value and value not in seen_nodes:
                fallback_nodes.append({"id": value, "source": "book_native"})
                seen_nodes.add(value)
    graph_path = _graph_path(paths)
    if graph_path is None:
        images = entity_image_records(paths, entity, top_k=8)
        return {
            "entity": entity,
            "nodes": fallback_nodes,
            "relations": fallback_relations,
            "images": images,
            "warnings": (
                ["NetworkX GraphML index does not exist yet; used book-native triples."]
                if fallback_relations
                else ["NetworkX GraphML index does not exist yet."]
            ),
        }
    graph = nx.read_graphml(graph_path)
    entity_lower = entity.lower()
    starts = [
        node
        for node, data in graph.nodes(data=True)
        if entity_lower in str(node).lower()
        or entity_lower in str(data.get("entity_name", "")).lower()
    ]
    selected = set(starts)
    frontier = set(starts)
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.neighbors(node))
        next_frontier -= selected
        selected.update(next_frontier)
        frontier = next_frontier
    nodes = [
        {"id": str(node), **{k: str(v) for k, v in graph.nodes[node].items()}}
        for node in selected
    ]
    relations = []
    for source, target, data in graph.edges(data=True):
        if source in selected and target in selected:
            relations.append(
                {
                    "source": str(source),
                    "target": str(target),
                    **{key: str(value) for key, value in data.items()},
                }
            )
    images = entity_image_records(paths, entity, top_k=8)
    if not starts and fallback_relations:
        return {
            "entity": entity,
            "depth": depth,
            "nodes": fallback_nodes,
            "relations": fallback_relations,
            "graph_file": str(graph_path),
            "images": images,
            "warnings": [
                f"Entity not found in GraphML: {entity}; used book-native triples."
            ],
        }
    return {
        "entity": entity,
        "depth": depth,
        "nodes": nodes,
        "relations": relations,
        "graph_file": str(graph_path),
        "images": images,
        "warnings": [] if starts else [f"Entity not found: {entity}"],
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_detail(
    paths: RAGAnythingPaths,
    doc_id: str = "",
    entity: str = "",
    image_id: str = "",
    top_k: int = 12,
) -> dict[str, Any]:
    """Return page/source details from the book-native PDF registries."""
    needle_values = [
        value.casefold().strip()
        for value in (doc_id, entity, image_id)
        if value and value.strip()
    ]
    if not needle_values:
        return {
            "query": {"doc_id": doc_id, "entity": entity, "image_id": image_id},
            "units": [],
            "chunks": [],
            "images": [],
            "relations": [],
            "warnings": ["No doc_id, entity or image_id was provided."],
        }

    book_root = paths.book_native_dir / "sa_invertebrates"
    image_root = paths.extracted_assets_dir / "sa_invertebrates" / "image_index"
    units = _jsonl_rows(book_root / "species_page_units.jsonl")
    chunks = _jsonl_rows(book_root / "rag_chunks.jsonl")
    relations = _jsonl_rows(book_root / "relation_triples.jsonl")
    images = _jsonl_rows(image_root / "pdf_image_captions.jsonl")

    def matches(row: dict[str, Any]) -> bool:
        aliases: list[Any] = [
            row.get("doc_id", ""),
            row.get("unit_id", ""),
            row.get("chunk_id", ""),
            row.get("image_id", ""),
            row.get("entity_id", ""),
            row.get("scientific_name", ""),
            row.get("common_name", ""),
            row.get("fb_code", ""),
            row.get("title", ""),
            row.get("genus", ""),
            row.get("family", ""),
            row.get("order", ""),
            row.get("class", ""),
            row.get("phylum", ""),
            row.get("source", ""),
            row.get("target", ""),
            *row.get("entity_names", []),
        ]
        taxonomy = row.get("taxonomy") or {}
        aliases.extend(taxonomy.values())
        haystack = " | ".join(str(value) for value in aliases if value)
        lowered = haystack.casefold()
        return any(needle in lowered for needle in needle_values)

    matched_units = [row for row in units if matches(row)]
    matched_chunks = [row for row in chunks if matches(row)]
    matched_images = [row for row in images if matches(row)]
    matched_relations = [row for row in relations if matches(row)]
    if doc_id and not matched_units:
        matched_units = [row for row in units if row.get("doc_id") == doc_id]
    return {
        "query": {"doc_id": doc_id, "entity": entity, "image_id": image_id},
        "units": matched_units[:top_k],
        "chunks": matched_chunks[:top_k],
        "images": matched_images[:top_k],
        "relations": matched_relations[:top_k],
        "counts": {
            "units": len(matched_units),
            "chunks": len(matched_chunks),
            "images": len(matched_images),
            "relations": len(matched_relations),
        },
        "warnings": [] if any(
            (matched_units, matched_chunks, matched_images, matched_relations)
        ) else ["No source detail matched the query."],
    }


def index_status(paths: RAGAnythingPaths) -> dict:
    rows = SegmentManifest(
        paths.manifests_dir / "segment_status.jsonl"
    ).rows()
    latest = {}
    for row in rows:
        latest[row["segment_id"]] = row
    status_counts: dict[str, int] = {}
    for row in latest.values():
        status = row.get("index_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    native_rows = SegmentManifest(
        paths.manifests_dir / "book_native_status.jsonl"
    ).rows()
    native_latest = {}
    for row in native_rows:
        native_latest[row["segment_id"]] = row
    native_status_counts: dict[str, int] = {}
    for row in native_latest.values():
        status = row.get("index_status", "unknown")
        native_status_counts[status] = (
            native_status_counts.get(status, 0) + 1
        )
    storage = audit_persistent_storages(paths)
    files = [path.name for path in paths.working_dir.glob("*") if path.is_file()]
    graph_nodes = storage["counts"]["graph_nodes"]
    graph_edges = storage["counts"]["graph_edges"]
    if graph_nodes == 0 and graph_edges == 0:
        for book_dir in paths.book_native_dir.iterdir():
            triples_file = book_dir / "relation_triples.jsonl"
            if triples_file.is_file():
                with triples_file.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            graph_edges += 1
                            item = json.loads(line)
                            if item.get("src_id"):
                                graph_nodes += 1
                            if item.get("tgt_id"):
                                graph_nodes += 1
        unique_nodes = set()
        for book_dir in paths.book_native_dir.iterdir():
            triples_file = book_dir / "relation_triples.jsonl"
            if triples_file.is_file():
                with triples_file.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            for key in ("src_id", "tgt_id"):
                                if item.get(key):
                                    unique_nodes.add(item[key])
        if unique_nodes:
            graph_nodes = len(unique_nodes)
    return {
        "segments_total": len(latest),
        "status_counts": status_counts,
        "book_native_units_total": len(native_latest),
        "book_native_status_counts": native_status_counts,
        "entities": storage["counts"]["entities"],
        "relations": storage["counts"]["relationships"],
        "chunks": storage["counts"]["chunks"],
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "storage_valid": storage["valid"],
        "storage_missing": storage["missing"],
        "working_files": sorted(files),
        "last_updated": max(
            (row.get("updated_at", "") for row in latest.values()),
            default="",
        ),
    }


async def hybrid_search(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
    query: str,
    top_k: int = 12,
) -> dict:
    started = time.perf_counter()
    rag = await _query_rag(paths, settings)
    initialized_seconds = time.perf_counter() - started
    query_timeout = float(
        os.getenv("RAGANYTHING_QUERY_TIMEOUT", "25")
    )
    context = await asyncio.wait_for(
        rag.aquery(
            query,
            mode="hybrid",
            only_need_context=True,
            top_k=top_k,
            chunk_top_k=top_k,
            vlm_enhanced=False,
        ),
        timeout=query_timeout,
    )
    query_seconds = time.perf_counter() - started - initialized_seconds
    try:
        content = (
            context
            if isinstance(context, str)
            else json.dumps(context, ensure_ascii=False)
        )
        if context is None or content.strip() in {"", "null"}:
            return {
                "query": query,
                "entities": [],
                "relations": [],
                "chunks": [],
                "evidence": [],
                "raw_context": context,
                "warnings": [
                    "LightRAG returned no context; retrieval was not successful."
                ],
                "timing": {
                    "initialize_seconds": round(initialized_seconds, 3),
                    "query_seconds": round(query_seconds, 3),
                },
            }
        entities, relations = _parse_graph_context(content)
        evidence = _build_evidence(
            content, entities, relations, top_k
        )
        if not evidence:
            evidence.append(
                {
                    "id": "graph_"
                    + hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest()[:12],
                    "source": "lightrag_graph",
                    "doc_id": "",
                    "page": None,
                    "source_file": "",
                    "modality": "mixed",
                    "entity_names": [],
                    "relation_path": [],
                    "content": content,
                    "rank": 1,
                }
            )
        return {
            "query": query,
            "entities": entities,
            "relations": relations,
            "chunks": [],
            "evidence": evidence,
            "raw_context": context,
            "warnings": [],
            "timing": {
                "initialize_seconds": round(initialized_seconds, 3),
                "query_seconds": round(query_seconds, 3),
            },
        }
    except Exception:
        raise
