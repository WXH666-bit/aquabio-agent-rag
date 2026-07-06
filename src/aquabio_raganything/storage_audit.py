from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from .config import RAGAnythingPaths


def _json_count(path: Path) -> int:
    if not path.is_file():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, list):
            return len(data)
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 0


def audit_persistent_storages(
    paths: RAGAnythingPaths,
    doc_id: str | None = None,
) -> dict[str, Any]:
    working = paths.working_dir
    graph_files = sorted(working.glob("*.graphml"))
    graph_nodes = 0
    graph_edges = 0
    graph_file = ""
    if graph_files:
        graph_file = str(graph_files[0])
        graph = nx.read_graphml(graph_files[0])
        graph_nodes = graph.number_of_nodes()
        graph_edges = graph.number_of_edges()

    doc_status_path = working / "kv_store_doc_status.json"
    doc_status = {}
    if doc_status_path.is_file():
        statuses = json.loads(doc_status_path.read_text(encoding="utf-8"))
        doc_status = statuses.get(doc_id, {}) if doc_id else statuses

    counts = {
        "chunks": _json_count(working / "vdb_chunks.json"),
        "entities": _json_count(working / "vdb_entities.json"),
        "relationships": _json_count(
            working / "vdb_relationships.json"
        ),
        "text_chunks": _json_count(
            working / "kv_store_text_chunks.json"
        ),
        "full_entities": _json_count(
            working / "kv_store_full_entities.json"
        ),
        "full_relations": _json_count(
            working / "kv_store_full_relations.json"
        ),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
    }
    required = {
        "document_status": bool(doc_status),
        "chunk_vector_store": counts["chunks"] > 0,
        "text_chunk_store": counts["text_chunks"] > 0,
        "entity_vector_store": counts["entities"] > 0,
        "relationship_vector_store": counts["relationships"] > 0,
        "networkx_graph": graph_nodes > 0 and graph_edges > 0,
        "full_entity_metadata": counts["full_entities"] > 0,
        "full_relation_metadata": counts["full_relations"] > 0,
    }
    return {
        "valid": all(required.values()),
        "required": required,
        "counts": counts,
        "doc_status": doc_status,
        "graph_file": graph_file,
        "working_dir": str(working),
        "missing": [
            name for name, available in required.items() if not available
        ],
    }
