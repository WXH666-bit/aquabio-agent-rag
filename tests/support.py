"""Temporary vector storage and deterministic retrieval for offline workflow tests."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def isolated_paths(paths, directory):
    return replace(paths, vector_dir=Path(directory) / "chroma", sessions_dir=Path(directory) / "sessions")


def fixture_search(request):
    root = Path(__file__).resolve().parents[1]
    rows = []
    for line in (root / "data/mrag/knowledge/rag_documents_combined.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if request.species_ids and row.get("species_id") not in request.species_ids:
            continue
        if request.source_types and row["source_type"] not in request.source_types:
            continue
        rows.append({"id": row["id"], "content": row["content"],
                     "metadata": {**row.get("metadata", {}), "species_id": row.get("species_id", ""), "source_type": row["source_type"]},
                     "semantic_similarity": 0.9, "final_score": 0.9})
    return rows[:request.top_k]
