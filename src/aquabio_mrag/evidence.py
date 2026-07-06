from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    id: str
    source: str
    doc_id: str = ""
    page: int | None = None
    source_file: str = ""
    modality: str = "text"
    content: str
    rank: int = 0
    score: float = 0.0
    entity_names: list[str] = Field(default_factory=list)
    relation_path: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        digest = hashlib.sha256(
            " ".join(self.content.casefold().split()).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.doc_id}|{self.page}|{digest}"


def weighted_rrf(
    ranked_sources: dict[str, list[EvidenceItem]],
    weights: dict[str, float],
    top_k: int = 12,
    constant: int = 60,
) -> list[EvidenceItem]:
    fused: dict[str, EvidenceItem] = {}
    scores: dict[str, float] = {}
    source_names: dict[str, set[str]] = {}
    for source, rows in ranked_sources.items():
        weight = weights.get(source, 0.0)
        for rank, row in enumerate(rows, start=1):
            key = row.dedupe_key
            scores[key] = scores.get(key, 0.0) + weight / (constant + rank)
            source_names.setdefault(key, set()).add(source)
            if key not in fused or len(row.content) > len(fused[key].content):
                fused[key] = row
    ordered = sorted(fused, key=lambda key: scores[key], reverse=True)
    result = []
    for rank, key in enumerate(ordered[:top_k], start=1):
        row = fused[key].model_copy(deep=True)
        row.rank = rank
        row.score = scores[key]
        row.metadata["fused_sources"] = sorted(source_names[key])
        result.append(row)
    return result
