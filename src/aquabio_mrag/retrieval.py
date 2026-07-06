from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .config import MRAGPaths, MRAGSettings
from .vector_db import ChromaMRAGStore


RetrievalTask = Literal[
    "text_qa",
    "followup_text_qa",
    "image_qa",
    "multimodal_qa",
    "comparison_qa",
    "source_trace",
    "pdf_qa",
]

SOURCE_WEIGHTS = {
    "followup_text_qa": {
        "species_text_chunk": 1.0,
        "species_card": 0.95,
        "pdf_chunk": 0.45,
        "multimodal_pair": 0.35,
        "image_doc": 0.25,
    },
    "text_qa": {
        "species_text_chunk": 1.0,
        "species_card": 0.9,
        "pdf_chunk": 0.85,
        "multimodal_pair": 0.65,
        "image_doc": 0.45,
    },
    "image_qa": {
        "multimodal_pair": 1.0,
        "image_doc": 0.95,
        "species_text_chunk": 0.85,
        "species_card": 0.7,
        "pdf_chunk": 0.6,
    },
    "multimodal_qa": {
        "multimodal_pair": 1.0,
        "species_text_chunk": 0.9,
        "image_doc": 0.85,
        "species_card": 0.75,
        "pdf_chunk": 0.75,
    },
    "comparison_qa": {
        "species_text_chunk": 1.0,
        "species_card": 0.85,
        "multimodal_pair": 0.75,
        "pdf_chunk": 0.7,
        "image_doc": 0.5,
    },
    "source_trace": {
        "pdf_chunk": 1.0,
        "species_text_chunk": 0.9,
        "image_doc": 0.9,
        "species_card": 0.8,
        "multimodal_pair": 0.75,
    },
    "pdf_qa": {
        "pdf_chunk": 1.0,
        "pdf_figure": 0.85,
        "species_text_chunk": 0.5,
        "species_card": 0.4,
        "multimodal_pair": 0.3,
        "image_doc": 0.2,
    },
}

CHUNK_WEIGHTS = {
    "followup_text_qa": {
        "overview": 1.0,
        "visual_features": 1.0,
        "image_recognition_tips": 0.9,
        "taxonomy": 0.85,
        "habitat": 0.85,
        "ecology_behavior": 0.85,
    },
    "comparison_qa": {
        "similar_species": 1.0,
        "visual_features": 0.95,
        "image_recognition_tips": 0.9,
    },
    "image_qa": {
        "visual_features": 1.0,
        "image_recognition_tips": 0.95,
        "similar_species": 0.85,
    },
    "text_qa": {
        "overview": 0.9,
        "taxonomy": 0.9,
        "habitat": 0.9,
        "ecology_behavior": 0.9,
    },
}


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    english = re.findall(r"[a-z0-9_]{2,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = ["".join(chinese[i : i + 2]) for i in range(len(chinese) - 1)]
    return set(english + bigrams)


@dataclass
class RetrievalRequest:
    query: str
    task_type: RetrievalTask
    top_k: int = 12
    species_ids: list[str] | None = None
    source_types: list[str] | None = None


class MultiSourceRetriever:
    def __init__(self, paths: MRAGPaths, settings: MRAGSettings):
        self.paths = paths
        self.settings = settings
        self.store = ChromaMRAGStore(paths, settings)

    @staticmethod
    def _where(request: RetrievalRequest) -> dict | None:
        clauses = []
        if request.species_ids:
            clauses.append({"species_id": {"$in": request.species_ids}})
        if request.source_types:
            clauses.append({"source_type": {"$in": request.source_types}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def search(self, request: RetrievalRequest) -> list[dict]:
        initial_k = max(request.top_k, self.settings.top_k)
        rows = self.store.query(
            request.query,
            top_k=initial_k,
            where=self._where(request),
        )
        query_terms = _terms(request.query)
        source_weights = SOURCE_WEIGHTS[request.task_type]
        chunk_weights = CHUNK_WEIGHTS.get(request.task_type, {})
        candidates = set(request.species_ids or [])

        for row in rows:
            metadata = row["metadata"]
            source_type = metadata.get("source_type", "")
            species_id = metadata.get("species_id", "")
            chunk_type = metadata.get("chunk_type", "")
            content_terms = _terms(row["content"])
            keyword_overlap = len(query_terms & content_terms) / max(
                1, len(query_terms)
            )
            species_match = 1.0 if candidates and species_id in candidates else (
                0.5 if not candidates else 0.0
            )
            source_weight = source_weights.get(source_type, 0.4)
            chunk_weight = chunk_weights.get(chunk_type, 0.6)
            row["final_score"] = round(
                0.45 * row["semantic_similarity"]
                + 0.20 * species_match
                + 0.15 * source_weight
                + 0.10 * chunk_weight
                + 0.10 * keyword_overlap,
                6,
            )

        rows.sort(key=lambda item: item["final_score"], reverse=True)
        return rows[: request.top_k]

    def text_search(
        self, query: str, top_k: int = 6, species_ids: list[str] | None = None
    ) -> list[dict]:
        return self.search(
            RetrievalRequest(
                query=query,
                task_type="text_qa",
                top_k=top_k,
                species_ids=species_ids,
                source_types=["species_card", "species_text_chunk"],
            )
        )

    def image_search(
        self,
        caption: str,
        top_k: int = 6,
        species_ids: list[str] | None = None,
    ) -> list[dict]:
        return self.search(
            RetrievalRequest(
                query=caption,
                task_type="image_qa",
                top_k=top_k,
                species_ids=species_ids,
                source_types=["image_doc", "multimodal_pair"],
            )
        )

    def multimodal_search(
        self,
        query: str,
        caption: str,
        top_k: int = 8,
        species_ids: list[str] | None = None,
    ) -> list[dict]:
        return self.search(
            RetrievalRequest(
                query=f"{query}\nImage caption: {caption}",
                task_type="multimodal_qa",
                top_k=top_k,
                species_ids=species_ids,
                source_types=[
                    "multimodal_pair",
                    "image_doc",
                    "species_text_chunk",
                    "species_card",
                    "pdf_chunk",
                ],
            )
        )

    def pdf_search(
        self,
        query: str,
        top_k: int = 6,
        species_ids: list[str] | None = None,
    ) -> list[dict]:
        return self.search(
            RetrievalRequest(
                query=query,
                task_type="pdf_qa",
                top_k=top_k,
                species_ids=species_ids,
                source_types=["pdf_chunk", "pdf_figure"],
            )
        )
