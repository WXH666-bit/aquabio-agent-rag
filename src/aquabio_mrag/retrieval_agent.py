from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .evidence import EvidenceItem, weighted_rrf
from .mcp_client import MCPStdioClient, project_mcp_client
from .retrieval import MultiSourceRetriever


def _tokens(text: str) -> list[str]:
    lowered = text.casefold()
    english = re.findall(r"[a-z0-9_]{2,}", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]{1,2}", lowered)
    return english + chinese


class BookNativeBM25:
    def __init__(self, root: Path):
        self.paths = sorted(
            (
                root
                / "data"
                / "mrag"
                / "raganything"
                / "book_native"
            ).glob("*/rag_chunks.jsonl")
        )
        self._rows: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = []
            for path in self.paths:
                with path.open(encoding="utf-8") as stream:
                    self._rows.extend(
                        json.loads(line) for line in stream if line.strip()
                    )
        return self._rows

    def search(self, query: str, top_k: int = 12) -> list[EvidenceItem]:
        rows = self._load()
        query_tokens = _tokens(query)
        if not rows or not query_tokens:
            return []
        document_tokens = [_tokens(row.get("content", "")) for row in rows]
        document_frequency = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))
        average_length = sum(map(len, document_tokens)) / max(1, len(rows))
        scored = []
        for row, tokens in zip(rows, document_tokens):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                frequency = frequencies[term]
                if not frequency:
                    continue
                inverse = math.log(
                    1
                    + (len(rows) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(tokens) / max(1.0, average_length)
                )
                score += inverse * frequency * 2.5 / denominator
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceItem(
                id=row.get("chunk_id") or row.get("unit_id", ""),
                source="book_bm25",
                doc_id=row.get("doc_id", ""),
                page=row.get("page") or row.get("pdf_page"),
                source_file=row.get("source_file", ""),
                modality=row.get("modality", "text"),
                content=row.get("content", ""),
                rank=rank,
                score=score,
                metadata=row,
            )
            for rank, (score, row) in enumerate(scored[:top_k], start=1)
        ]


class RetrievalAgent:
    def __init__(
        self,
        root: Path,
        local_retriever: MultiSourceRetriever,
        mcp_client: MCPStdioClient | None = None,
    ):
        self.root = root
        self.local = local_retriever
        self.mcp = mcp_client or project_mcp_client(root)
        self.bm25 = BookNativeBM25(root)

    @staticmethod
    def _local_evidence(rows: list[dict[str, Any]]) -> list[EvidenceItem]:
        result = []
        for rank, row in enumerate(rows, start=1):
            metadata = row.get("metadata", {})
            result.append(
                EvidenceItem(
                    id=row["id"],
                    source="chroma",
                    doc_id=metadata.get("doc_id", row["id"]),
                    page=metadata.get("page"),
                    source_file=metadata.get("doc_title", ""),
                    modality=metadata.get("modality", "text"),
                    content=row.get("content", ""),
                    rank=rank,
                    score=row.get("final_score", 0.0),
                    metadata=metadata,
                )
            )
        return result

    @staticmethod
    def _graph_evidence(payload: Any) -> list[EvidenceItem]:
        if not isinstance(payload, dict):
            return []
        return [
            EvidenceItem(
                id=row.get("id", f"graph_{rank}"),
                source="lightrag_graph",
                doc_id=row.get("doc_id", ""),
                page=row.get("page"),
                source_file=row.get("source_file", ""),
                modality=row.get("modality", "mixed"),
                content=row.get("content", ""),
                rank=row.get("rank", rank),
                entity_names=row.get("entity_names", []),
                relation_path=row.get("relation_path", []),
                metadata={"relations": payload.get("relations", [])},
            )
            for rank, row in enumerate(payload.get("evidence", []), start=1)
            if row.get("content")
        ]

    @staticmethod
    def _neighbor_evidence(
        payload: Any, entity: str, top_k: int
    ) -> list[EvidenceItem]:
        if not isinstance(payload, dict):
            return []
        relations = payload.get("relations", [])
        relation_by_node: dict[str, list[dict[str, Any]]] = {}
        for relation in relations:
            for node_id in (
                str(relation.get("source", "")),
                str(relation.get("target", "")),
            ):
                if node_id:
                    relation_by_node.setdefault(node_id, []).append(
                        relation
                    )
        rows = []
        for rank, node in enumerate(
            payload.get("nodes", [])[:top_k], start=1
        ):
            node_id = str(node.get("id", ""))
            description = str(node.get("description", "")).strip()
            linked = relation_by_node.get(node_id, [])
            relation_text = "\n".join(
                (
                    f"{item.get('source', '')} -> "
                    f"{item.get('target', '')}: "
                    f"{item.get('description', item.get('keywords', ''))}"
                )
                for item in linked[:4]
            )
            content = "\n".join(
                item
                for item in (
                    f"Graph entity: {node_id}",
                    description,
                    relation_text,
                )
                if item
            )
            if not content:
                continue
            rows.append(
                EvidenceItem(
                    id=f"graph_neighbor_{rank}_{node_id}",
                    source="lightrag_graph",
                    doc_id="",
                    page=None,
                    source_file=str(payload.get("graph_file", "")),
                    modality="mixed",
                    content=content,
                    rank=rank,
                    entity_names=[entity, node_id],
                    relation_path=(
                        [
                            str(linked[0].get("source", "")),
                            str(
                                linked[0].get(
                                    "description",
                                    linked[0].get("keywords", ""),
                                )
                            ),
                            str(linked[0].get("target", "")),
                        ]
                        if linked
                        else []
                    ),
                    metadata={"relations": linked},
                )
            )
        return rows

    @staticmethod
    def _source_detail_evidence(payload: Any, top_k: int) -> list[EvidenceItem]:
        if not isinstance(payload, dict):
            return []
        rows: list[EvidenceItem] = []
        for rank, row in enumerate(payload.get("chunks", [])[:top_k], start=1):
            rows.append(
                EvidenceItem(
                    id=row.get("chunk_id") or f"source_chunk_{rank}",
                    source="lightrag_graph",
                    doc_id=row.get("doc_id", ""),
                    page=row.get("page") or row.get("pdf_page"),
                    source_file=row.get("source_file", ""),
                    modality=row.get("modality", "text"),
                    content=row.get("content", ""),
                    rank=rank,
                    score=0.85,
                    entity_names=[
                        name
                        for name in (
                            row.get("scientific_name", ""),
                            row.get("common_name", ""),
                            row.get("fb_code", ""),
                        )
                        if name
                    ],
                    metadata=row,
                )
            )
        offset = len(rows)
        for index, relation in enumerate(
            payload.get("relations", [])[:top_k], start=1
        ):
            content = (
                f"{relation.get('source', '')} "
                f"--{relation.get('relation', '')}--> "
                f"{relation.get('target', '')}"
            )
            evidence = relation.get("evidence", "")
            if evidence:
                content += f"\nEvidence: {evidence}"
            rows.append(
                EvidenceItem(
                    id=relation.get("triple_id")
                    or f"source_relation_{index}",
                    source="lightrag_graph",
                    doc_id=relation.get("doc_id", ""),
                    page=relation.get("page"),
                    source_file="Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf",
                    modality="graph_relation",
                    content=content,
                    rank=offset + index,
                    score=0.80,
                    entity_names=[
                        str(relation.get("source", "")),
                        str(relation.get("target", "")),
                    ],
                    relation_path=[
                        str(relation.get("source", "")),
                        str(relation.get("relation", "")),
                        str(relation.get("target", "")),
                    ],
                    metadata=relation,
                )
            )
        return [row for row in rows if row.content][:top_k]

    def search_pdf(
        self,
        query: str,
        top_k: int = 12,
        species_ids: list[str] | None = None,
        graph_entities: list[str] | None = None,
        use_mcp: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        warnings = []
        tool_calls: list[dict[str, Any]] = []
        local_rows = self.local.pdf_search(
            query, top_k=top_k, species_ids=species_ids
        )
        sources = {
            "chroma": self._local_evidence(local_rows),
            "bm25": self.bm25.search(query, top_k=top_k),
        }
        graph_enabled = os.getenv("MRAG_USE_GRAPH_MCP", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if graph_enabled and use_mcp:
            sources["graph"] = []
            for entity in graph_entities or []:
                started = time.perf_counter()
                try:
                    payload = self.mcp.call_tool_sync(
                        "raganything",
                        "raganything_graph_neighbors",
                        {"entity": entity, "depth": 1},
                    )
                    rows = self._neighbor_evidence(payload, entity, top_k)
                    sources["graph"].extend(rows)
                    tool_calls.append(
                        {
                            "tool_name": "raganything_graph_neighbors",
                            "tool_source": "mcp",
                            "server": "raganything",
                            "status": "success",
                            "latency_ms": int(
                                (time.perf_counter() - started) * 1000
                            ),
                            "result_count": len(rows),
                            "input": {"entity": entity, "depth": 1},
                        }
                    )
                except Exception as error:
                    warnings.append(
                        "RAG-Anything graph_neighbors MCP unavailable: "
                        f"{type(error).__name__}: {error}"
                    )
                    tool_calls.append(
                        {
                            "tool_name": "raganything_graph_neighbors",
                            "tool_source": "mcp",
                            "server": "raganything",
                            "status": "failed",
                            "latency_ms": int(
                                (time.perf_counter() - started) * 1000
                            ),
                            "result_count": 0,
                            "input": {"entity": entity, "depth": 1},
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                started = time.perf_counter()
                try:
                    payload = self.mcp.call_tool_sync(
                        "raganything",
                        "get_source_detail",
                        {"entity": entity, "top_k": top_k},
                    )
                    rows = self._source_detail_evidence(payload, top_k)
                    sources["graph"].extend(rows)
                    tool_calls.append(
                        {
                            "tool_name": "get_source_detail",
                            "tool_source": "mcp",
                            "server": "raganything",
                            "status": "success",
                            "latency_ms": int(
                                (time.perf_counter() - started) * 1000
                            ),
                            "result_count": len(rows),
                            "input": {"entity": entity, "top_k": top_k},
                        }
                    )
                except Exception as error:
                    warnings.append(
                        "RAG-Anything source_detail MCP unavailable: "
                        f"{type(error).__name__}: {error}"
                    )
                    tool_calls.append(
                        {
                            "tool_name": "get_source_detail",
                            "tool_source": "mcp",
                            "server": "raganything",
                            "status": "failed",
                            "latency_ms": int(
                                (time.perf_counter() - started) * 1000
                            ),
                            "result_count": 0,
                            "input": {"entity": entity, "top_k": top_k},
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            started = time.perf_counter()
            try:
                payload = self.mcp.call_tool_sync(
                    "raganything",
                    "raganything_hybrid_search",
                    {"query": query, "top_k": top_k},
                )
                rows = self._graph_evidence(payload)
                sources["graph"].extend(rows)
                tool_calls.append(
                    {
                        "tool_name": "raganything_hybrid_search",
                        "tool_source": "mcp",
                        "server": "raganything",
                        "status": "success",
                        "latency_ms": int(
                            (time.perf_counter() - started) * 1000
                        ),
                        "result_count": len(rows),
                        "input": {"query": query, "top_k": top_k},
                    }
                )
            except Exception as error:
                warnings.append(
                    "RAG-Anything hybrid_search MCP unavailable; used "
                    f"Chroma/BM25/source detail: {type(error).__name__}: {error}"
                )
                tool_calls.append(
                    {
                        "tool_name": "raganything_hybrid_search",
                        "tool_source": "mcp",
                        "server": "raganything",
                        "status": "failed",
                        "latency_ms": int(
                            (time.perf_counter() - started) * 1000
                        ),
                        "result_count": 0,
                        "input": {"query": query, "top_k": top_k},
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        fused = weighted_rrf(
            sources,
            {"graph": 0.50, "chroma": 0.35, "bm25": 0.15},
            top_k=top_k,
        )
        legacy = [
            {
                "id": row.id,
                "content": row.content,
                "semantic_similarity": row.score,
                "final_score": row.score,
                "metadata": {
                    **row.metadata,
                    "doc_id": row.doc_id,
                    "page": row.page,
                    "doc_title": row.source_file,
                    "source_type": (
                        "lightrag_graph"
                        if row.source == "lightrag_graph"
                        else "pdf_chunk"
                    ),
                    "modality": row.modality,
                    "entity_names": row.entity_names,
                    "relation_path": row.relation_path,
                    "retrieval_source": row.source,
                },
            }
            for row in fused
        ]
        return legacy, {
            "warnings": warnings,
            "counts": {name: len(rows) for name, rows in sources.items()},
            "tool_calls": tool_calls,
        }

    def search_pdf_images(
        self,
        query: str,
        top_k: int = 6,
        entity: str = "",
        use_mcp: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        warnings = []
        tool_calls: list[dict[str, Any]] = []
        payload: Any = None
        if use_mcp:
            started = time.perf_counter()
            try:
                payload = self.mcp.call_tool_sync(
                    "raganything",
                    "search_pdf_images",
                    {
                        "query": query,
                        "top_k": top_k,
                        "entity": entity,
                    },
                )
                if not isinstance(payload, dict):
                    raise RuntimeError(str(payload))
                tool_calls.append(
                    {
                        "tool_name": "search_pdf_images",
                        "tool_source": "mcp",
                        "server": "raganything",
                        "status": "success",
                        "latency_ms": int(
                            (time.perf_counter() - started) * 1000
                        ),
                        "result_count": len(payload.get("results", [])),
                        "input": {
                            "query": query,
                            "top_k": top_k,
                            "entity": entity,
                        },
                    }
                )
            except Exception as error:
                payload = None
                warnings.append(
                    "PDF image MCP unavailable; falling back to local "
                    f"registry/vector search: {type(error).__name__}: {error}"
                )
                tool_calls.append(
                    {
                        "tool_name": "search_pdf_images",
                        "tool_source": "mcp",
                        "server": "raganything",
                        "status": "failed",
                        "latency_ms": int(
                            (time.perf_counter() - started) * 1000
                        ),
                        "result_count": 0,
                        "input": {
                            "query": query,
                            "top_k": top_k,
                            "entity": entity,
                        },
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        try:
            if payload is None:
                from aquabio_raganything.config import (
                    RAGAnythingPaths,
                    RAGAnythingSettings,
                )
                from aquabio_raganything.image_rag import query_pdf_images

                local_payload = query_pdf_images(
                    RAGAnythingPaths.from_root(self.root),
                    RAGAnythingSettings.from_env(),
                    query,
                    top_k=top_k,
                    entity=entity,
                )
                if isinstance(local_payload, dict):
                    warnings.extend(local_payload.get("warnings", []))
                    payload = local_payload
        except Exception as error:
            warnings.append(
                f"Local PDF image retrieval unavailable: "
                f"{type(error).__name__}: {error}"
            )
            if not use_mcp:
                payload = {"results": [], "warnings": warnings}
        try:
            if not isinstance(payload, dict):
                raise RuntimeError(str(payload))
            raw_rows = payload.get("results", [])
            warnings.extend(
                payload.get("warnings", [])
            )
        except Exception as error:
            raw_rows = []
            warnings.append(
                f"PDF image retrieval unavailable: {type(error).__name__}: "
                f"{error}"
            )
        rows = []
        for row in raw_rows:
            rows.append(
                {
                    "id": row.get("id") or row.get("image_id", ""),
                    "content": row.get("caption")
                    or row.get("content", ""),
                    "semantic_similarity": row.get(
                        "semantic_similarity", 0.0
                    ),
                    "final_score": row.get("final_score", 0.0),
                    "metadata": {
                        **row.get("metadata", {}),
                        "source_type": "pdf_image_caption",
                        "modality": "image",
                        "image_id": row.get("image_id", ""),
                        "image_path": row.get("image_path", ""),
                        "absolute_image_path": row.get(
                            "absolute_image_path", ""
                        ),
                        "image_exists": row.get("image_exists", False),
                        "image_role": row.get("image_role", ""),
                        "entity_id": row.get("entity_id", ""),
                        "entity_names": row.get("entity_names", []),
                        "scientific_name": row.get(
                            "scientific_name", ""
                        ),
                        "common_name": row.get("common_name", ""),
                        "fb_code": row.get("fb_code", ""),
                        "class": row.get("class", ""),
                        "order": row.get("order", ""),
                        "family": row.get("family", ""),
                        "phylum": row.get("phylum", ""),
                        "doc_id": row.get("doc_id", ""),
                        "doc_title": row.get("source_file", ""),
                        "source_file": row.get("source_file", ""),
                        "book_id": row.get("metadata", {}).get(
                            "book_id", ""
                        ),
                        "page": row.get("page"),
                        "printed_page": row.get("printed_page"),
                        "retrieval_source": "raganything_pdf_image",
                    },
                }
            )
        return rows, {
            "warnings": warnings,
            "counts": {"raganything_pdf_images": len(rows)},
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _mcp_tool_call_record(
        tool_name: str,
        server: str,
        arguments: dict[str, Any],
        started: float,
        result_count: int,
        status: str = "success",
        error: str = "",
    ) -> dict[str, Any]:
        record = {
            "tool_name": tool_name,
            "tool_source": "mcp",
            "server": server,
            "status": status,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "result_count": result_count,
            "input": arguments,
        }
        if error:
            record["error"] = error
        return record

    # ── MCP-based retrieval (Chroma MCP server via stdio) ──────────────

    def _call_chroma_mcp(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Call a Chroma MCP tool and return rows normalized to local format."""
        try:
            result = self.mcp.call_tool_sync(
                "chroma", tool_name, arguments
            )
            if isinstance(result, str):
                result = json.loads(result)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("results", [result])
            return []
        except Exception as error:
            raise RuntimeError(
                f"Chroma MCP tool {tool_name} failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _normalize_chroma_rows(
        self,
        rows: list[dict[str, Any]],
        retrieval_source: str = "mcp_chroma",
    ) -> list[dict[str, Any]]:
        """Ensure Chroma MCP rows match the format expected by workflow."""
        normalized = []
        for row in rows:
            metadata = row.get("metadata", {})
            normalized.append(
                {
                    "id": row.get("id", ""),
                    "content": row.get("content", ""),
                    "semantic_similarity": row.get(
                        "semantic_similarity", row.get("final_score", 0.0)
                    ),
                    "final_score": row.get("final_score", 0.0),
                    "metadata": {
                        **metadata,
                        "retrieval_source": retrieval_source,
                        "source_type": metadata.get(
                            "source_type", row.get("source_type", "")
                        ),
                        "species_id": metadata.get("species_id", ""),
                        "chunk_type": metadata.get("chunk_type", ""),
                        "modality": metadata.get("modality", "text"),
                        "doc_title": metadata.get(
                            "doc_title", metadata.get("source_url", "")
                        ),
                        "page": metadata.get("page"),
                    },
                }
            )
        return normalized

    def search_text_mcp(
        self,
        query: str,
        top_k: int = 12,
        species_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search species text via Chroma MCP tool."""
        arguments = {"query": query, "top_k": top_k}
        started = time.perf_counter()
        rows = self._call_chroma_mcp("search_species_text", arguments)
        normalized = self._normalize_chroma_rows(rows, "mcp_chroma_text")
        if species_ids:
            species_set = set(species_ids)
            normalized = [
                row
                for row in normalized
                if row["metadata"].get("species_id", "") in species_set
            ]
        normalized = normalized[:top_k]
        return normalized, {
            "warnings": [],
            "counts": {"mcp_chroma_text": len(normalized)},
            "tool_calls": [
                self._mcp_tool_call_record(
                    "search_species_text",
                    "chroma",
                    arguments,
                    started,
                    len(normalized),
                )
            ],
        }

    def search_image_mcp(
        self,
        query: str,
        top_k: int = 12,
        species_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search image captions via Chroma MCP tool."""
        arguments = {"query": query, "top_k": top_k}
        started = time.perf_counter()
        rows = self._call_chroma_mcp("search_image_captions", arguments)
        normalized = self._normalize_chroma_rows(rows, "mcp_chroma_image")
        if species_ids:
            species_set = set(species_ids)
            normalized = [
                row
                for row in normalized
                if row["metadata"].get("species_id", "") in species_set
            ]
        normalized = normalized[:top_k]
        return normalized, {
            "warnings": [],
            "counts": {"mcp_chroma_image": len(normalized)},
            "tool_calls": [
                self._mcp_tool_call_record(
                    "search_image_captions",
                    "chroma",
                    arguments,
                    started,
                    len(normalized),
                )
            ],
        }

    def search_multimodal_mcp(
        self,
        query: str,
        image_caption: str,
        top_k: int = 12,
        species_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search multimodal pairs via Chroma MCP tool."""
        candidate_species = ",".join(species_ids) if species_ids else ""
        arguments = {
            "query": query,
            "image_caption": image_caption,
            "candidate_species": candidate_species,
            "top_k": top_k,
        }
        started = time.perf_counter()
        rows = self._call_chroma_mcp("search_multimodal", arguments)
        normalized = self._normalize_chroma_rows(
            rows, "mcp_chroma_multimodal"
        )[:top_k]
        return normalized, {
            "warnings": [],
            "counts": {"mcp_chroma_multimodal": len(normalized)},
            "tool_calls": [
                self._mcp_tool_call_record(
                    "search_multimodal",
                    "chroma",
                    arguments,
                    started,
                    len(normalized),
                )
            ],
        }

    def search_pdf_chunks_mcp(
        self,
        query: str,
        top_k: int = 12,
        species_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search PDF chunks via Chroma MCP tool."""
        candidate_species = ",".join(species_ids) if species_ids else ""
        arguments = {
            "query": query,
            "candidate_species": candidate_species,
            "top_k": top_k,
        }
        started = time.perf_counter()
        rows = self._call_chroma_mcp("search_pdf", arguments)
        normalized = self._normalize_chroma_rows(
            rows, "mcp_chroma_pdf"
        )[:top_k]
        return normalized, {
            "warnings": [],
            "counts": {"mcp_chroma_pdf": len(normalized)},
            "tool_calls": [
                self._mcp_tool_call_record(
                    "search_pdf",
                    "chroma",
                    arguments,
                    started,
                    len(normalized),
                )
            ],
        }
