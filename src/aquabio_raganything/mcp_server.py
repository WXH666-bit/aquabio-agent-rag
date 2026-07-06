from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import RAGAnythingPaths, RAGAnythingSettings
from .image_rag import entity_image_records, query_pdf_images
from .query_adapter import (
    graph_neighbors,
    hybrid_search,
    index_status,
    source_detail,
)


ROOT = Path(__file__).resolve().parents[2]
PATHS = RAGAnythingPaths.from_root(ROOT)
SETTINGS = RAGAnythingSettings.from_env()
MCP = FastMCP(
    "AquaBio-RAGAnything",
    host=os.getenv("AQUABIO_RAG_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("AQUABIO_RAG_MCP_PORT", "8765")),
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@MCP.tool()
def raganything_index_status() -> dict:
    """Read the RAG-Anything/LightRAG persistent index status."""
    return index_status(PATHS)


@MCP.tool()
def raganything_graph_neighbors(entity: str, depth: int = 1) -> dict:
    """Return NetworkX graph neighbors for an entity."""
    return graph_neighbors(PATHS, entity, depth)


@MCP.tool()
async def raganything_hybrid_search(
    query: str, top_k: int = 12
) -> dict:
    """Run LightRAG hybrid graph and vector retrieval."""
    return await hybrid_search(PATHS, SETTINGS, query, top_k)


@MCP.tool()
def raganything_image_search(
    query: str, top_k: int = 5, entity: str = ""
) -> dict:
    """Retrieve real PDF images with taxon, page, and local path metadata."""
    return query_pdf_images(PATHS, SETTINGS, query, top_k, entity)


@MCP.tool()
def search_pdf_images(
    query: str, top_k: int = 5, entity: str = ""
) -> dict:
    """Alias used by LangGraph: retrieve PDF images via MCP."""
    return query_pdf_images(PATHS, SETTINGS, query, top_k, entity)


@MCP.tool()
def raganything_entity_images(entity: str, top_k: int = 5) -> dict:
    """Return PDF sample images directly bound to a taxon entity."""
    rows = entity_image_records(PATHS, entity, top_k)
    return {"entity": entity, "count": len(rows), "results": rows}


@MCP.tool()
def raganything_source_detail(doc_id: str) -> dict:
    """Return book-native source details for a document ID."""
    detail = source_detail(PATHS, doc_id=doc_id)
    detail["index_status"] = index_status(PATHS)
    return detail


@MCP.tool()
def get_source_detail(
    doc_id: str = "",
    entity: str = "",
    image_id: str = "",
    top_k: int = 12,
) -> dict:
    """Return PDF page, taxon, image and relation evidence via MCP."""
    detail = source_detail(
        PATHS,
        doc_id=doc_id,
        entity=entity,
        image_id=image_id,
        top_k=top_k,
    )
    detail["index_status"] = index_status(PATHS)
    return detail


def main() -> None:
    transport = os.getenv(
        "AQUABIO_RAG_MCP_TRANSPORT", "stdio"
    ).lower()
    MCP.run(
        transport=(
            "streamable-http"
            if transport in {"http", "streamable-http"}
            else "stdio"
        )
    )


if __name__ == "__main__":
    main()
