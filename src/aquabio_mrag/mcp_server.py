from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from aquabio.config import Settings
from aquabio.openrouter import OpenRouterClient
from aquabio_raganything.config import (
    RAGAnythingPaths,
    RAGAnythingSettings,
)
from aquabio_raganything.image_rag import (
    entity_image_records,
    query_pdf_images,
)

from .config import MRAGPaths, MRAGSettings
from .retrieval import MultiSourceRetriever, RetrievalRequest


ROOT = Path(__file__).resolve().parents[2]
PATHS = MRAGPaths.from_root(ROOT)
SETTINGS = MRAGSettings.from_env()
RETRIEVER = MultiSourceRetriever(PATHS, SETTINGS)
RAG_PATHS = RAGAnythingPaths.from_root(ROOT)
RAG_SETTINGS = RAGAnythingSettings.from_env()
VLM = OpenRouterClient(Settings.from_env())
mcp = FastMCP("AquaBio-MRAG Tools")


@mcp.tool()
def search_species_text(query: str, top_k: int = 6) -> str:
    """Search species cards and species text chunks."""
    return json.dumps(
        RETRIEVER.text_search(query, top_k=top_k), ensure_ascii=False
    )


@mcp.tool()
def search_image_captions(query: str, top_k: int = 6) -> str:
    """Search image captions and image-text pairs."""
    return json.dumps(
        RETRIEVER.image_search(query, top_k=top_k), ensure_ascii=False
    )


@mcp.tool()
def search_pdf_entity_images(
    query: str, top_k: int = 5, entity: str = ""
) -> dict:
    """Search real images extracted from PDFs with entity/page metadata."""
    return query_pdf_images(
        RAG_PATHS, RAG_SETTINGS, query, top_k=top_k, entity=entity
    )


@mcp.tool()
def get_entity_sample_images(entity: str, top_k: int = 5) -> dict:
    """Read sample PDF images bound directly to a taxon entity."""
    rows = entity_image_records(RAG_PATHS, entity, top_k=top_k)
    return {"entity": entity, "count": len(rows), "results": rows}


@mcp.tool()
def search_multimodal(
    query: str,
    image_caption: str,
    candidate_species: str = "",
    top_k: int = 8,
) -> str:
    """Search image-text pairs, species knowledge and PDF evidence."""
    species = [item.strip() for item in candidate_species.split(",") if item.strip()]
    return json.dumps(
        RETRIEVER.multimodal_search(
            query,
            image_caption,
            top_k=top_k,
            species_ids=species or None,
        ),
        ensure_ascii=False,
    )


@mcp.tool()
def search_pdf(
    query: str,
    candidate_species: str = "",
    top_k: int = 6,
) -> str:
    """Search registered PDF chunks and return document/page metadata."""
    species = [item.strip() for item in candidate_species.split(",") if item.strip()]
    return json.dumps(
        RETRIEVER.pdf_search(query, top_k=top_k, species_ids=species or None),
        ensure_ascii=False,
    )


@mcp.tool()
def generate_image_caption(image_path: str) -> str:
    """Generate a structured caption for a local underwater image."""
    result = VLM.analyze_image(
        image_path,
        "描述水下生物外观、形状、颜色、环境、候选类别和不确定性。",
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_source_detail(doc_id: str) -> str:
    """Fetch one document and all of its stored metadata by exact id."""
    result = RETRIEVER.store.collection().get(
        ids=[doc_id], include=["documents", "metadatas"]
    )
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
