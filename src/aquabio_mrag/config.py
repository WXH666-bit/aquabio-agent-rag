from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from aquabio.config import load_env


@dataclass(frozen=True)
class MRAGPaths:
    root: Path
    seed_db: Path
    species_list: Path
    raw_dir: Path
    images_dir: Path
    knowledge_dir: Path
    pdf_dir: Path
    pdf_figures_dir: Path
    vector_dir: Path
    uploads_dir: Path
    sessions_dir: Path
    outputs_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "MRAGPaths":
        project = Path(root).resolve()
        seed = project / "aquabio_mrag_long_text_database"
        data = project / "data" / "mrag"
        return cls(
            root=project,
            seed_db=seed,
            species_list=seed / "data" / "species_list.json",
            raw_dir=data / "raw",
            images_dir=data / "images",
            knowledge_dir=data / "knowledge",
            pdf_dir=data / "pdfs",
            pdf_figures_dir=data / "pdf_figures",
            vector_dir=data / "vector_db" / "chroma",
            uploads_dir=data / "uploads",
            sessions_dir=data / "sessions",
            outputs_dir=project / "data" / "outputs",
        )

    def ensure(self) -> None:
        for path in (
            self.raw_dir,
            self.images_dir,
            self.knowledge_dir,
            self.pdf_dir,
            self.pdf_figures_dir,
            self.vector_dir,
            self.uploads_dir,
            self.sessions_dir,
            self.outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MRAGSettings:
    embedding_model: str = "all-MiniLM-L6-v2"
    model_cache: str = ""
    local_files_only: bool = False
    collection_name: str = "aquabio_mrag"
    top_k: int = 12
    rerank_top_k: int = 6
    max_retry: int = 2

    @classmethod
    def from_env(cls) -> "MRAGSettings":
        load_env()
        return cls(
            embedding_model=os.getenv("MRAG_EMBEDDING_MODEL", cls.embedding_model),
            model_cache=os.getenv("MRAG_MODEL_CACHE", cls.model_cache),
            local_files_only=os.getenv(
                "MRAG_LOCAL_FILES_ONLY", "true"
            ).lower() in {"1", "true", "yes", "on"},
            collection_name=os.getenv("MRAG_COLLECTION_NAME", cls.collection_name),
            top_k=int(os.getenv("MRAG_TOP_K", str(cls.top_k))),
            rerank_top_k=int(
                os.getenv("MRAG_RERANK_TOP_K", str(cls.rerank_top_k))
            ),
            max_retry=int(os.getenv("MRAG_MAX_RETRY", str(cls.max_retry))),
        )
