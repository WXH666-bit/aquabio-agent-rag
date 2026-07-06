from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from aquabio.config import load_env


BOOKS = {
    "living_guide": "FIELD IDENTIFICATION GUIDE TO THE LIVING.pdf",
    "sa_invertebrates": (
        "Field-Guide-to-SA-Offshore-Marine-Invertebrates_"
        "web-full-version_compressed.pdf"
    ),
}

ENTITY_TYPES = [
    "species",
    "taxon",
    "anatomical_feature",
    "habitat",
    "behavior",
    "distribution",
    "conservation_status",
    "image",
    "table",
    "equation",
    "document",
    "section",
]

ENTITY_GUIDANCE = (
    "Prefer canonical English taxon names. Map common names and scientific "
    "names to one species entity. Extract diagnostic morphology, habitat, "
    "behavior, distribution, and cross-modal evidence. Preserve document ID, "
    "source file, and page provenance in descriptions. Useful relations include "
    "is_a, has_feature, lives_in, distributed_in, exhibits_behavior, similar_to, "
    "distinguished_from, illustrated_by, listed_in, described_in, part_of, "
    "belongs_to, and associated_with. Do not invent species identification."
)


@dataclass(frozen=True)
class RAGAnythingPaths:
    root: Path
    data_dir: Path
    pdf_dir: Path
    inventory_dir: Path
    parser_output_dir: Path
    extracted_assets_dir: Path
    segment_pdf_dir: Path
    manifests_dir: Path
    working_dir: Path
    logs_dir: Path
    book_native_dir: Path
    species_list: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "RAGAnythingPaths":
        project = Path(root).resolve()
        data = project / "data" / "mrag" / "raganything"
        return cls(
            root=project,
            data_dir=data,
            pdf_dir=project / "data" / "mrag" / "pdfs",
            inventory_dir=data / "inventory",
            parser_output_dir=data / "parser_output",
            extracted_assets_dir=data / "extracted_assets",
            segment_pdf_dir=data / "segment_pdfs",
            manifests_dir=data / "manifests",
            working_dir=data / "working",
            logs_dir=data / "logs",
            book_native_dir=data / "book_native",
            species_list=(
                project
                / "aquabio_mrag_long_text_database"
                / "data"
                / "species_list.json"
            ),
        )

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.inventory_dir,
            self.parser_output_dir,
            self.extracted_assets_dir,
            self.segment_pdf_dir,
            self.manifests_dir,
            self.working_dir,
            self.logs_dir,
            self.book_native_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RAGAnythingSettings:
    parser: str = "mineru"
    parse_method: str = "auto"
    parser_backend: str = "pipeline"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    model_cache: str = ""
    local_files_only: bool = False
    max_segment_pages: int = 40
    segment_overlap: int = 1
    min_image_size: int = 256
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nex-agi/nex-n2-pro:free"
    text_llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_key_file: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.5-flash"
    gemini_fallback_model: str = "gemini-2.5-flash-lite"

    @classmethod
    def from_env(cls) -> "RAGAnythingSettings":
        load_env()
        deepseek_key_file = os.getenv("DEEPSEEK_KEY_FILE", "")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not deepseek_api_key and deepseek_key_file:
            key_path = Path(deepseek_key_file)
            if key_path.is_file():
                match = re.search(
                    r"(?m)^(sk-[A-Za-z0-9_-]{20,})\s*$",
                    key_path.read_text(encoding="utf-8", errors="ignore"),
                )
                if match:
                    deepseek_api_key = match.group(1)
        return cls(
            parser=os.getenv("RAGANYTHING_PARSER", cls.parser),
            parse_method=os.getenv(
                "RAGANYTHING_PARSE_METHOD", cls.parse_method
            ),
            parser_backend=os.getenv(
                "RAGANYTHING_PARSER_BACKEND", cls.parser_backend
            ),
            embedding_model=os.getenv(
                "MRAG_EMBEDDING_MODEL", cls.embedding_model
            ),
            embedding_dim=int(
                os.getenv("RAGANYTHING_EMBEDDING_DIM", str(cls.embedding_dim))
            ),
            model_cache=os.getenv("MRAG_MODEL_CACHE", ""),
            local_files_only=os.getenv(
                "MRAG_LOCAL_FILES_ONLY", "true"
            ).lower()
            in {"1", "true", "yes", "on"},
            max_segment_pages=int(
                os.getenv(
                    "RAGANYTHING_MAX_SEGMENT_PAGES",
                    str(cls.max_segment_pages),
                )
            ),
            segment_overlap=int(
                os.getenv(
                    "RAGANYTHING_SEGMENT_OVERLAP",
                    str(cls.segment_overlap),
                )
            ),
            min_image_size=int(
                os.getenv(
                    "RAGANYTHING_MIN_IMAGE_SIZE", str(cls.min_image_size)
                )
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", cls.openrouter_base_url
            ),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL", cls.openrouter_model
            ),
            text_llm_provider=os.getenv(
                "RAGANYTHING_TEXT_LLM_PROVIDER",
                cls.text_llm_provider,
            ).lower(),
            deepseek_api_key=deepseek_api_key,
            deepseek_key_file=deepseek_key_file,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", cls.deepseek_base_url
            ),
            deepseek_model=os.getenv(
                "DEEPSEEK_MODEL", cls.deepseek_model
            ),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_base_url=os.getenv(
                "GEMINI_BASE_URL", cls.gemini_base_url
            ),
            gemini_model=os.getenv(
                "GEMINI_VISION_MODEL", cls.gemini_model
            ),
            gemini_fallback_model=os.getenv(
                "GEMINI_VISION_FALLBACK_MODEL",
                cls.gemini_fallback_model,
            ),
        )
