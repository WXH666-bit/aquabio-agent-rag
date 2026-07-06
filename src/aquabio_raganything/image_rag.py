from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz

from aquabio_mrag.io_utils import scalar_metadata

from .config import BOOKS, RAGAnythingPaths, RAGAnythingSettings


IMAGE_COLLECTION = "aquabio_pdf_images"
IMAGE_QUERY_MARKERS = (
    "图片",
    "照片",
    "图像",
    "样例图",
    "参考图",
    "长什么样",
    "image",
    "photo",
    "picture",
    "figure",
)
QUERY_EXPANSIONS = {
    "海星": "starfish sea star Asteroidea",
    "海胆": "sea urchin Echinoidea",
    "海参": "sea cucumber Holothuroidea",
    "龙虾": "lobster rock lobster Decapoda Palinuridae",
    "螃蟹": "crab Brachyura Decapoda",
    "水母": "jellyfish Medusozoa",
    "珊瑚": "coral Anthozoa",
    "海绵": "sponge Porifera",
    "章鱼": "octopus Octopoda",
    "鱿鱼": "squid Cephalopoda",
    "贝类": "shellfish mollusc Mollusca",
    "虾": "shrimp prawn Decapoda",
}
CATEGORY_ALIASES = {
    "海星": "Asteroidea",
    "海胆": "Echinoidea",
    "海参": "Holothuroidea",
    "龙虾": "Palinuridae",
    "螃蟹": "Brachyura",
    "水母": "Medusozoa",
    "珊瑚": "Anthozoa",
    "海绵": "Porifera",
    "章鱼": "Octopoda",
    "鱿鱼": "Cephalopoda",
    "贝类": "Mollusca",
    "虾": "Decapoda",
}


SPECIMEN_ROLE = "specimen"
DISTRIBUTION_ROLE = "distribution_map"

IMAGE_QUERY_MARKERS = (
    *IMAGE_QUERY_MARKERS,
    "图片",
    "照片",
    "图像",
    "样例图",
    "参考图",
    "长什么样",
)
QUERY_EXPANSIONS.update(
    {
        "海星": "starfish sea star Asteroidea",
        "海胆": "sea urchin Echinoidea",
        "海参": "sea cucumber Holothuroidea",
        "龙虾": "lobster rock lobster Decapoda Palinuridae",
        "螃蟹": "crab Brachyura Decapoda",
        "水母": "jellyfish Medusozoa",
        "珊瑚": "coral Anthozoa",
        "海绵": "sponge Porifera",
        "章鱼": "octopus Octopoda",
        "鱿鱼": "squid Cephalopoda",
        "贝类": "shellfish mollusc Mollusca",
        "虾": "shrimp prawn Decapoda",
    }
)
CATEGORY_ALIASES.update(
    {
        "海星": "Asteroidea",
        "海胆": "Echinoidea",
        "海参": "Holothuroidea",
        "龙虾": "Palinuridae",
        "螃蟹": "Brachyura",
        "水母": "Medusozoa",
        "珊瑚": "Anthozoa",
        "海绵": "Porifera",
        "章鱼": "Octopoda",
        "鱿鱼": "Cephalopoda",
        "贝类": "Mollusca",
        "虾": "Decapoda",
    }
)


def classify_image_role(image: dict[str, Any]) -> str:
    bbox = image.get("bbox") or []
    if len(bbox) == 4:
        left, top, _, _ = (float(value) for value in bbox)
        if top < 250:
            return DISTRIBUTION_ROLE
        if left < 250:
            return "specimen_overview"
    return "specimen_detail"


def requested_image_roles(query: str) -> set[str]:
    lowered = query.casefold()
    roles: set[str] = set()
    if any(marker in lowered for marker in ("分布图", "分布地图", "范围图")):
        roles.add(DISTRIBUTION_ROLE)
    if any(
        marker in lowered
        for marker in ("实例图", "生物例图", "样例图", "参考图", "照片", "长什么样")
    ):
        roles.add(SPECIMEN_ROLE)
    if any(
        marker in lowered
        for marker in (
            "分布图",
            "分布地图",
            "范围图",
            "distribution map",
            "range map",
        )
    ):
        roles.add(DISTRIBUTION_ROLE)
    if any(
        marker in lowered
        for marker in (
            "实例图",
            "生物例图",
            "样例图",
            "参考图",
            "照片",
            "长什么样",
            "specimen",
            "reference image",
            "photo",
        )
    ):
        roles.add(SPECIMEN_ROLE)
    return roles


def role_matches(image_role: str, requested_role: str) -> bool:
    if requested_role == SPECIMEN_ROLE:
        return image_role.startswith("specimen")
    return image_role == requested_role


def select_requested_roles(
    rows: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    roles = requested_image_roles(query)
    if not roles:
        return rows[:top_k]
    selected = []
    for role in (DISTRIBUTION_ROLE, SPECIMEN_ROLE):
        if role not in roles:
            continue
        for row in rows:
            if not role_matches(str(row.get("image_role", "")), role):
                continue
            if row not in selected:
                selected.append(row)
            if len(selected) >= top_k:
                return selected[:top_k]
    return selected[:top_k]


def requested_page(query: str) -> int | None:
    for pattern in (
        r"(?:PDF\s*)?(?:印刷页|纸质页|页码|页)\s*[:：]?\s*(\d{1,3})",
        r"(?:第\s*)?(\d{1,3})\s*(?:页|面)",
        r"\b(?:printed\s+)?page\s*[:#]?\s*(\d{1,3})\b",
    ):
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized:
        return normalized
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _relative(paths: RAGAnythingPaths, path: Path) -> str:
    try:
        value = path.resolve().relative_to(paths.root)
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def _image_dir(paths: RAGAnythingPaths, book_id: str) -> Path:
    return paths.extracted_assets_dir / book_id / "images"


def _index_dir(paths: RAGAnythingPaths, book_id: str) -> Path:
    return paths.extracted_assets_dir / book_id / "image_index"


def _entity_names(unit: dict[str, Any]) -> list[str]:
    taxonomy = unit.get("taxonomy") or {}
    values = [
        unit.get("scientific_name", ""),
        unit.get("common_name", ""),
        unit.get("title", ""),
        unit.get("fb_code", ""),
        taxonomy.get("genus", ""),
        taxonomy.get("family", ""),
        taxonomy.get("order", ""),
        taxonomy.get("class", ""),
        taxonomy.get("phylum", ""),
    ]
    return list(dict.fromkeys(str(value).strip() for value in values if value))


def _context_caption(
    unit: dict[str, Any],
    image: dict[str, Any],
    image_role: str,
) -> str:
    scientific = unit.get("scientific_name") or unit.get("title") or "taxon"
    common = unit.get("common_name") or ""
    identity = f"{scientific} ({common})" if common else scientific
    role_description = {
        DISTRIBUTION_ROLE: "Distribution map",
        "specimen_overview": "Whole-specimen reference image",
        "specimen_detail": "Specimen detail reference image",
    }.get(image_role, "Reference image")
    parts = [
        f"{role_description} for {identity} from the species identification page.",
        (
            f"PDF page {unit.get('pdf_page')}, printed page "
            f"{unit.get('printed_page') or 'unknown'}."
        ),
    ]
    features = str(unit.get("distinguishing_features") or "").strip()
    colour = str(unit.get("colour") or "").strip()
    size = str(unit.get("size") or "").strip()
    distribution = str(unit.get("distribution") or "").strip()
    if features:
        parts.append(f"Associated diagnostic features: {features}")
    if colour:
        parts.append(f"Associated colour description: {colour}")
    if size:
        parts.append(f"Associated size description: {size}")
    if distribution:
        parts.append(f"Associated distribution: {distribution}")
    parts.append(
        f"Extracted image dimensions: {image.get('width', 0)}x"
        f"{image.get('height', 0)}."
    )
    return " ".join(parts)


def _embedding_text(unit: dict[str, Any], caption: str) -> str:
    taxonomy = unit.get("taxonomy") or {}
    aliases = " ".join(_entity_names(unit))
    ranks = " ".join(
        str(taxonomy.get(rank, ""))
        for rank in ("phylum", "class", "order", "family", "genus")
    )
    return (
        f"{aliases} {ranks} specimen reference image photo figure "
        f"distribution map identification guide {caption}"
    )


def expand_image_query(query: str) -> str:
    expansions = [
        value for key, value in QUERY_EXPANSIONS.items() if key in query
    ]
    return " ".join([query, *expansions]).strip()


def _registry_query_candidates(entity: str, query: str) -> list[str]:
    values = []
    for source in (entity, query):
        lowered = str(source or "").casefold()
        for marker, alias in CATEGORY_ALIASES.items():
            if marker in lowered and alias not in values:
                values.append(alias)
    for value in (entity, query):
        value = str(value or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def build_pdf_image_assets(
    paths: RAGAnythingPaths,
    book_id: str = "sa_invertebrates",
    min_image_dimension: int = 128,
    limit_units: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract embedded PDF images and bind every image to its page taxon."""
    if book_id not in BOOKS:
        raise ValueError(f"Unsupported book: {book_id}")
    paths.ensure()
    native_dir = paths.book_native_dir / book_id
    units_path = native_dir / "species_page_units.jsonl"
    units = _read_jsonl(units_path)
    if not units:
        raise FileNotFoundError(
            f"{units_path} does not exist or is empty; run book-native first."
        )
    if limit_units is not None:
        units = units[: max(0, limit_units)]

    source_path = paths.pdf_dir / BOOKS[book_id]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    image_dir = _image_dir(paths, book_id)
    index_dir = _index_dir(paths, book_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    objects: list[dict[str, Any]] = []
    linked: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    rag_docs: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    canonical_by_sha: dict[str, str] = {}
    failures: list[dict[str, Any]] = []

    document = fitz.open(source_path)
    try:
        for unit in units:
            page_number = int(unit.get("pdf_page") or 0)
            if page_number < 1 or page_number > document.page_count:
                failures.append(
                    {
                        "unit_id": unit.get("unit_id", ""),
                        "reason": f"invalid_pdf_page:{page_number}",
                    }
                )
                continue
            page = document[page_number - 1]
            page_images = {
                int(info.get("xref", 0)): info
                for info in page.get_image_info(xrefs=True)
                if int(info.get("xref", 0))
            }
            records = unit.get("image_records") or []
            if not records:
                records = [
                    {
                        "image_id": (
                            f"{unit['unit_id']}_img_{index:02d}"
                        ),
                        "xref": xref,
                        "width": int(info.get("width", 0)),
                        "height": int(info.get("height", 0)),
                        "bbox": [
                            round(float(value), 2)
                            for value in info.get("bbox", ())
                        ],
                    }
                    for index, (xref, info) in enumerate(
                        page_images.items(), start=1
                    )
                ]

            entity_names = _entity_names(unit)
            canonical_name = (
                unit.get("scientific_name")
                or unit.get("title")
                or unit.get("common_name")
                or unit["unit_id"]
            )
            entity_id = f"taxon:{_slug(str(canonical_name))}"
            taxonomy = unit.get("taxonomy") or {}
            for image in records:
                width = int(image.get("width") or 0)
                height = int(image.get("height") or 0)
                if min(width, height) < min_image_dimension:
                    continue
                xref = int(image.get("xref") or 0)
                image_id = str(image.get("image_id") or "")
                if not xref or not image_id:
                    continue
                try:
                    extracted = document.extract_image(xref)
                    binary = extracted["image"]
                    extension = str(extracted.get("ext") or "png").lower()
                    sha256 = hashlib.sha256(binary).hexdigest()
                    target = image_dir / f"{image_id}.{extension}"
                    if overwrite or not target.is_file():
                        target.write_bytes(binary)
                    relative_path = _relative(paths, target)
                    duplicate_of = canonical_by_sha.get(sha256, "")
                    canonical_by_sha.setdefault(sha256, image_id)
                except Exception as error:
                    failures.append(
                        {
                            "unit_id": unit.get("unit_id", ""),
                            "image_id": image_id,
                            "xref": xref,
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue

                image_role = classify_image_role(image)
                caption = _context_caption(unit, image, image_role)
                aliases = "|".join(entity_names)
                base = {
                    "image_id": image_id,
                    "book_id": book_id,
                    "doc_id": unit.get("doc_id", ""),
                    "unit_id": unit.get("unit_id", ""),
                    "entity_id": entity_id,
                    "entity_names": entity_names,
                    "scientific_name": unit.get("scientific_name", ""),
                    "common_name": unit.get("common_name", ""),
                    "fb_code": unit.get("fb_code", ""),
                    "phylum": taxonomy.get("phylum", ""),
                    "class": taxonomy.get("class", ""),
                    "order": taxonomy.get("order", ""),
                    "family": taxonomy.get("family", ""),
                    "genus": taxonomy.get("genus", ""),
                    "pdf_page": page_number,
                    "printed_page": unit.get("printed_page"),
                    "source_file": unit.get("source_file", source_path.name),
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "bbox": image.get("bbox", []),
                    "extension": extension,
                    "sha256": sha256,
                    "duplicate_of": duplicate_of,
                    "image_path": relative_path,
                    "extraction_method": "pymupdf_extract_image",
                    "image_role": image_role,
                }
                objects.append(base)
                linked_row = {
                    **base,
                    "image_role": image_role,
                    "link_method": "same_species_page",
                    "link_confidence": 1.0,
                    "index_for_entity_search": True,
                }
                linked.append(linked_row)
                caption_row = {
                    **linked_row,
                    "caption": caption,
                    "caption_method": "entity_page_context",
                    "visual_keywords": entity_names,
                    "caption_status": "context_generated",
                }
                captions.append(caption_row)
                rag_docs.append(
                    {
                        "id": f"imgdoc_{image_id}",
                        "source_type": "pdf_image_caption",
                        "species_id": "",
                        "modality": "image_caption",
                        "content": caption,
                        "embedding_text": _embedding_text(unit, caption),
                        "metadata": {
                            "book_id": book_id,
                            "doc_id": unit.get("doc_id", ""),
                            "unit_id": unit.get("unit_id", ""),
                            "entity_id": entity_id,
                            "entity_names": aliases,
                            "scientific_name": unit.get(
                                "scientific_name", ""
                            ),
                            "common_name": unit.get("common_name", ""),
                            "image_id": image_id,
                            "image_path": relative_path,
                            "page": page_number,
                            "printed_page": unit.get("printed_page") or 0,
                            "source_file": unit.get(
                                "source_file", source_path.name
                            ),
                            "image_role": image_role,
                            "phylum": taxonomy.get("phylum", ""),
                            "class": taxonomy.get("class", ""),
                            "order": taxonomy.get("order", ""),
                            "family": taxonomy.get("family", ""),
                        },
                    }
                )
                relations.extend(
                    [
                        {
                            "source": entity_id,
                            "relation": (
                                "has_distribution_map"
                                if image_role == DISTRIBUTION_ROLE
                                else "depicted_by"
                            ),
                            "target": f"image:{image_id}",
                            "page": page_number,
                            "doc_id": unit.get("doc_id", ""),
                        },
                        {
                            "source": f"image:{image_id}",
                            "relation": (
                                "maps_distribution_of"
                                if image_role == DISTRIBUTION_ROLE
                                else "depicts"
                            ),
                            "target": entity_id,
                            "page": page_number,
                            "doc_id": unit.get("doc_id", ""),
                        },
                        {
                            "source": f"image:{image_id}",
                            "relation": "located_on_page",
                            "target": f"page:{page_number}",
                            "page": page_number,
                            "doc_id": unit.get("doc_id", ""),
                        },
                    ]
                )
    finally:
        document.close()

    _write_jsonl(index_dir / "image_objects.jsonl", objects)
    _write_jsonl(index_dir / "linked_pdf_images.jsonl", linked)
    _write_jsonl(index_dir / "pdf_image_captions.jsonl", captions)
    _write_jsonl(index_dir / "pdf_image_rag_docs.jsonl", rag_docs)
    _write_jsonl(index_dir / "image_relation_triples.jsonl", relations)
    _write_jsonl(index_dir / "image_extraction_failures.jsonl", failures)
    report = {
        "book_id": book_id,
        "source_file": source_path.name,
        "units_processed": len(units),
        "images_extracted": len(objects),
        "unique_binaries": len(canonical_by_sha),
        "duplicate_bindings": sum(bool(row["duplicate_of"]) for row in objects),
        "entity_bindings": len(linked),
        "rag_documents": len(rag_docs),
        "relations": len(relations),
        "failures": len(failures),
        "image_dir": str(image_dir),
        "index_dir": str(index_dir),
        "caption_method": "entity_page_context",
    }
    (index_dir / "image_pipeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


class PDFImageVectorStore:
    def __init__(
        self,
        paths: RAGAnythingPaths,
        settings: RAGAnythingSettings,
    ):
        self.paths = paths
        self.settings = settings
        vector_dir = paths.data_dir.parent / "vector_db" / "chroma"
        vector_dir.mkdir(parents=True, exist_ok=True)
        import chromadb

        self.client = chromadb.PersistentClient(path=str(vector_dir))
        self.embedder: Any | None = None

    def _embedder(self) -> Any:
        if self.embedder is None:
            from aquabio_mrag.vector_db import BGEEmbedder

            self.embedder = BGEEmbedder(
                self.settings.embedding_model,
                cache_folder=self.settings.model_cache,
                local_files_only=self.settings.local_files_only,
            )
        return self.embedder

    def build(
        self,
        book_id: str = "sa_invertebrates",
        batch_size: int = 16,
        reset: bool = False,
    ) -> dict[str, Any]:
        source = _index_dir(self.paths, book_id) / "pdf_image_rag_docs.jsonl"
        documents = _read_jsonl(source)
        if not documents:
            raise ValueError(
                f"No PDF image documents found at {source}; "
                "run image-assets first."
            )
        if reset:
            try:
                self.client.delete_collection(IMAGE_COLLECTION)
            except Exception:
                pass
        collection = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.settings.embedding_model,
            },
        )
        existing_ids = set(collection.get(include=[])["ids"])
        pending = [row for row in documents if row["id"] not in existing_ids]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            embeddings = self._embedder().encode_documents(
                [row["embedding_text"] for row in batch],
                batch_size=batch_size,
            )
            collection.add(
                ids=[row["id"] for row in batch],
                documents=[row["content"] for row in batch],
                embeddings=embeddings,
                metadatas=[
                    scalar_metadata(
                        {
                            "source_type": row["source_type"],
                            "modality": row["modality"],
                            **row["metadata"],
                        }
                    )
                    for row in batch
                ],
            )
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_name": IMAGE_COLLECTION,
            "embedding_model": self.settings.embedding_model,
            "book_id": book_id,
            "document_file": str(source),
            "document_count": len(documents),
            "existing_before_run": len(existing_ids),
            "indexed_this_run": len(pending),
            "collection_count": collection.count(),
        }
        manifest_path = _index_dir(
            self.paths, book_id
        ) / "image_vector_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def query(
        self,
        query: str,
        top_k: int = 5,
        entity: str = "",
    ) -> list[dict[str, Any]]:
        collection = self.client.get_collection(IMAGE_COLLECTION)
        expanded_query = expand_image_query(query)
        vector = self._embedder().encode_query(expanded_query)
        result = collection.query(
            query_embeddings=[vector],
            n_results=max(top_k * 4, top_k),
            include=["documents", "metadatas", "distances"],
        )
        query_lower = expanded_query.casefold()
        entity_lower = entity.casefold().strip()
        rows = []
        for row_id, content, metadata, distance in zip(
            result.get("ids", [[]])[0],
            result.get("documents", [[]])[0],
            result.get("metadatas", [[]])[0],
            result.get("distances", [[]])[0],
        ):
            metadata = metadata or {}
            aliases = str(metadata.get("entity_names", "")).casefold()
            exact_entity = bool(
                entity_lower
                and (
                    entity_lower in aliases
                    or entity_lower
                    in str(metadata.get("entity_id", "")).casefold()
                )
            )
            query_entity_match = any(
                len(alias.strip()) >= 3
                and alias.strip().casefold() in query_lower
                for alias in str(
                    metadata.get("entity_names", "")
                ).split("|")
            )
            semantic = max(0.0, 1.0 - float(distance))
            score = min(
                1.0,
                semantic + 0.25 * exact_entity + 0.15 * query_entity_match,
            )
            if entity_lower and not exact_entity:
                continue
            image_path = str(metadata.get("image_path", ""))
            absolute = self.paths.root / image_path
            rows.append(
                {
                    "id": row_id,
                    "image_id": metadata.get("image_id", ""),
                    "entity_id": metadata.get("entity_id", ""),
                    "entity_names": [
                        value
                        for value in str(
                            metadata.get("entity_names", "")
                        ).split("|")
                        if value
                    ],
                    "scientific_name": metadata.get(
                        "scientific_name", ""
                    ),
                    "common_name": metadata.get("common_name", ""),
                    "caption": content or "",
                    "content": content or "",
                    "image_path": image_path,
                    "absolute_image_path": str(absolute.resolve()),
                    "image_exists": absolute.is_file(),
                    "doc_id": metadata.get("doc_id", ""),
                    "source_file": metadata.get("source_file", ""),
                    "page": metadata.get("page"),
                    "printed_page": metadata.get("printed_page"),
                    "modality": "image",
                    "source_type": "pdf_image_caption",
                    "image_role": metadata.get("image_role", ""),
                    "semantic_similarity": round(semantic, 6),
                    "final_score": round(score, 6),
                    "metadata": metadata,
                }
            )
        rows.sort(key=lambda item: item["final_score"], reverse=True)
        page = requested_page(query)
        if page is not None:
            rows = [
                row
                for row in rows
                if page in {row.get("page"), row.get("printed_page")}
            ]
        explicit = [
            row
            for row in rows
            if any(
                len(alias.strip()) >= 4
                and alias.strip().casefold() in query_lower
                for alias in row["entity_names"]
            )
        ]
        if explicit:
            rows = explicit
        return select_requested_roles(rows, query, top_k)

    def info(self) -> dict[str, Any]:
        try:
            count = self.client.get_collection(IMAGE_COLLECTION).count()
        except Exception:
            count = 0
        return {
            "collection": IMAGE_COLLECTION,
            "count": count,
            "path": str(self.paths.data_dir.parent / "vector_db" / "chroma"),
        }


def query_pdf_images(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
    query: str,
    top_k: int = 5,
    entity: str = "",
) -> dict[str, Any]:
    page = requested_page(query)
    registry_rows: list[dict[str, Any]] = []
    for candidate in _registry_query_candidates(entity, query):
        registry_rows = entity_image_records(
            paths,
            candidate,
            top_k=max(top_k * 4, top_k),
            page=page,
        )
        registry_rows = select_requested_roles(registry_rows, query, top_k)
        if registry_rows:
            break
    if registry_rows:
        return {
            "query": query,
            "expanded_query": expand_image_query(query),
            "entity": entity,
            "count": len(registry_rows),
            "results": registry_rows,
            "retrieval_order": "local_entity_registry_first",
            "warnings": [],
        }
    try:
        store = PDFImageVectorStore(paths, settings)
        rows = store.query(query, top_k=top_k, entity=entity)
        if not rows:
            rows = registry_rows
        warnings = []
    except Exception as error:
        rows = registry_rows
        warnings = (
            []
            if isinstance(error, ModuleNotFoundError)
            and error.name == "chromadb"
            else [
                "Image vector collection unavailable; used entity registry: "
                f"{type(error).__name__}: {error}"
            ]
        )
    return {
        "query": query,
        "expanded_query": expand_image_query(query),
        "entity": entity,
        "count": len(rows),
        "results": rows,
        "retrieval_order": "vector_after_local_registry",
        "warnings": warnings,
    }


def entity_image_records(
    paths: RAGAnythingPaths,
    entity: str,
    top_k: int = 5,
    book_id: str = "sa_invertebrates",
    page: int | None = None,
) -> list[dict[str, Any]]:
    source = _index_dir(paths, book_id) / "pdf_image_captions.jsonl"
    rows = _read_jsonl(source)
    needle = entity.casefold().strip()
    if not needle and page is None:
        return []
    exact_matches = []
    broad_matches = []
    prefer_printed_page = bool(
        page is not None
        and any(row.get("printed_page") == page for row in rows)
    )
    for row in rows:
        if page is not None:
            page_value = (
                row.get("printed_page")
                if prefer_printed_page
                else row.get("pdf_page")
            )
            if page_value != page:
                continue
        aliases = [
            row.get("entity_id", ""),
            row.get("scientific_name", ""),
            row.get("common_name", ""),
            row.get("fb_code", ""),
            row.get("genus", ""),
            row.get("family", ""),
            row.get("order", ""),
            row.get("class", ""),
            row.get("phylum", ""),
            *row.get("entity_names", []),
        ]
        normalized_aliases = {
            str(value).casefold().strip() for value in aliases if value
        }
        compact_aliases = {
            re.sub(r"[^a-z0-9]+", "", value)
            for value in normalized_aliases
        }
        compact_needle = re.sub(r"[^a-z0-9]+", "", needle)
        searchable = " ".join(normalized_aliases)
        is_exact = (
            page is not None
            or not needle
            or needle in normalized_aliases
            or (compact_needle and compact_needle in compact_aliases)
        )
        is_broad = (
            page is not None
            or not needle
            or needle in searchable
            or any(
                token in searchable
                for token in re.findall(r"[a-z0-9]{3,}", needle)
            )
        )
        if not is_broad:
            continue
        image_path = str(row.get("image_path", ""))
        absolute = paths.root / image_path
        result = (
            {
                "id": f"imgdoc_{row['image_id']}",
                "image_id": row["image_id"],
                "entity_id": row.get("entity_id", ""),
                "entity_names": row.get("entity_names", []),
                "scientific_name": row.get("scientific_name", ""),
                "common_name": row.get("common_name", ""),
                "fb_code": row.get("fb_code", ""),
                "phylum": row.get("phylum", ""),
                "class": row.get("class", ""),
                "order": row.get("order", ""),
                "family": row.get("family", ""),
                "source_file": row.get("source_file", ""),
                "book_id": row.get("book_id", ""),
                "caption": row.get("caption", ""),
                "content": row.get("caption", ""),
                "image_path": image_path,
                "absolute_image_path": str(absolute.resolve()),
                "image_exists": absolute.is_file(),
                "doc_id": row.get("doc_id", ""),
                "source_file": row.get("source_file", ""),
                "page": row.get("pdf_page"),
                "printed_page": row.get("printed_page"),
                "modality": "image",
                "source_type": "pdf_image_caption",
                "image_role": row.get("image_role", ""),
                "semantic_similarity": 0.0,
                "final_score": 1.0,
                "metadata": row,
            }
        )
        (exact_matches if is_exact else broad_matches).append(result)
    selected = exact_matches or broad_matches
    return selected[:top_k]


def asks_for_reference_images(query: str) -> bool:
    lowered = query.casefold()
    return bool(requested_image_roles(query)) or any(
        marker in lowered for marker in IMAGE_QUERY_MARKERS
    )
