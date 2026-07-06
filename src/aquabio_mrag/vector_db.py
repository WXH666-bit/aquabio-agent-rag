from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from .config import MRAGPaths, MRAGSettings
from .io_utils import read_jsonl, scalar_metadata


class BGEEmbedder:
    def __init__(
        self,
        model_name: str,
        cache_folder: str = "",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        model_source = model_name
        if local_files_only and cache_folder:
            cache_root = Path(cache_folder)
            model_dir = "models--" + model_name.replace("/", "--")
            for root in (cache_root, cache_root / "hub"):
                repository = root / model_dir
                ref = repository / "refs" / "main"
                if not ref.is_file():
                    continue
                snapshot = repository / "snapshots" / ref.read_text(
                    encoding="utf-8"
                ).strip()
                if (snapshot / "config.json").is_file():
                    model_source = str(snapshot)
                    break
        self.model = SentenceTransformer(
            model_source,
            trust_remote_code=True,
            cache_folder=cache_folder or None,
            local_files_only=local_files_only,
        )

    def encode_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


class ChromaMRAGStore:
    def __init__(self, paths: MRAGPaths, settings: MRAGSettings):
        self.paths = paths
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(paths.vector_dir))
        self.embedder: BGEEmbedder | None = None

    def _embedder(self) -> BGEEmbedder | None:
        if self.embedder is None:
            try:
                self.embedder = BGEEmbedder(
                    self.settings.embedding_model,
                    cache_folder=self.settings.model_cache,
                    local_files_only=self.settings.local_files_only,
                )
            except Exception:
                self.embedder = None
        return self.embedder

    def build(
        self,
        document_file: str | Path | None = None,
        batch_size: int = 16,
    ) -> dict:
        source = Path(document_file) if document_file else (
            self.paths.knowledge_dir / "rag_documents_combined.jsonl"
        )
        documents = read_jsonl(source)
        if not documents:
            raise ValueError(f"没有可写入 Chroma 的统一文档：{source}")

        try:
            self.client.delete_collection(self.settings.collection_name)
        except Exception:
            pass
        collection = self.client.create_collection(
            name=self.settings.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.settings.embedding_model,
            },
        )
        embeddings = self._embedder().encode_documents(
            [item["embedding_text"] for item in documents],
            batch_size=batch_size,
        )

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["content"] for item in batch],
                embeddings=embeddings[start : start + batch_size],
                metadatas=[
                    scalar_metadata(
                        {
                            "source_type": item["source_type"],
                            "species_id": item.get("species_id", ""),
                            "modality": item["modality"],
                            **item.get("metadata", {}),
                        }
                    )
                    for item in batch
                ],
            )

        counts: dict[str, int] = {}
        for item in documents:
            source_type = item["source_type"]
            counts[source_type] = counts.get(source_type, 0) + 1
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_name": self.settings.collection_name,
            "embedding_model": self.settings.embedding_model,
            "document_file": str(source),
            "document_count": len(documents),
            "collection_count": collection.count(),
            "source_counts": counts,
        }
        manifest_path = self.paths.vector_dir / "mrag_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def collection(self):
        return self.client.get_collection(self.settings.collection_name)

    def query(
        self,
        query: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[dict]:
        embedder = self._embedder()
        if embedder is not None:
            try:
                vector = embedder.encode_query(query)
                result = self.collection().query(
                    query_embeddings=[vector],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                result = self.collection().query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
        else:
            result = self.collection().query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        rows = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            rows.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata or {},
                    "semantic_similarity": max(0.0, 1.0 - float(distance)),
                }
            )
        return rows

    def info(self) -> dict:
        manifest_path = self.paths.vector_dir / "mrag_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        try:
            count = self.collection().count()
        except Exception:
            count = 0
        return {
            "path": str(self.paths.vector_dir),
            "collection": self.settings.collection_name,
            "count": count,
            **manifest,
        }
