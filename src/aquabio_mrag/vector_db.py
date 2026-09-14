from __future__ import annotations

import json
import os
import uuid
from filelock import FileLock
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import chromadb

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
        self._onnx = None
        if os.getenv("MRAG_EMBEDDING_BACKEND", "sentence_transformers") == "onnx":
            if model_name.removeprefix("sentence-transformers/") != "all-MiniLM-L6-v2":
                raise ValueError("ONNX backend supports only all-MiniLM-L6-v2")
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            self._onnx = ONNXMiniLM_L6_V2()
            return
        from sentence_transformers import SentenceTransformer
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
        if self._onnx is not None:
            return [row.tolist() for row in self._onnx(texts)]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

    def encode_query(self, query: str) -> list[float]:
        if self._onnx is not None:
            return self._onnx([query])[0].tolist()
        return self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


class ChromaMRAGStore:
    def __init__(self, paths: MRAGPaths, settings: MRAGSettings):
        self.paths = paths
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(paths.vector_dir))
        self.embedder: BGEEmbedder | None = None
        self._embedding_attempted = False
        self.embedding_error: str | None = None

    def _embedder(self) -> BGEEmbedder | None:
        if self.embedder is None and not getattr(self, "_embedding_attempted", False):
            self._embedding_attempted = True
            try:
                self.embedder = BGEEmbedder(
                    self.settings.embedding_model,
                    cache_folder=self.settings.model_cache,
                    local_files_only=self.settings.local_files_only,
                )
            except Exception as error:
                self.embedding_error = str(error)
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

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        embedder = self._embedder()
        if embedder is None:
            raise RuntimeError("嵌入模型加载失败，原索引未修改。")
        embeddings = embedder.encode_documents(
            [item["embedding_text"] for item in documents], batch_size=batch_size
        )
        if len(embeddings) != len(documents):
            raise ValueError("Embedding count does not match documents")
        # Readers use the manifest as an atomic pointer. Old collections remain
        # available to in-flight readers and for rollback.
        name = f"{self.settings.collection_name}_{uuid.uuid4().hex}"
        with FileLock(str(self.manifest_path) + ".lock"):
            previous = self._active_collection_name()
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine",
                          "embedding_model": self.settings.embedding_model},
            )
            temporary = self.manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                for start in range(0, len(documents), batch_size):
                    batch = documents[start:start + batch_size]
                    collection.add(
                        ids=[item["id"] for item in batch],
                        documents=[item["content"] for item in batch],
                        embeddings=embeddings[start:start + batch_size],
                        metadatas=[scalar_metadata({
                            **item.get("metadata", {}),
                            "source_type": item["source_type"],
                            "species_id": item.get("species_id", ""),
                            "modality": item["modality"],
                        }) for item in batch],
                    )
                if collection.count() != len(documents):
                    raise RuntimeError("New index failed document count validation")
                counts: dict[str, int] = {}
                for item in documents:
                    key = item["source_type"]
                    counts[key] = counts.get(key, 0) + 1
                manifest = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "collection_name": name,
                    "logical_collection_name": self.settings.collection_name,
                    "previous_collection_name": previous,
                    "embedding_model": self.settings.embedding_model,
                    "document_file": str(source.relative_to(self.paths.root)) if source.is_relative_to(self.paths.root) else str(source),
                    "document_count": len(documents),
                    "collection_count": collection.count(),
                    "source_counts": counts,
                }
                temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
                temporary.replace(self.manifest_path)
            except BaseException:
                self.client.delete_collection(name)
                raise
            finally:
                temporary.unlink(missing_ok=True)
        return manifest

    def repair_metadata(self) -> dict:
        """Recover legacy metadata without changing documents or embeddings."""
        source = self.paths.knowledge_dir / "rag_documents_combined.jsonl"
        documents = read_jsonl(source)
        by_id = {row["id"]: row for row in documents}
        if not documents or len(by_id) != len(documents):
            raise ValueError("Source documents must have unique IDs")
        with FileLock(str(self.manifest_path) + ".lock"):
            collection = self.collection()
            stored = collection.get(include=["documents", "metadatas"])
            if set(stored["ids"]) != set(by_id):
                raise ValueError("Index IDs do not match source; rebuild the index instead")
            if any(text != by_id[key]["content"] for key, text in zip(stored["ids"], stored["documents"])):
                raise ValueError("Index content differs from source; rebuild the index instead")
            for start in range(0, len(stored["ids"]), 100):
                ids = stored["ids"][start:start + 100]
                metadatas = []
                for offset, key in enumerate(ids, start=start):
                    row = by_id[key]
                    metadatas.append(scalar_metadata({
                        **(stored["metadatas"][offset] or {}),
                        **row.get("metadata", {}),
                        "species_id": row.get("species_id", ""),
                        "source_type": row["source_type"], "modality": row["modality"],
                    }))
                collection.update(ids=ids, metadatas=metadatas)
            counts: dict[str, int] = {}
            for row in documents:
                counts[row["source_type"]] = counts.get(row["source_type"], 0) + 1
            manifest = {
                **self._manifest(),
                "collection_name": collection.name,
                "logical_collection_name": self.settings.collection_name,
                "document_file": str(source.relative_to(self.paths.root)) if source.is_relative_to(self.paths.root) else str(source), "document_count": len(documents),
                "collection_count": collection.count(), "source_counts": counts,
                "metadata_repaired_at": datetime.now(timezone.utc).isoformat(),
            }
            temporary = self.manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.manifest_path)
            finally:
                temporary.unlink(missing_ok=True)
        return manifest

    @property
    def manifest_path(self) -> Path:
        name = self.settings.collection_name
        filename = "mrag_manifest.json" if name == "aquabio_mrag" else f"{name}_manifest.json"
        return self.paths.vector_dir / filename

    def _manifest(self) -> dict:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _active_collection_name(self) -> str:
        return self._manifest().get("collection_name", self.settings.collection_name)

    def collection(self):
        return self.client.get_collection(self._active_collection_name())

    def query(
        self,
        query: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[dict]:
        collection = self.collection()
        configured_model = self.settings.embedding_model.removeprefix("sentence-transformers/")
        indexed_model = (collection.metadata or {}).get("embedding_model", "").removeprefix("sentence-transformers/")
        if indexed_model and indexed_model != configured_model:
            raise ValueError("查询模型与索引嵌入模型不一致，请恢复原模型或重建索引")
        embedder = self._embedder()
        options = dict(n_results=top_k, where=where, include=["documents", "metadatas", "distances"])
        if embedder is not None:
            result = collection.query(query_embeddings=[embedder.encode_query(query)], **options)
        elif configured_model == "all-MiniLM-L6-v2" and indexed_model in {"", "all-MiniLM-L6-v2"}:
            # Legacy collections used Chroma's default MiniLM ONNX embedder.
            result = collection.query(query_texts=[query], **options)
        else:
            raise RuntimeError("嵌入模型不可用；已阻止使用不同模型查询现有索引")
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
        manifest = self._manifest()
        try:
            collection = self.collection()
            count = collection.count()
            model = (collection.metadata or {}).get("embedding_model", "")
        except chromadb.errors.NotFoundError:
            count, model = 0, ""
        source = self.paths.knowledge_dir / "rag_documents_combined.jsonl"
        return {
            **manifest,
            "path": str(self.paths.vector_dir),
            "collection": self._active_collection_name(),
            "count": count,
            "collection_count": count,
            "document_count": manifest.get("document_count", len(read_jsonl(source)) if source.is_file() else 0),
            "embedding_model": manifest.get("embedding_model", model),
            "manifest_present": bool(manifest),
        }
