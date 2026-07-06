from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def record_text(record: dict) -> str:
    """Convert heterogeneous knowledge records into indexable text."""
    fields = [
        str(record.get("content", "")),
        str(record.get("dataset_name", "")),
        str(record.get("class_name", "")),
        str(record.get("chinese_name", "")),
        str(record.get("task", "")),
        " ".join(record.get("keywords", [])),
        " ".join(record.get("visual_features", [])),
    ]
    return " ".join(field for field in fields if field).strip()


def load_source_records(*directories: str | Path) -> list[dict]:
    records: list[dict] = []
    seen_ids: set[str] = set()
    for directory_value in directories:
        directory = Path(directory_value)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(record.get("id", ""))
                if record_id and record_id in seen_ids:
                    continue
                if record_id:
                    seen_ids.add(record_id)
                records.append(record)
    return records


class LocalVectorStore:
    """Persistent sparse-vector store for the small AquaBio knowledge base."""

    RECORDS_FILE = "records.jsonl"
    VECTORIZER_FILE = "vectorizer.joblib"
    VECTORS_FILE = "vectors.npz"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, directory: str | Path = "data/vector_db"):
        self.directory = Path(directory)
        self.records: list[dict] = []
        self.texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.vectors = None

    @property
    def exists(self) -> bool:
        required = (
            self.RECORDS_FILE,
            self.VECTORIZER_FILE,
            self.VECTORS_FILE,
            self.MANIFEST_FILE,
        )
        return all((self.directory / filename).exists() for filename in required)

    def build(
        self,
        knowledge_dir: str | Path = "data/knowledge",
        index_dir: str | Path = "data/index",
    ) -> dict:
        records = load_source_records(knowledge_dir, index_dir)
        if not records:
            raise ValueError("没有找到可写入向量库的 JSONL 知识记录。")

        texts = [record_text(record) for record in records]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        vectors = vectorizer.fit_transform(texts)

        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / self.RECORDS_FILE).open(
            "w", encoding="utf-8"
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        joblib.dump(vectorizer, self.directory / self.VECTORIZER_FILE)
        save_npz(self.directory / self.VECTORS_FILE, vectors)

        source_counts: dict[str, int] = {}
        for record in records:
            source_type = str(record.get("source_type", "unknown"))
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        manifest = {
            "store_type": "local_tfidf_sparse_vector_store",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records),
            "vector_count": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "source_counts": source_counts,
            "knowledge_dir": str(Path(knowledge_dir)),
            "index_dir": str(Path(index_dir)),
        }
        (self.directory / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.records = records
        self.texts = texts
        self.vectorizer = vectorizer
        self.vectors = vectors
        return manifest

    def load(self) -> "LocalVectorStore":
        if not self.exists:
            raise FileNotFoundError(
                f"向量库不完整：{self.directory}。请先执行 build-vector-db。"
            )
        self.records = [
            json.loads(line)
            for line in (self.directory / self.RECORDS_FILE)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.texts = [record_text(record) for record in self.records]
        self.vectorizer = joblib.load(self.directory / self.VECTORIZER_FILE)
        self.vectors = load_npz(self.directory / self.VECTORS_FILE)
        if self.vectors.shape[0] != len(self.records):
            raise ValueError("向量数量与知识记录数量不一致，请重建向量库。")
        return self

    def info(self) -> dict:
        if not self.exists:
            return {"exists": False, "directory": str(self.directory)}
        manifest = json.loads(
            (self.directory / self.MANIFEST_FILE).read_text(encoding="utf-8")
        )
        return {"exists": True, "directory": str(self.directory), **manifest}
