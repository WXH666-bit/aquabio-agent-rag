from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .vector_store import LocalVectorStore, load_source_records, record_text

DOMAIN_TRANSLATIONS = {
    "水下图像": "underwater image",
    "颜色偏移": "color cast color distortion",
    "偏蓝": "blue color cast",
    "偏绿": "green color cast",
    "低对比度": "low contrast",
    "光吸收": "light absorption wavelength attenuation",
    "散射": "scattering",
    "图像增强": "image enhancement",
    "目标检测": "object detection",
    "机器人抓取": "robot picking",
    "海星": "starfish sea star",
    "海胆": "echinus sea urchin",
    "海参": "holothurian sea cucumber",
    "扇贝": "scallop",
    "水母": "jellyfish",
}


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_]{2,}", lowered)
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = ["".join(cjk[index : index + 2]) for index in range(len(cjk) - 1)]
    return set(latin + bigrams)


def _expand_query(query: str) -> str:
    additions = [english for chinese, english in DOMAIN_TRANSLATIONS.items() if chinese in query]
    return " ".join([query, *additions])


class HybridRetriever:
    def __init__(
        self,
        knowledge_dir: str | Path = "data/knowledge",
        index_dir: str | Path = "data/index",
        vector_db_dir: str | Path | None = None,
    ):
        if vector_db_dir is None:
            vector_db_dir = Path(knowledge_dir).parent / "vector_db"
        store = LocalVectorStore(vector_db_dir)
        self.using_persistent_store = store.exists
        if self.using_persistent_store:
            store.load()
            self.records = store.records
            self.texts = store.texts
            self.vectorizer = store.vectorizer
            self.matrix = store.vectors
            return

        self.records = load_source_records(knowledge_dir, index_dir)
        self.texts = [record_text(record) for record in self.records]
        self.vectorizer = None
        self.matrix = None
        if self.texts:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
                sublinear_tf=True,
                norm="l2",
            )
            self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 6, source_types: set[str] | None = None) -> list[dict]:
        if not self.records or self.vectorizer is None or self.matrix is None:
            return []
        expanded_query = _expand_query(query)
        vector_scores = (self.matrix @ self.vectorizer.transform([expanded_query]).T).toarray().ravel()
        query_terms = _terms(expanded_query)
        lexical_scores = []
        for text in self.texts:
            terms = _terms(text)
            lexical_scores.append(len(query_terms & terms) / max(1, len(query_terms)))
        scores = 0.78 * vector_scores + 0.22 * np.asarray(lexical_scores)
        ranked = np.argsort(-scores)
        results = []
        for index in ranked:
            record = self.records[int(index)]
            if source_types and record.get("source_type") not in source_types:
                continue
            if scores[index] <= 0:
                break
            item = dict(record)
            item["score"] = round(float(scores[index]), 4)
            results.append(item)
            if len(results) >= top_k:
                break
        return results
