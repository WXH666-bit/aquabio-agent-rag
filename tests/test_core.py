from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aquabio.image_tools import analyze_quality, create_enhancements
from aquabio.retriever import HybridRetriever
from aquabio.vector_store import LocalVectorStore


class RetrieverTests(unittest.TestCase):
    def test_species_retrieval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            record = {
                "id": "starfish",
                "source_type": "species_card",
                "content": "海星具有放射状腕和星形轮廓",
                "keywords": ["starfish", "海星"],
            }
            (knowledge / "species.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            result = HybridRetriever(knowledge, root / "index").search("海星有哪些视觉特征")
            self.assertEqual(result[0]["id"], "starfish")

    def test_persistent_vector_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            knowledge = root / "knowledge"
            index = root / "index"
            vector_db = root / "vector_db"
            knowledge.mkdir()
            index.mkdir()
            records = [
                {
                    "id": "uieb",
                    "source_type": "dataset_card",
                    "content": "underwater image enhancement color cast low contrast",
                    "keywords": ["UIEB"],
                },
                {
                    "id": "starfish",
                    "source_type": "species_card",
                    "content": "海星具有中央盘和放射状腕",
                    "keywords": ["starfish", "海星"],
                },
            ]
            (knowledge / "records.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
                + "\n",
                encoding="utf-8",
            )
            manifest = LocalVectorStore(vector_db).build(knowledge, index)
            self.assertEqual(manifest["record_count"], 2)
            self.assertTrue((vector_db / "vectors.npz").exists())
            retriever = HybridRetriever(knowledge, index, vector_db)
            self.assertTrue(retriever.using_persistent_store)
            self.assertEqual(retriever.search("海星特征")[0]["id"], "starfish")


class ImageTests(unittest.TestCase):
    def test_quality_and_enhancement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "blue.jpg"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, 0] = 180
            image[:, :, 1] = 60
            image[:, :, 2] = 30
            cv2.imwrite(str(image_path), image)
            quality = analyze_quality(image_path)
            self.assertEqual(quality["dominant_channel"], "blue")
            outputs = create_enhancements(image_path, root / "out")
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(Path(item["path"]).exists() for item in outputs))


if __name__ == "__main__":
    unittest.main()
