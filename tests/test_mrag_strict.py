from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.models import ImageDocument, RAGDocument
from aquabio_mrag.vector_db import ChromaMRAGStore
from aquabio_mrag.workflow import AquaBioMRAGWorkflow
from aquabio.openrouter import OpenRouterClient


ROOT = Path(__file__).resolve().parents[1]
PATHS = MRAGPaths.from_root(ROOT)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class StrictMRAGDataTests(unittest.TestCase):
    def test_twenty_species_have_ten_images_each(self) -> None:
        rows = read_jsonl(PATHS.knowledge_dir / "image_docs.jsonl")
        counts = Counter(row["species_id"] for row in rows)
        self.assertEqual(len(rows), 200)
        self.assertEqual(len(counts), 20)
        self.assertEqual(set(counts.values()), {10})
        for row in rows:
            ImageDocument.model_validate(row)
            self.assertTrue((ROOT / row["image_path"]).is_file())
            self.assertTrue(row["license"])
            self.assertTrue(row["source_page"])

    def test_unified_documents_are_strict_and_complete(self) -> None:
        rows = read_jsonl(
            PATHS.knowledge_dir / "rag_documents_combined.jsonl"
        )
        counts = Counter(row["source_type"] for row in rows)
        self.assertEqual(len(rows), 772)
        self.assertEqual(len({row["id"] for row in rows}), 772)
        self.assertEqual(
            counts,
            {
                "species_card": 20,
                "species_text_chunk": 160,
                "image_doc": 200,
                "multimodal_pair": 200,
                "pdf_chunk": 192,
            },
        )
        for row in rows:
            RAGDocument.model_validate(row)
            self.assertTrue(row["embedding_text"].strip())

    def test_chroma_count_matches_manifest(self) -> None:
        store = ChromaMRAGStore(PATHS, MRAGSettings.from_env())
        info = store.info()
        self.assertEqual(info["count"], 772)
        self.assertEqual(info["document_count"], 772)
        self.assertEqual(info["collection_count"], 772)


class StrictMRAGRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = AquaBioMRAGWorkflow(
            PATHS, MRAGSettings.from_env(), offline=True
        )

    def route(self, query: str, image_path: str | None = None) -> str:
        result = self.workflow.router_node(
            {"original_query": query, "image_path": image_path, "trace": []}
        )
        return result["route"]["task_type"]

    def test_all_router_modes(self) -> None:
        self.assertEqual(self.route("海星有什么特征？"), "text_qa")
        self.assertEqual(self.route("", "image.jpg"), "image_qa")
        self.assertEqual(
            self.route("这是什么？", "image.jpg"), "multimodal_qa"
        )
        self.assertEqual(
            self.route("海星和海胆有什么区别？"), "comparison_qa"
        )
        self.assertEqual(
            self.route("这个结论的来源是什么？"), "source_trace"
        )
        self.assertEqual(
            self.route("根据PDF报告说明珊瑚生态。"), "pdf_qa"
        )

    def test_truncated_answer_fails_evaluation(self) -> None:
        previous = self.workflow.offline
        self.workflow.offline = False
        try:
            result = self.workflow.evaluation_node(
                {
                    "draft_answer": "海星通常呈放射状[E",
                    "final_context": "[E1] 海星具有中央盘和腕足。",
                    "route": {"need_vlm": False},
                    "trace": [],
                }
            )["evaluation_result"]
        finally:
            self.workflow.offline = previous
        self.assertFalse(result["passed"])
        self.assertEqual(result["retry_target"], "answer")

    def test_answer_without_citation_cannot_pass_on_score_alone(self) -> None:
        previous = self.workflow.offline
        self.workflow.offline = False
        try:
            result = self.workflow.evaluation_node(
                {
                    "draft_answer": (
                        "这是一段长度足够、句子完整，但完全没有证据编号的回答。"
                        "它不应仅凭其他评分项达到阈值后被错误放行。"
                        "回答还包含外观、栖息环境、生态行为和相似物种等多个完整说明，"
                        "用于确保测试只针对缺少引用这一项，而不会触发答案过短或截断检查。"
                    ),
                    "final_context": "[E1] 海星具有中央盘和腕足。",
                    "route": {"need_vlm": False},
                    "trace": [],
                }
            )["evaluation_result"]
        finally:
            self.workflow.offline = previous
        self.assertFalse(result["passed"])
        self.assertEqual(result["retry_target"], "retrieval")

    def test_provider_failure_cannot_pass_evaluation(self) -> None:
        result = self.workflow.evaluation_node(
            {
                "draft_answer": "API failed [E1]",
                "generation_failed": True,
                "final_context": "[E1] evidence",
                "route": {"need_vlm": False},
                "trace": [],
            }
        )["evaluation_result"]
        self.assertFalse(result["passed"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["retry_target"], "none")

    def test_vision_failure_degrades_without_raising(self) -> None:
        class FailingVisionClient:
            enabled = True

            @staticmethod
            def analyze_image(*args, **kwargs):
                raise ValueError("invalid vision JSON")

        previous = self.workflow.llm
        self.workflow.llm = FailingVisionClient()
        try:
            result = self.workflow.vision_node(
                {
                    "image_path": "image.jpg",
                    "warnings": [],
                    "trace": [],
                }
            )
        finally:
            self.workflow.llm = previous
        self.assertIn("image_caption", result)
        self.assertTrue(result["warnings"])
        self.assertEqual(result["trace"][-1], "vision:fallback:ValueError")


class OpenRouterResponseTests(unittest.TestCase):
    def test_incomplete_citation_is_truncated(self) -> None:
        self.assertTrue(OpenRouterClient._looks_truncated("结论[E"))
        self.assertTrue(OpenRouterClient._looks_truncated("结论[E1"))
        self.assertFalse(OpenRouterClient._looks_truncated("结论[E1]。"))

    def test_extract_json_from_markdown_or_prose(self) -> None:
        fenced = '```json\n{"description":"海星"}\n```'
        prose = '分析结果如下：{"description":"海胆","confidence":0.8}。'
        self.assertEqual(
            OpenRouterClient._extract_json_object(fenced)["description"],
            "海星",
        )
        self.assertEqual(
            OpenRouterClient._extract_json_object(prose)["description"],
            "海胆",
        )

    def test_normalize_partial_vision_json(self) -> None:
        result = OpenRouterClient._normalize_image_analysis(
            {
                "detailed_description": "五腕放射状生物",
                "possible_species": "starfish",
                "confidence": "0.7",
            }
        )
        self.assertEqual(result["description"], "五腕放射状生物")
        self.assertEqual(result["possible_species"], ["starfish"])
        self.assertEqual(result["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
