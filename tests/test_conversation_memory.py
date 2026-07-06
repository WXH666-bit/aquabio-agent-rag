from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.conversation import ConversationStore
from aquabio_mrag.workflow import AquaBioMRAGWorkflow


ROOT = Path(__file__).resolve().parents[1]
BASE_PATHS = MRAGPaths.from_root(ROOT)


class ConversationStoreTests(unittest.TestCase):
    def test_turns_are_persistent_and_sessions_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory))
            store.append_turn(
                "alpha",
                {"user_query": "海星", "assistant_answer": "回答"},
                {"last_species_ids": ["starfish"]},
            )

            self.assertEqual(len(store.load("alpha")["turns"]), 1)
            self.assertEqual(store.load("beta")["turns"], [])
            self.assertTrue(store.path_for("alpha").is_file())

    def test_session_id_is_normalized(self) -> None:
        self.assertEqual(
            ConversationStore.normalize_session_id(" demo session "),
            "demo_session",
        )


class FollowupWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = MRAGSettings.from_env()

    def test_current_image_reference_is_not_a_followup(self) -> None:
        workflow = AquaBioMRAGWorkflow(
            BASE_PATHS, self.settings, offline=True
        )
        result = workflow.followup_resolver_node(
            {
                "original_query": "这个生物是什么？",
                "image_path": "image.jpg",
                "memory_summary": {
                    "last_species_ids": ["sea_urchin"]
                },
                "warnings": [],
                "trace": [],
            }
        )
        self.assertFalse(result["followup_detected"])
        self.assertEqual(result["resolved_species_ids"], [])

    def test_followup_without_history_requests_clarification(self) -> None:
        workflow = AquaBioMRAGWorkflow(
            BASE_PATHS, self.settings, offline=True
        )
        result = workflow.followup_resolver_node(
            {
                "original_query": "刚才那个生物是什么？",
                "memory_summary": {},
                "warnings": [],
                "trace": [],
            }
        )
        self.assertTrue(result["need_clarification"])
        self.assertIn("请先提供图片", result["clarification_question"])

    def test_cross_process_equivalent_followup_resolves_starfish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = replace(
                BASE_PATHS, sessions_dir=Path(directory)
            )
            image = (
                ROOT
                / "data"
                / "mrag"
                / "images"
                / "starfish"
                / "img_starfish_001.jpg"
            )
            first = AquaBioMRAGWorkflow(
                paths, self.settings, offline=True
            ).invoke(
                "这个生物的外貌是什么样子的？",
                str(image),
                session_id="memory_test",
            )
            self.assertIn("starfish", first["memory_summary"]["last_species_ids"])

            second = AquaBioMRAGWorkflow(
                paths, self.settings, offline=True
            ).invoke(
                "刚才问到的是什么生物，然后只回答它的常见颜色，"
                "要简短，不要回答其他内容？",
                session_id="memory_test",
            )

            self.assertEqual(second["route"]["task_type"], "followup_text_qa")
            self.assertEqual(second["selected_tools"], ["text_retriever"])
            self.assertEqual(second["resolved_species_ids"], ["starfish"])
            self.assertIn("橙色", second["final_answer"])
            self.assertTrue(
                any(
                    "filter_species=starfish" in item
                    for item in second["trace"]
                )
            )
            saved = ConversationStore(Path(directory)).load("memory_test")
            self.assertTrue(
                saved["turns"][-1]["trace"][-1].startswith("memory_save:")
            )
            self.assertEqual(
                saved["summary"]["last_answer_summary"],
                second["final_answer"],
            )
            self.assertTrue(saved["summary"]["last_evidence_ids"])


if __name__ == "__main__":
    unittest.main()
