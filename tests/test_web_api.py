from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from aquabio_web.api import app
from aquabio_web.presentation import (
    evidence_source_counts,
    mcp_activity,
    selected_tools,
    trace_phase_summary,
)
from aquabio_web.schemas import ChatOptions, ChatRequest
from aquabio_web.service import ChatService
from aquabio_web.store import WebStore
from aquabio_web.web_knowledge import plan_web_research


class WebStoreTests(unittest.TestCase):
    def test_session_messages_evidence_and_trace_are_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WebStore(Path(directory) / "web.sqlite")
            session = store.create_session("测试会话", "web_test")
            self.assertEqual(session["title"], "测试会话")
            store.save_turn(
                "web_test",
                "turn_1",
                "海星有什么特征？",
                "海星具有放射状腕足。[E1]",
                [],
                [
                    {
                        "evidence_id": "E1",
                        "source_system": "chroma",
                        "source_type": "species_text_chunk",
                        "content": "放射状腕足",
                        "score": 0.9,
                        "metadata": {},
                    }
                ],
                [
                    {
                        "node": "router",
                        "event": "completed",
                        "detail": "text_qa",
                    }
                ],
            )
            loaded = store.get_session("web_test")
            self.assertEqual(len(loaded["messages"]), 2)
            listed = store.list_sessions()
            self.assertEqual(listed[0]["turn_count"], 1)
            self.assertEqual(
                loaded["messages"][0]["attachments"], []
            )
            self.assertEqual(
                store.evidence_for_turn("web_test", "turn_1")[0][
                    "evidence_id"
                ],
                "E1",
            )

    def test_first_turn_replaces_default_session_title(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WebStore(Path(directory) / "web.sqlite")
            store.create_session("新会话", "title_test")
            store.save_turn(
                "title_test",
                "turn_1",
                "这是什么海马？",
                "这是海马。",
                [],
                [],
                [],
            )
            self.assertEqual(
                store.get_session("title_test")["title"],
                "这是什么海马？",
            )

    def test_single_image_prefers_pdf_and_returns_one(self) -> None:
        images = [
            {
                "image_id": "network",
                "source": "wikimedia_commons",
                "score": 0.99,
                "image_role": "specimen",
            },
            {
                "image_id": "local",
                "source": "image_retrieval",
                "score": 0.95,
                "image_role": "specimen",
            },
            {
                "image_id": "pdf",
                "source": "raganything_pdf_image",
                "score": 0.5,
                "image_role": "specimen",
            },
        ]
        selected = ChatService._single_best_image(images, "给我一张图片")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["image_id"], "pdf")

    def test_distribution_map_never_falls_back_to_specimen(self) -> None:
        selected = ChatService._single_best_image(
            [
                {
                    "image_id": "specimen",
                    "source": "image_retrieval",
                    "score": 0.99,
                    "image_role": "specimen",
                }
            ],
            "给我它的分布图",
        )
        self.assertEqual(selected, [])

    def test_distribution_notice_names_network_source(self) -> None:
        notice = ChatService._distribution_map_notice(
            {
                "caption": "File:Tiger shark distmap.png",
                "source": "wikimedia_commons",
            }
        )
        self.assertIn("Wikimedia Commons", notice)
        self.assertIn("Tiger shark", notice)
        self.assertIn("不代表整个上位类群", notice)

    def test_image_identification_without_image_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ChatService(Path(directory))
            service.store.create_session("新会话", "missing_image")
            with patch.object(service, "workflow") as workflow:
                workflow.return_value.llm.settings.provider = "test"
                workflow.return_value.llm.settings.model = "mock"
                response = service.chat(
                    ChatRequest(
                        session_id="missing_image",
                        query="这是什么生物",
                        options=ChatOptions(log_enabled=False),
                    )
                )
            self.assertFalse(workflow.return_value.invoke.called)
            self.assertIn("没有收到", response["answer"])
            self.assertFalse(response["model"]["called"])
            self.assertEqual(
                response["trace"][0]["node"], "attachment_guard"
            )

    def test_starfish_web_fallback_targets_predator(self) -> None:
        class DisabledLLM:
            enabled = False

        plan = plan_web_research(
            DisabledLLM(),
            "Starfish",
            "它的天敌有哪些并给我图片",
        )
        self.assertEqual(plan["search_query"], "Starfish")
        self.assertEqual(plan["image_target"], "sea otter")

    def test_feedback_is_upserted_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WebStore(Path(directory) / "web.sqlite")
            store.create_session("反馈", "feedback_test")
            first = store.save_feedback(
                "feedback_test", "turn_1", 1, "helpful"
            )
            self.assertEqual(first["rating"], 1)
            second = store.save_feedback(
                "feedback_test", "turn_1", -1, "needs sources"
            )
            self.assertEqual(second["rating"], -1)
            stats = store.feedback_stats()
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["negative"], 1)

    def test_trace_presentation_uses_real_events(self) -> None:
        trace = [
            {
                "node": "source_selection",
                "detail": "text_retriever,pdf_retriever",
            },
            {
                "node": "retrieval",
                "detail": "text=3,mcp_pdf_images=2",
            },
            {"node": "evaluation", "detail": "True:1.00"},
        ]
        phases = trace_phase_summary(trace)
        self.assertEqual([row["status"] for row in phases], [
            "complete", "complete", "pending", "complete"
        ])
        self.assertEqual(
            selected_tools(trace), ["text_retriever", "pdf_retriever"]
        )
        self.assertEqual(mcp_activity(trace)["pdf_images"], 2)
        graph_activity = mcp_activity(
            [],
            [{"source_system": "lightrag_graph", "metadata": {}}],
        )
        self.assertTrue(graph_activity["enabled"])
        self.assertTrue(graph_activity["graph_retrieval"])
        sources = evidence_source_counts(
            [{"source_system": "chroma"}, {"source_system": "chroma"}]
        )
        self.assertEqual(sources, [{"source": "Chroma", "count": 2}])

    def test_invalid_question_mark_title_keeps_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WebStore(Path(directory) / "web.sqlite")
            store.create_session("新会话", "invalid_title")
            store.save_turn(
                "invalid_title",
                "turn_1",
                "????????",
                "answer",
                [],
                [],
                [],
            )
            self.assertEqual(
                store.get_session("invalid_title")["title"],
                "新会话",
            )

    def test_image_attachment_is_restored_with_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = WebStore(Path(directory) / "web.sqlite")
            attachment = {
                "file_id": "image_1",
                "file_type": "image",
                "file_name": "seahorse.jpg",
                "file_path": "uploads/seahorse.jpg",
                "size_bytes": 123,
            }
            store.save_turn(
                "attachment_test",
                "turn_1",
                "这是什么？",
                "这是海马。",
                [attachment],
                [],
                [],
            )
            loaded = store.get_session("attachment_test")
            self.assertEqual(
                loaded["messages"][0]["attachments"][0]["file_id"],
                "image_1",
            )


class FastAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_and_session_crud(self) -> None:
        self.assertEqual(
            self.client.get("/api/health").json()["status"], "ok"
        )
        session_id = "api_test_session"
        self.client.delete(f"/api/sessions/{session_id}")
        created = self.client.post(
            "/api/sessions",
            json={"session_id": session_id, "title": "API 测试"},
        )
        self.assertEqual(created.status_code, 200)
        updated = self.client.patch(
            f"/api/sessions/{session_id}",
            json={"is_favorite": True, "tags": ["测试"]},
        ).json()
        self.assertTrue(updated["is_favorite"])
        self.assertEqual(updated["tags"], ["测试"])
        self.assertEqual(
            self.client.get(f"/api/sessions/{session_id}").status_code,
            200,
        )
        self.client.delete(f"/api/sessions/{session_id}")

    def test_architecture_and_feedback_contracts(self) -> None:
        self.client.delete("/api/sessions/api_feedback")
        architecture = self.client.get("/api/system/architecture")
        self.assertEqual(architecture.status_code, 200)
        self.assertEqual(
            architecture.json()["workflow"]["type"],
            "LangGraph StateGraph",
        )
        response = self.client.post(
            "/api/feedback",
            json={
                "session_id": "api_feedback",
                "turn_id": "turn_1",
                "rating": 1,
                "comment": "good",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rating"], 1)
        deleted = self.client.delete("/api/sessions/api_feedback")
        self.assertTrue(deleted.json()["deleted"])

    def test_chat_response_contract(self) -> None:
        fake = {
            "session_id": "contract",
            "turn_id": "turn_contract",
            "answer": "测试回答 [E1]",
            "answer_type": "text",
            "images": [],
            "evidence": [],
            "trace": [],
            "warnings": [],
            "memory": {},
            "route": {"task_type": "text_qa"},
            "pending_review": False,
        }
        with patch(
            "aquabio_web.api.SERVICE.chat", return_value=fake
        ):
            response = self.client.post(
                "/api/chat",
                json={
                    "session_id": "contract",
                    "query": "测试",
                    "attachments": [],
                    "options": {},
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "测试回答 [E1]")

    def test_streaming_contract_emits_final_event(self) -> None:
        fake = {
            "session_id": "stream",
            "turn_id": "turn_stream",
            "answer": "完成",
            "answer_type": "text",
            "images": [],
            "evidence": [],
            "trace": [],
            "warnings": [],
            "memory": {},
            "route": {"task_type": "text_qa"},
            "pending_review": False,
        }
        with patch(
            "aquabio_web.api.SERVICE.chat", return_value=fake
        ):
            response = self.client.post(
                "/api/chat/stream",
                json={
                    "session_id": "stream",
                    "query": "测试",
                    "attachments": [],
                    "options": {},
                },
            )
        self.assertIn("event: node_start", response.text)
        self.assertIn("event: final", response.text)

    def test_background_task_contract(self) -> None:
        fake = {
            "session_id": "background",
            "turn_id": "turn_background",
            "answer": "后台完成",
            "answer_type": "text",
            "images": [],
            "evidence": [],
            "trace": [],
            "warnings": [],
            "memory": {},
            "route": {"task_type": "text_qa"},
            "pending_review": False,
        }
        with patch(
            "aquabio_web.api.SERVICE.chat", return_value=fake
        ):
            created = self.client.post(
                "/api/chat/tasks",
                json={
                    "session_id": "background",
                    "query": "测试",
                    "attachments": [],
                    "options": {},
                },
            ).json()
            task_id = created["task_id"]
            for _ in range(50):
                task = self.client.get(
                    f"/api/chat/tasks/{task_id}"
                ).json()
                if task["status"] == "completed":
                    break
                import time

                time.sleep(0.01)
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result"]["answer"], "后台完成")


if __name__ == "__main__":
    unittest.main()
