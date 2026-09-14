from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from aquabio_mrag.cancellation import check_cancelled
from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.conversation import ConversationStore
from aquabio_mrag.vector_db import ChromaMRAGStore
from aquabio_web.api import app
from aquabio_web.schemas import ChatRequest
from aquabio_web.service import ChatService
from aquabio_web.tasks import BackgroundTasks, TaskQueueFull


class APIRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.service = ChatService(self.root)
        self.addCleanup(self.service.close)
        for target, value in [("aquabio_web.api.SERVICE", self.service), ("aquabio_web.api.ROOT", self.root)]:
            replacement = patch(target, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.client = TestClient(app)
        image = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        self.png = image.getvalue()

    def test_files_only_serve_public_media(self):
        (self.root / ".env").write_text("placeholder")
        for name in [".env", "data/mrag/sessions/web_app.sqlite"]:
            self.assertEqual(self.client.get("/files/" + name).status_code, 404)
        upload = self.service.save_upload("alpha", "x.png", self.png, "image")
        response = self.client.get(upload["url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.png)
        self.assertEqual(self.client.get("/files/data/mrag/uploads/%2e%2e/%2e%2e/%2e%2e/.env").status_code, 404)

    def test_invalid_upload_does_not_leave_files(self):
        for filename, kind in [("x.png", "image"), ("x.pdf", "pdf")]:
            with self.assertRaises(ValueError):
                self.service.save_upload("alpha", filename, b"invalid", kind)
        self.assertEqual(list(self.service.paths.uploads_dir.rglob("x.*")), [])
        self.assertFalse(any(p.is_file() for p in self.service.paths.uploads_dir.rglob("*")))
        with patch("aquabio_web.api.UPLOAD_LIMITS", {"image": 1}):
            response = self.client.post("/api/uploads/image", data={"session_id": "alpha"}, files={"file": ("x.png", self.png)})
        self.assertEqual(response.status_code, 400)

    def test_detection_receives_bytes(self):
        detector = SimpleNamespace(YOLOv8Detector=lambda: object(),
            AgentDetectionOrchestrator=lambda detector: SimpleNamespace(analyze_and_detect=lambda *a, **kw: {"original_detection": {"detections": []}}))
        with patch.dict("sys.modules", {"aquabio.detector": detector}):
            response = self.client.post("/api/detection", data={"session_id": "alpha"}, files={"file": ("x.png", self.png, "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        path = self.root / response.json()["upload"]["file_path"]
        self.assertEqual(path.read_bytes(), self.png)

    def test_attachment_cannot_cross_sessions(self):
        upload = self.service.save_upload("alpha", "x.png", self.png, "image")
        response = self.client.post("/api/chat", json={"session_id": "beta", "query": "identify", "attachments": [{"file_id": upload["file_id"], "type": "image"}]})
        self.assertEqual(response.status_code, 400)

    def test_invalid_session_ids_rejected(self):
        for value in ["a b", "Alpha", "../x", "con", "x" * 81]:
            response = self.client.post("/api/sessions", json={"session_id": value})
            self.assertEqual(response.status_code, 422, value)

    def test_sse_emits_actual_node_progress(self):
        def chat(request, progress_callback=None):
            progress_callback("retrieval:done")
            progress_callback("answer:done")
            return {"answer": "ok"}
        with patch.object(self.service, "chat", side_effect=chat):
            response = self.client.post("/api/chat/stream", json={"session_id": "alpha", "query": "test"})
        self.assertIn("event: node_progress", response.text)
        self.assertLess(response.text.index('"node": "retrieval"'), response.text.index("event: final"))


class ConversationConcurrencyTests(unittest.TestCase):
    def test_concurrent_writes_do_not_lose_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def append(index):
                ConversationStore(root).append_turn("alpha", {"user_query": str(index)}, {})
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(append, range(32)))
            turns = ConversationStore(root).load("alpha")["turns"]
            self.assertEqual(len(turns), 32)
            self.assertEqual([turn["turn_index"] for turn in turns], list(range(1, 33)))
            self.assertEqual(len({turn["user_query"] for turn in turns}), 32)


class TaskTests(unittest.TestCase):
    def wait_finished(self, queue, task_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            task = queue.chat_task(task_id)
            if task["finished_at"]:
                return task
            time.sleep(.01)
        self.fail("worker did not finish")

    def test_bounded_queue_and_cancel_before_side_effect(self):
        queue = BackgroundTasks()
        queue._init_tasks(max_workers=1, max_pending=2)
        self.addCleanup(queue.close)
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        completed = []
        def chat(request, progress_callback):
            started.set()
            release.wait(2)
            check_cancelled()
            completed.append(request.session_id)
            return {}
        queue.chat = chat
        request = ChatRequest(session_id="alpha", query="test")
        first = queue.submit_chat(request)
        self.assertTrue(started.wait(1))
        second = queue.submit_chat(request)
        with self.assertRaises(TaskQueueFull):
            queue.submit_chat(request)
        queue.cancel_chat_task(first["task_id"])
        queue.cancel_chat_task(second["task_id"])
        release.set()
        self.assertEqual(self.wait_finished(queue, first["task_id"])["status"], "cancelled")
        self.assertEqual(self.wait_finished(queue, second["task_id"])["status"], "cancelled")
        self.assertEqual(completed, [])

    def test_completed_tasks_expire(self):
        queue = BackgroundTasks()
        queue._init_tasks(retention_seconds=0)
        self.addCleanup(queue.close)
        queue.chat = lambda *a, **kw: {}
        task = queue.submit_chat(ChatRequest(session_id="alpha", query="test"))
        self.wait_finished(queue, task["task_id"])
        self.assertEqual(queue.chat_tasks(), [])


class BM25CacheTests(unittest.TestCase):
    def test_statistics_reused_and_invalidated_after_source_change(self):
        from aquabio_mrag.retrieval_agent import BookNativeBM25, _tokens
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/mrag/raganything/book_native/book/rag_chunks.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({"chunk_id": "one", "content": "starfish orange"}) + "\n")
            index = BookNativeBM25(root)
            with patch("aquabio_mrag.retrieval_agent._tokens", wraps=_tokens) as tokens:
                self.assertEqual(index.search("starfish")[0].id, "one")
                before = tokens.call_count
                index.search("starfish")
                self.assertEqual(tokens.call_count - before, 1)
            source.write_text(json.dumps({"chunk_id": "two", "content": "lobster antennae"}) + "\n")
            self.assertEqual(index.search("lobster")[0].id, "two")
            self.assertEqual(index.search("starfish"), [])


class FakeCollection:
    def __init__(self, name, count=0, fail=False):
        self.name, self.total, self.fail = name, count, fail
        self.metadata = {}
    def count(self):
        return self.total
    def add(self, **kwargs):
        if self.fail:
            raise RuntimeError("injected write failure")
        self.total += len(kwargs["ids"])


class FakeClient:
    def __init__(self):
        self.collections = {"aquabio_mrag": FakeCollection("aquabio_mrag", 5)}
        self.fail = False
    def create_collection(self, name, **kwargs):
        self.collections[name] = FakeCollection(name, fail=self.fail)
        return self.collections[name]
    def get_collection(self, name):
        return self.collections[name]
    def delete_collection(self, name):
        del self.collections[name]


class IndexRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = ChromaMRAGStore.__new__(ChromaMRAGStore)
        self.store.paths = MRAGPaths.from_root(directory.name)
        self.store.paths.ensure()
        self.store.settings = MRAGSettings()
        self.store.client = FakeClient()
        self.store.embedder = SimpleNamespace(encode_documents=lambda texts, **kw: [[1., 0.] for _ in texts])
        source = self.store.paths.knowledge_dir / "rag_documents_combined.jsonl"
        source.write_text(json.dumps({"id": "one", "content": "evidence", "embedding_text": "evidence", "source_type": "species_card", "modality": "text"}) + "\n")

    def test_embedding_failure_preserves_old_collection(self):
        with patch.object(self.store, "_embedder", return_value=None):
            with self.assertRaises(RuntimeError):
                self.store.build()
        self.assertEqual(self.store.collection().count(), 5)
        self.assertFalse(self.store.manifest_path.exists())

    def test_partial_write_failure_preserves_old_collection(self):
        self.store.client.fail = True
        with self.assertRaises(RuntimeError):
            self.store.build()
        self.assertEqual(list(self.store.client.collections), ["aquabio_mrag"])
        self.assertEqual(self.store.collection().count(), 5)

    def test_success_switches_pointer_and_retains_previous(self):
        manifest = self.store.build()
        self.assertNotEqual(manifest["collection_name"], "aquabio_mrag")
        self.assertEqual(self.store.collection().count(), 1)
        self.assertEqual(self.store.client.get_collection("aquabio_mrag").count(), 5)
        self.assertEqual(self.store.info()["collection_count"], 1)

    def test_manifest_write_failure_preserves_old_index(self):
        with patch("pathlib.Path.replace", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.store.build()
        self.assertEqual(self.store.collection().count(), 5)
        self.assertEqual(list(self.store.client.collections), ["aquabio_mrag"])
        self.assertEqual(list(self.store.paths.vector_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
