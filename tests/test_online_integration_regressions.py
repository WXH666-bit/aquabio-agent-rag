"""Offline regressions for failures discovered during real StepFun/MCP tests."""
import asyncio
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from aquabio.config import Settings
from aquabio.openrouter import OpenRouterClient
from aquabio_mrag.mcp_client import MCPServerConfig, MCPStdioClient, project_mcp_client
from aquabio_raganything.config import RAGAnythingSettings
from aquabio_raganything.runtime import LocalBGEEmbedding


class OnlineIntegrationRegressions(unittest.TestCase):
    def test_full_mcp_preserves_graph_page_during_rerank(self):
        from aquabio_mrag.config import MRAGPaths, MRAGSettings
        from aquabio_mrag.workflow import AquaBioMRAGWorkflow
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            paths = replace(MRAGPaths.from_root(root), sessions_dir=Path(temp) / "sessions", vector_dir=Path(temp) / "chroma")
            with patch("aquabio_mrag.vector_db.chromadb.PersistentClient"):
                workflow = AquaBioMRAGWorkflow(paths, MRAGSettings())
            workflow.llm.offline = True
            def tool(server, name, arguments):
                if name == "raganything_hybrid_search":
                    return {"evidence": [{"id": "graph_target", "doc_id": "doc_target", "page": 429, "source_file": "guide.pdf", "content": "Luidia sarsii africana has long flattened arms.", "rank": 1}], "relations": [], "warnings": []}
                if name == "search_pdf":
                    return [{"id": f"irrelevant_{i}", "content": "Coral ecology", "final_score": 0.99, "metadata": {"species_id": "coral", "source_type": "pdf_chunk"}} for i in range(12)]
                return []
            try:
                with patch.object(workflow.retrieval_agent.mcp, "call_tool_sync", side_effect=tool):
                    state = workflow.invoke("根据 PDF 第429页介绍 Luidia sarsii africana 的特征", session_id="graph-regression", options={"mcp_retrieval_enabled": True})
                self.assertIn("page 429", state["final_context"])
                self.assertIn("long flattened arms", state["final_context"])
                self.assertTrue(any(c["tool_name"] == "raganything_hybrid_search" for c in state["tool_calls"]))
                self.assertNotIn("coral", state["memory_summary"]["last_species_ids"])
            finally:
                workflow.close()

    def test_short_complete_cited_answer_is_not_treated_as_truncated(self):
        from aquabio_mrag.answers import AnswerNodes
        nodes = AnswerNodes()
        nodes.offline = False
        result = nodes.evaluation_node({"draft_answer": "海星通常生活在海底岩石和沙泥底环境[E1]。",
                                        "final_context": "[E1] 海星栖息环境", "route": {"need_vlm": False}, "trace": []})
        self.assertTrue(result["evaluation_result"]["passed"])

    def test_stepfun_configuration_and_reasoning_budget(self):
        with patch.dict(os.environ, {"AQUABIO_LLM_PROVIDER": "stepfun", "STEPFUN_API_KEY": "test-key"}, clear=True), patch("aquabio.config.load_env"):
            settings = Settings.from_env()
            self.assertEqual(settings.provider, "stepfun")
            self.assertEqual(settings.model, "step-3.7-flash")
            self.assertEqual(settings.base_url, "https://api.stepfun.com/step_plan/v1")
            client = OpenRouterClient(settings)
            response = MagicMock(ok=True)
            response.json.return_value = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"function": {"name": "search_pdf", "arguments": '{"query":"sea star"}'}}]}}]}
            with patch("aquabio.openrouter.requests.post", return_value=response) as post:
                calls = client.select_tools([], [])
                self.assertEqual(calls[0]["name"], "search_pdf")
                self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")
                self.assertGreaterEqual(post.call_args.kwargs["json"]["max_tokens"], 8192)

    def test_empty_reasoning_only_completion_is_an_error(self):
        response = MagicMock(ok=True)
        response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        client = OpenRouterClient(Settings(api_key="test-key", provider="stepfun"))
        with patch("aquabio.openrouter.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty content"):
                client.chat([{"role": "user", "content": "test"}])
            with self.assertRaisesRegex(RuntimeError, "token budget"):
                client.select_tools([], [])

    def test_mcp_error_response_is_not_a_successful_result(self):
        async def run():
            session = AsyncMock()
            session.call_tool.return_value = SimpleNamespace(isError=True, content=[SimpleNamespace(text="broken tool")])
            session_context = AsyncMock()
            session_context.__aenter__.return_value = session
            transport = AsyncMock()
            transport.__aenter__.return_value = (None, None, None)
            client = MCPStdioClient({"test": MCPServerConfig("test", "", [], url="http://localhost/mcp", retries=0)})
            with patch("aquabio_mrag.mcp_client.streamable_http_client", return_value=transport), patch("aquabio_mrag.mcp_client.ClientSession", return_value=session_context):
                with self.assertRaisesRegex(RuntimeError, "MCP tool error"):
                    await client._session_call("test", "call", "broken")
        asyncio.run(run())

    def test_mcp_uses_current_interpreter_without_project_venv(self):
        import sys
        with patch.object(Path, "is_file", return_value=False):
            self.assertEqual(project_mcp_client(Path.cwd()).servers["chroma"].command, sys.executable)

    def test_onnx_rejects_different_embedding_space(self):
        with patch.dict(os.environ, {"MRAG_EMBEDDING_BACKEND": "onnx"}):
            embedding = LocalBGEEmbedding(RAGAnythingSettings(embedding_model="BAAI/bge-m3", embedding_dim=1024))
            with self.assertRaisesRegex(ValueError, "384"):
                embedding._load()


if __name__ == "__main__":
    unittest.main()
