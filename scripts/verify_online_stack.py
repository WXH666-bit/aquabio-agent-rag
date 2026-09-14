"""Opt-in paid integration test: real LLM, stdio + HTTP MCP, and LangGraph.

Run with the project's Python after configuring .env and indexing one PDF unit.
No mocks or local answer fallback count as success. Does not store credentials.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from aquabio.config import Settings, load_env
from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.mcp_client import project_mcp_client


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path("docs/online-validation.json"))
    parser.add_argument("--tools-only", action="store_true")
    parser.add_argument("--only-failed", action="store_true", help="Re-run failed checks in the previous report")
    args = parser.parse_args()
    root = args.root.resolve()
    load_env(root / ".env")
    settings = Settings.from_env()
    require(settings.api_key, "Online provider key is required")
    os.environ.setdefault("AQUABIO_LLM_READ_TIMEOUT", "180")
    os.environ.setdefault("RAGANYTHING_QUERY_TIMEOUT", "180")
    os.environ.setdefault("RAGANYTHING_MCP_TIMEOUT", "210")
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "provider": settings.provider,
              "model": settings.model, "base_url": settings.base_url, "checks": [],
              "scope": "One real PDF unit: sa_taxon_lucafr_p0429; existing Chroma corpus; 16 MCP tools"}
    report_path = args.report.resolve()
    previous = {}
    if args.only_failed and report_path.is_file():
        old = json.loads(report_path.read_text(encoding="utf-8"))
        require(old.get("model") == settings.model and old.get("base_url") == settings.base_url, "Report provider changed; run a full validation")
        previous = {row["name"]: row for row in old["checks"]}

    def check(name, operation):
        if previous.get(name, {}).get("passed") and not name.startswith("workflow."):
            report["checks"].append(previous[name])
            return previous[name]
        started = time.perf_counter()
        try:
            detail = operation()
            row = {"name": name, "passed": True, "detail": detail}
        except Exception as error:
            # Reports contain only final answers/metadata, never HTTP headers or reasoning.
            row = {"name": name, "passed": False, "error": str(error).replace(settings.api_key, "[REDACTED]")[:1000]}
        row["seconds"] = round(time.perf_counter() - started, 3)
        report["checks"].append(row)
        report["passed"] = all(item["passed"] for item in report["checks"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name}: {row.get('error', '')}", flush=True)
        return row

    with tempfile.TemporaryDirectory(prefix="aquabio-online-") as scratch:
        scratch = Path(scratch)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update(AQUABIO_RAG_MCP_PORT=str(port), AQUABIO_RAG_MCP_TRANSPORT="streamable-http",
                   PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        os.environ["RAGANYTHING_MCP_URL"] = f"http://127.0.0.1:{port}/mcp"
        with (scratch / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable, "-m", "aquabio_raganything.mcp_server"],
                                      cwd=root, env=env, stdout=log, stderr=log,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workflow = None
            try:
                deadline = time.monotonic() + 40
                while True:
                    require(server.poll() is None, "Graph MCP server exited during startup")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        require(time.monotonic() < deadline, "Graph MCP startup timeout")
                        time.sleep(0.3)
                client = project_mcp_client(root)
                entity = "Luidia sarsii africana"
                doc = "doc_sa_invertebrates_p0429"
                image = root / "data/mrag/images/starfish/img_starfish_001.jpg"
                query = "Luidia sarsii africana distinguishing features habitat"
                ids = {}

                def discovery(server_name):
                    names = [item["name"] for item in client.list_tools_sync(server_name)]
                    require(len(names) == 8, f"Expected 8 tools, got {names}")
                    return names

                for name in ("chroma", "raganything"):
                    check(f"{name}.discovery", lambda name=name: discovery(name))

                def native_tool_chain():
                    from aquabio.openrouter import OpenRouterClient
                    schema = next(t for t in client.list_tools_sync("chroma") if t["name"] == "search_species_text")
                    calls = OpenRouterClient(settings).select_tools(
                        [{"role": "user", "content": "Call search_species_text to retrieve starfish morphology. Use query='starfish morphology' and top_k=3."}],
                        [{"type": "function", "function": {"name": schema["name"], "description": schema["description"], "parameters": schema["input_schema"]}}],
                    )
                    require(calls and calls[0]["name"] == "search_species_text", "No native function call")
                    rows = client.call_tool_sync("chroma", calls[0]["name"], calls[0]["arguments"])
                    require(rows, "Native tool call returned no MCP evidence")
                    return {"tool": calls[0]["name"], "result_count": len(rows)}

                check("model.native_tool_to_mcp", native_tool_chain)

                def call(server_name, tool, arguments, predicate):
                    value = client.call_tool_sync(server_name, tool, arguments)
                    require(predicate(value), f"Empty, incomplete or fallback result from {tool}")
                    if isinstance(value, list):
                        if value:
                            ids["chroma"] = value[0].get("id", "")
                        return {"count": len(value)}
                    if tool == "raganything_index_status":
                        return {key: value[key] for key in ("storage_valid", "graph_nodes", "graph_edges", "chunks")}
                    if tool == "raganything_hybrid_search":
                        return {"entities": len(value["entities"]), "relations": len(value["relations"]),
                                "pages": sorted({e["page"] for e in value["evidence"] if e.get("page")})}
                    return {"keys": list(value)}

                jobs = [
                    ("chroma", "search_species_text", {"query": "starfish", "candidate_species": "starfish", "top_k": 3}, lambda v: bool(v) and all(x["metadata"]["species_id"] == "starfish" for x in v)),
                    ("chroma", "search_image_captions", {"query": "starfish", "top_k": 3}, bool),
                    ("chroma", "search_pdf", {"query": query, "top_k": 3}, bool),
                    ("chroma", "search_multimodal", {"query": "starfish", "image_caption": "sea star with arms", "top_k": 3}, bool),
                    ("chroma", "search_pdf_entity_images", {"query": entity, "entity": entity}, lambda v: v.get("count", 0) > 0 and not v.get("warnings")),
                    ("chroma", "get_entity_sample_images", {"entity": entity}, lambda v: v.get("count", 0) > 0),
                    ("chroma", "generate_image_caption", {"image_path": str(image)}, lambda v: bool(v.get("description")) and v.get("structured_output") is True),
                    ("chroma", "get_source_detail", None, lambda v: bool(v.get("documents"))),
                    ("raganything", "raganything_index_status", {}, lambda v: v.get("storage_valid") and v.get("graph_nodes", 0) > 0 and v.get("graph_edges", 0) > 0),
                    ("raganything", "raganything_graph_neighbors", {"entity": entity}, lambda v: bool(v.get("nodes")) and bool(v.get("relations")) and not v.get("warnings")),
                    ("raganything", "raganything_hybrid_search", {"query": query, "top_k": 6}, lambda v: bool(v.get("entities")) and bool(v.get("relations")) and any(e.get("page") == 429 for e in v.get("evidence", [])) and not v.get("warnings")),
                    ("raganything", "raganything_image_search", {"query": entity, "entity": entity}, lambda v: v.get("count", 0) > 0),
                    ("raganything", "search_pdf_images", {"query": entity, "entity": entity}, lambda v: v.get("count", 0) > 0),
                    ("raganything", "raganything_entity_images", {"entity": entity}, lambda v: v.get("count", 0) > 0),
                    ("raganything", "raganything_source_detail", {"doc_id": doc}, lambda v: doc in json.dumps(v) and "429" in json.dumps(v)),
                    ("raganything", "get_source_detail", {"doc_id": doc}, lambda v: doc in json.dumps(v) and "429" in json.dumps(v)),
                ]
                for name, tool, arguments, predicate in jobs:
                    arguments = arguments if arguments is not None else {"doc_id": ids.get("chroma", "")}
                    check(f"{name}.{tool}", lambda n=name, t=tool, a=arguments, p=predicate: call(n, t, a, p))

                if not args.tools_only:
                    from aquabio_mrag.workflow import AquaBioMRAGWorkflow
                    paths = replace(MRAGPaths.from_root(root), sessions_dir=scratch / "sessions")
                    workflow = AquaBioMRAGWorkflow(paths, MRAGSettings.from_env())
                    copied_image = scratch / "unregistered-image.jpg"
                    shutil.copyfile(image, copied_image)

                    def conversation(question, session, image_path=None, followup=False):
                        state = workflow.invoke(question, session_id=session, image_path=image_path,
                                                progress_callback=lambda node: print(f"  {session}: {node}", flush=True),
                                                options={"mcp_enabled": True, "mcp_retrieval_enabled": True,
                                                         "web_enabled": False, "network_image_enabled": False})
                        require(not state.get("provider_fallback") and not state.get("generation_failed"), "Answer used provider fallback")
                        require(any(t.startswith(f"answer_generation:{settings.provider}:") for t in state.get("trace", [])), "No online answer generation")
                        answer = state.get("final_answer", "")
                        require(answer and re.search(r"\[E\d+\]", answer), "No cited final answer: " + json.dumps({"answer": answer, "trace": state.get("trace"), "evaluation": state.get("evaluation_result")}, ensure_ascii=False))
                        calls = [c for c in state.get("tool_calls", []) if c.get("tool_source") == "mcp"]
                        require(calls and all(c["status"] == "success" for c in calls), "Missing or failed MCP calls")
                        require(state.get("evaluation_result", {}).get("passed"), "Answer failed internal evaluation")
                        if session == "online-pdf":
                            context = state.get("final_context", "")
                            require("page 429" in context and "Luidia" in context, "Target PDF page was discarded from final context")
                            require("429" in answer and ("腕" in answer or "arms" in answer.lower()), "Answer did not explain the indexed morphology with page provenance")
                        if followup:
                            require(state.get("followup_detected"), "Conversation memory was not used")
                        if image_path:
                            require(any(t.startswith("vision:completed:") for t in state.get("trace", [])), "No online structured vision result")
                        return {"answer": answer, "mcp_tools": [c["tool_name"] for c in calls],
                                "provider_fallback": state.get("provider_fallback"), "warnings": state.get("warnings", []),
                                "trace": state.get("trace", [])}

                    check("workflow.text", lambda: conversation("海星有哪些外形特点？", "online-text"))
                    check("workflow.memory", lambda: conversation("它通常生活在哪里？", "online-text", followup=True))
                    check("workflow.pdf", lambda: conversation("请根据 PDF 第429页介绍 Luidia sarsii africana 的鉴别特征与栖息地，并引用来源。", "online-pdf"))
                    check("workflow.image", lambda: conversation("请分析图片中的生物形态，并用检索证据说明。", "online-image", str(copied_image)))
            finally:
                if workflow is not None:
                    workflow.close()
                if os.name == "nt" and server.poll() is None:
                    # Windows venv python.exe is a launcher with a child process.
                    # Terminate our own tree so the child releases log handles.
                    subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True, check=False)
                elif server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    report["passed"] = bool(report["checks"]) and all(row["passed"] for row in report["checks"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
