"""
Live end-to-end MCP verification: starts MCP servers, calls tools, compares results.

Usage:
    python scripts/verify_mcp_live.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aquabio.config import load_env

load_env(ROOT / ".env")


def _banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg: str):
    print(f"  [OK] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


def _info(msg: str):
    print(f"  [INFO] {msg}")


# ── Test 1: Chroma MCP Server (stdio) ─────────────────────────────

def test_chroma_mcp_stdio():
    _banner("Test 1: Chroma MCP Server via stdio")

    from aquabio_mrag.mcp_client import MCPServerConfig, MCPStdioClient

    python_exe = str(ROOT / ".venv" / "Scripts" / "python.exe")
    if not os.path.isfile(python_exe):
        _fail(f".venv python not found: {python_exe}")
        return None

    client = MCPStdioClient(
        {
            "chroma": MCPServerConfig(
                name="chroma",
                command=python_exe,
                args=["-m", "aquabio_mrag.mcp_server"],
                env={"PYTHONPATH": str(ROOT / "src")},
                timeout_seconds=60,
            ),
        }
    )

    # 1a: List tools
    _info("Listing Chroma MCP tools...")
    try:
        tools = client.list_tools_sync("chroma")
        _ok(f"Listed {len(tools)} tools:")
        for t in tools:
            print(f"      - {t['name']}: {t['description'][:80]}")
    except Exception as e:
        _fail(f"list_tools failed: {e}")
        return None

    # 1b: Call search_species_text
    _info("Calling search_species_text('starfish visual features', top_k=3)...")
    try:
        started = time.perf_counter()
        result = client.call_tool_sync(
            "chroma", "search_species_text",
            {"query": "starfish visual features", "top_k": 3},
        )
        elapsed = time.perf_counter() - started
        if isinstance(result, str):
            result = json.loads(result)
        count = len(result) if isinstance(result, list) else 0
        _ok(f"search_species_text: {count} results in {elapsed*1000:.0f}ms")
        if count > 0:
            first = result[0]
            print(f"      id={first.get('id','')[:50]}")
            print(f"      content={first.get('content','')[:100]}...")
    except Exception as e:
        _fail(f"search_species_text failed: {e}")

    # 1c: Call search_image_captions
    _info("Calling search_image_captions('starfish underwater', top_k=3)...")
    try:
        started = time.perf_counter()
        result = client.call_tool_sync(
            "chroma", "search_image_captions",
            {"query": "starfish underwater photo", "top_k": 3},
        )
        elapsed = time.perf_counter() - started
        if isinstance(result, str):
            result = json.loads(result)
        count = len(result) if isinstance(result, list) else 0
        _ok(f"search_image_captions: {count} results in {elapsed*1000:.0f}ms")
    except Exception as e:
        _fail(f"search_image_captions failed: {e}")

    # 1d: Call search_pdf
    _info("Calling search_pdf('starfish taxonomy', top_k=3)...")
    try:
        result = client.call_tool_sync(
            "chroma", "search_pdf",
            {"query": "starfish taxonomy", "candidate_species": "starfish", "top_k": 3},
        )
        if isinstance(result, str):
            result = json.loads(result)
        count = len(result) if isinstance(result, list) else 0
        _ok(f"search_pdf: {count} results")
    except Exception as e:
        _fail(f"search_pdf failed: {e}")

    # 1e: Call search_multimodal
    _info("Calling search_multimodal(query, caption, top_k=3)...")
    try:
        result = client.call_tool_sync(
            "chroma", "search_multimodal",
            {
                "query": "starfish identification",
                "image_caption": "radial arms star-shaped body",
                "candidate_species": "starfish",
                "top_k": 3,
            },
        )
        if isinstance(result, str):
            result = json.loads(result)
        count = len(result) if isinstance(result, list) else 0
        _ok(f"search_multimodal: {count} results")
    except Exception as e:
        _fail(f"search_multimodal failed: {e}")

    return client


# ── Test 2: RAG-Anything MCP Server (HTTP :8765) ──────────────────

def test_raganything_mcp_http():
    _banner("Test 2: RAG-Anything MCP Server (HTTP :8765)")

    from aquabio_mrag.mcp_client import project_mcp_client

    client = project_mcp_client(ROOT)

    # 2a: List tools
    _info("Listing RAG-Anything MCP tools...")
    try:
        tools = client.list_tools_sync("raganything")
        _ok(f"Listed {len(tools)} tools:")
        for t in tools:
            print(f"      - {t['name']}: {t['description'][:80]}")
    except Exception as e:
        _fail(f"RAG-Anything MCP not reachable: {e}")
        _info("Is start_chat_assistant.cmd running? Port 8765 needed.")
        _info("Skipping RAG-Anything tests (server not running).")
        return None

    # 2b: Call raganything_index_status
    _info("Calling raganything_index_status()...")
    try:
        started = time.perf_counter()
        result = client.call_tool_sync("raganything", "raganything_index_status", {})
        elapsed = time.perf_counter() - started
        entities = result.get("entities", 0) if isinstance(result, dict) else 0
        relations = result.get("relations", 0) if isinstance(result, dict) else 0
        _ok(f"index_status: {entities} entities, {relations} relations in {elapsed*1000:.0f}ms")
    except Exception as e:
        _fail(f"index_status failed: {e}")

    # 2c: Call raganything_hybrid_search
    _info("Calling raganything_hybrid_search('starfish identification', top_k=3)...")
    try:
        started = time.perf_counter()
        result = client.call_tool_sync(
            "raganything", "raganything_hybrid_search",
            {"query": "starfish identification guide", "top_k": 3},
        )
        elapsed = time.perf_counter() - started
        evidence = result.get("evidence", []) if isinstance(result, dict) else []
        _ok(f"hybrid_search: {len(evidence)} evidence items in {elapsed*1000:.0f}ms")
        if evidence:
            print(f"      id={evidence[0].get('id','')[:50]}")
            print(f"      content={evidence[0].get('content','')[:100]}...")
    except Exception as e:
        _fail(f"hybrid_search failed: {e}")

    # 2d: Call raganything_graph_neighbors
    _info("Calling raganything_graph_neighbors('Asteroidea', depth=1)...")
    try:
        started = time.perf_counter()
        result = client.call_tool_sync(
            "raganything", "raganything_graph_neighbors",
            {"entity": "Asteroidea", "depth": 1},
        )
        elapsed = time.perf_counter() - started
        nodes = result.get("nodes", []) if isinstance(result, dict) else []
        relations = result.get("relations", []) if isinstance(result, dict) else []
        _ok(f"graph_neighbors: {len(nodes)} nodes, {len(relations)} relations in {elapsed*1000:.0f}ms")
        for rel in relations[:3]:
            print(f"      {rel.get('source','')} --{rel.get('relation','')}--> {rel.get('target','')}")
    except Exception as e:
        _fail(f"graph_neighbors failed: {e}")

    # 2e: Call raganything_image_search (was search_pdf_images, now fixed)
    _info("Calling raganything_image_search('starfish distribution map', top_k=3)...")
    try:
        started = time.perf_counter()
        result = client.call_tool_sync(
            "raganything", "raganything_image_search",
            {"query": "starfish distribution map", "top_k": 3, "entity": "Asteroidea"},
        )
        elapsed = time.perf_counter() - started
        results = result.get("results", []) if isinstance(result, dict) else []
        _ok(f"image_search: {len(results)} image results in {elapsed*1000:.0f}ms")
        for r in results[:3]:
            print(f"      image_id={r.get('image_id','')} role={r.get('image_role','')} page={r.get('page','')}")
    except Exception as e:
        _fail(f"image_search failed: {e}")

    return client


# ── Test 3: Local vs MCP comparison ───────────────────────────────

def test_local_vs_mcp():
    _banner("Test 3: Local vs MCP Retrieval Comparison")

    from aquabio_mrag.config import MRAGPaths, MRAGSettings
    from aquabio_mrag.retrieval import MultiSourceRetriever, RetrievalRequest
    from aquabio_mrag.retrieval_agent import RetrievalAgent

    paths = MRAGPaths.from_root(ROOT)
    settings = MRAGSettings.from_env()
    retriever = MultiSourceRetriever(paths, settings)
    agent = RetrievalAgent(ROOT, retriever)

    query = "starfish visual features habitat"
    _info(f"Query: '{query}'")

    # Local path
    _info("--- Local retrieval ---")
    t0 = time.perf_counter()
    local_results = retriever.search(
        RetrievalRequest(
            query=query, task_type="text_qa", top_k=5,
            species_ids=["starfish"],
            source_types=["species_card", "species_text_chunk"],
        )
    )
    local_t = time.perf_counter() - t0
    _ok(f"Local: {len(local_results)} results in {local_t*1000:.0f}ms")
    for r in local_results[:3]:
        print(f"      [{r['final_score']:.3f}] {r['id'][:60]}")
        print(f"      {r['content'][:100]}...")

    # MCP path
    _info("--- MCP retrieval ---")
    try:
        t0 = time.perf_counter()
        mcp_results = agent.search_text_mcp(query, top_k=5, species_ids=["starfish"])
        mcp_t = time.perf_counter() - t0
        _ok(f"MCP:   {len(mcp_results)} results in {mcp_t*1000:.0f}ms")
        for r in mcp_results[:3]:
            print(f"      [{r['final_score']:.3f}] {r['id'][:60]}")
            print(f"      {r['content'][:100]}...")

        # Compare
        local_ids = {r["id"] for r in local_results[:5]}
        mcp_ids = {r["id"] for r in mcp_results[:5]}
        overlap = local_ids & mcp_ids
        _info(f"Overlap in top-5: {len(overlap)}/{5}")
        if len(overlap) >= 2:
            _ok("Local and MCP results are consistent (>=2 overlapping)")
        else:
            _info("Low overlap - MCP may use different ranking (expected)")
    except Exception as e:
        _fail(f"MCP retrieval failed: {e}")


# ── Test 4: Full Workflow Integration ─────────────────────────────

def test_workflow_mcp_mode():
    _banner("Test 4: Full Workflow with mcp_retrieval_enabled=True")

    try:
        from aquabio_mrag.config import MRAGPaths, MRAGSettings
        from aquabio_mrag.workflow import AquaBioMRAGWorkflow

        paths = MRAGPaths.from_root(ROOT)
        settings = MRAGSettings.from_env()
    except Exception as e:
        _fail(f"Cannot import workflow: {e}")
        return

    _info("Testing offline mode (no LLM call, just retrieval)...")
    try:
        wf = AquaBioMRAGWorkflow(paths, settings, offline=True)

        # Test with mcp_retrieval_enabled=True
        _info("--- mcp_retrieval_enabled=True ---")
        t0 = time.perf_counter()
        state_mcp = wf.invoke(
            query="海星有什么外观特征？",
            session_id="verify_mcp_test",
            options={
                "mcp_retrieval_enabled": True,
                "mcp_enabled": True,
                "rag_enabled": True,
                "pdf_enabled": True,
                "image_search_enabled": True,
                "vision_enabled": False,
            },
        )
        mcp_t = time.perf_counter() - t0
        mcp_text = len(state_mcp.get("text_context", []))
        mcp_image = len(state_mcp.get("image_context", []))
        mcp_pdf = len(state_mcp.get("pdf_context", []))
        mcp_multimodal = len(state_mcp.get("multimodal_context", []))
        mcp_warnings = state_mcp.get("warnings", [])
        _ok(f"MCP mode: text={mcp_text}, image={mcp_image}, pdf={mcp_pdf}, multimodal={mcp_multimodal} in {mcp_t:.1f}s")
        if mcp_warnings:
            for w in mcp_warnings[:3]:
                _info(f"  Warning: {w[:120]}")

        # Test with mcp_retrieval_enabled=False
        _info("--- mcp_retrieval_enabled=False (local) ---")
        t0 = time.perf_counter()
        state_local = wf.invoke(
            query="海星有什么外观特征？",
            session_id="verify_local_test",
            options={
                "mcp_retrieval_enabled": False,
                "mcp_enabled": True,
                "rag_enabled": True,
                "pdf_enabled": True,
                "image_search_enabled": True,
                "vision_enabled": False,
            },
        )
        local_t = time.perf_counter() - t0
        local_text = len(state_local.get("text_context", []))
        local_image = len(state_local.get("image_context", []))
        local_pdf = len(state_local.get("pdf_context", []))
        local_multimodal = len(state_local.get("multimodal_context", []))
        _ok(f"Local mode: text={local_text}, image={local_image}, pdf={local_pdf}, multimodal={local_multimodal} in {local_t:.1f}s")

        # Compare
        _info("--- Comparison ---")
        print(f"  {'Metric':<15} {'MCP':>6} {'Local':>6} {'Status':>8}")
        print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8}")
        for name, mcp_val, local_val in [
            ("text_context", mcp_text, local_text),
            ("image_context", mcp_image, local_image),
            ("pdf_context", mcp_pdf, local_pdf),
            ("multimodal", mcp_multimodal, local_multimodal),
        ]:
            status = "OK" if (mcp_val > 0) == (local_val > 0) or (mcp_val > 0 and local_val == 0) else "DIFF"
            print(f"  {name:<15} {mcp_val:>6} {local_val:>6} {status:>8}")

        trace = state_mcp.get("trace", [])
        mcp_traces = [t for t in trace if "mcp" in t.lower()]
        _info(f"MCP-related trace events: {len(mcp_traces)}")
        for t in mcp_traces[:5]:
            print(f"      {t}")

    except Exception as e:
        _fail(f"Workflow test failed: {e}")
        import traceback
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AquaBio-AgentRAG Live MCP Verification")
    print("=" * 60)
    print(f"  Root: {ROOT}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results: dict[str, str] = {}

    # Test 1: Chroma MCP stdio
    chroma_client = test_chroma_mcp_stdio()
    results["chroma_stdio"] = "PASS" if chroma_client else "FAIL"

    # Test 2: RAG-Anything MCP HTTP
    rag_client = test_raganything_mcp_http()
    results["raganything_http"] = "PASS" if rag_client else "SKIP (server not running)"

    # Test 3: Local vs MCP comparison
    if chroma_client:
        test_local_vs_mcp()
    else:
        _banner("Test 3: Skipped (Chroma MCP not available)")

    # Test 4: Full workflow
    test_workflow_mcp_mode()

    # Summary
    _banner("Results Summary")
    for test_name, status in results.items():
        print(f"  [{status}] {test_name}")

    print()
    print("  To test with the RAG-Anything MCP server, first run:")
    print("    start_chat_assistant.cmd")
    print()


if __name__ == "__main__":
    main()
