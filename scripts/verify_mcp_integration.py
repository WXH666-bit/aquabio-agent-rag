"""
Verify MCP integration: check all 14 tools are defined, accessible, and
correctly called from LangGraph nodes.

Usage:
    python scripts/verify_mcp_integration.py              # full check
    python scripts/verify_mcp_integration.py --quick      # definitions only
    python scripts/verify_mcp_integration.py --live       # live tool calls
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


# ── Tool registry: all 14 tools across 2 servers ────────────────────

CHROMA_MCP_TOOLS = {
    "search_species_text": {
        "params": {"query": "starfish visual features", "top_k": 3},
        "called_by": "retrieval_node (text_retriever, mcp_retrieval=True)",
        "server": "chroma",
        "transport": "stdio",
    },
    "search_image_captions": {
        "params": {"query": "starfish underwater photo", "top_k": 3},
        "called_by": "retrieval_node (image_retriever, mcp_retrieval=True)",
        "server": "chroma",
        "transport": "stdio",
    },
    "search_pdf_entity_images": {
        "params": {"query": "starfish distribution map", "top_k": 3, "entity": "Asteroidea"},
        "called_by": "retrieval_node (pdf_image_retriever, mcp_retrieval=True)",
        "server": "chroma",
        "transport": "stdio",
    },
    "get_entity_sample_images": {
        "params": {"entity": "Asteroidea", "top_k": 3},
        "called_by": "retrieval_node (pdf_image_retriever, mcp_retrieval=True)",
        "server": "chroma",
        "transport": "stdio",
    },
    "search_multimodal": {
        "params": {
            "query": "starfish identification",
            "image_caption": "radial arms star-shaped body",
            "candidate_species": "starfish",
            "top_k": 3,
        },
        "called_by": "retrieval_node (multimodal_retriever, mcp_retrieval=True)",
        "server": "chroma",
        "transport": "stdio",
    },
    "search_pdf": {
        "params": {"query": "starfish taxonomy", "candidate_species": "starfish", "top_k": 3},
        "called_by": "retrieval_node (pdf_retriever, mcp_retrieval=True)",
        "server": "chroma",
        "transport": "stdio",
    },
    "generate_image_caption": {
        "params": {"image_path": ""},  # needs a real image
        "called_by": "vision_node (via VLM, can also be called via MCP)",
        "server": "chroma",
        "transport": "stdio",
    },
    "get_source_detail": {
        "params": {"doc_id": "card_starfish"},
        "called_by": "retrieval_node (source_trace tasks)",
        "server": "chroma",
        "transport": "stdio",
    },
}

RAGANYTHING_MCP_TOOLS = {
    "raganything_index_status": {
        "params": {},
        "called_by": "warmup / system status endpoint",
        "server": "raganything",
        "transport": "streamable-http (port 8765)",
    },
    "raganything_graph_neighbors": {
        "params": {"entity": "Asteroidea", "depth": 1},
        "called_by": "retrieval_node → RetrievalAgent.search_pdf() → MCP",
        "server": "raganything",
        "transport": "streamable-http (port 8765)",
    },
    "raganything_hybrid_search": {
        "params": {"query": "starfish identification guide", "top_k": 3},
        "called_by": "retrieval_node → RetrievalAgent.search_pdf() → MCP",
        "server": "raganything",
        "transport": "streamable-http (port 8765)",
    },
    "raganything_image_search": {
        "params": {"query": "starfish distribution map", "top_k": 3, "entity": "Asteroidea"},
        "called_by": "retrieval_node → RetrievalAgent.search_pdf_images() → MCP",
        "server": "raganything",
        "transport": "streamable-http (port 8765)",
    },
    "raganything_entity_images": {
        "params": {"entity": "Asteroidea", "top_k": 3},
        "called_by": "retrieval_node → entity_image_records() local path",
        "server": "raganything",
        "transport": "streamable-http (port 8765)",
    },
    "raganything_source_detail": {
        "params": {"doc_id": "fao_species_catalogue"},
        "called_by": "retrieval_node → RetrievalAgent.search_pdf() → MCP",
        "server": "raganything",
        "transport": "streamable-http (port 8765)",
    },
}

ALL_TOOLS = {**CHROMA_MCP_TOOLS, **RAGANYTHING_MCP_TOOLS}


def check_definitions() -> dict:
    """Check that all 14 MCP tools exist in source code."""
    import ast

    results = {"total": 0, "found": 0, "missing": []}

    def _collect_tools(tree: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    attr = getattr(dec.func, "attr", "")
                    if attr == "tool":
                        names.add(node.name)
                        break
        return names

    # Check Chroma MCP server
    chroma_mcp = ROOT / "src" / "aquabio_mrag" / "mcp_server.py"
    tree = ast.parse(chroma_mcp.read_text(encoding="utf-8"))
    defined = _collect_tools(tree)

    # Check RAG-Anything MCP server
    rag_mcp = ROOT / "src" / "aquabio_raganything" / "mcp_server.py"
    tree2 = ast.parse(rag_mcp.read_text(encoding="utf-8"))
    defined |= _collect_tools(tree2)

    for tool_name in ALL_TOOLS:
        results["total"] += 1
        if tool_name in defined:
            results["found"] += 1
        else:
            results["missing"].append(tool_name)

    return results


def check_call_sites() -> dict:
    """Check which MCP tools are actually called from retrieval_agent.py."""
    import re

    agent_file = ROOT / "src" / "aquabio_mrag" / "retrieval_agent.py"
    content = agent_file.read_text(encoding="utf-8")

    results = {}
    for tool_name in ALL_TOOLS:
        # Look for call_tool_sync("server", "tool_name", ...) patterns
        pattern = rf'call_tool_sync\s*\(\s*"[^"]*"\s*,\s*"{re.escape(tool_name)}"\s*,'
        called = bool(re.search(pattern, content))

        # Also check for wrong tool name patterns (the bugs we fixed)
        wrong_names = []
        if tool_name == "raganything_image_search":
            if re.search(r'call_tool_sync\s*\(\s*"[^"]*"\s*,\s*"search_pdf_images"\s*,', content):
                wrong_names.append("search_pdf_images (OLD BUG — should be fixed)")
        if tool_name == "raganything_source_detail":
            if re.search(r'call_tool_sync\s*\(\s*"[^"]*"\s*,\s*"get_source_detail"\s*,', content):
                wrong_names.append("get_source_detail (OLD BUG — should be fixed)")

        results[tool_name] = {
            "called_via_mcp": called,
            "wrong_names_found": wrong_names,
            "status": "[OK] called" if called else ("[WARN] not called via MCP" if not wrong_names else "[BUG] had wrong name"),
        }
    return results


def try_list_tools(server: str) -> dict:
    """Try to list tools from a running MCP server."""
    from aquabio_mrag.mcp_client import project_mcp_client

    client = project_mcp_client(ROOT)
    try:
        tools = client.list_tools_sync(server)
        return {
            "server": server,
            "status": "available",
            "tool_count": len(tools),
            "tools": [t["name"] for t in tools],
        }
    except Exception as error:
        return {"server": server, "status": "unavailable", "error": str(error)}


def try_call_tools() -> dict:
    """Try a subset of tools with live MCP calls."""
    from aquabio_mrag.mcp_client import project_mcp_client

    client = project_mcp_client(ROOT)
    results = {}

    # Test Chroma tools
    test_tools = [
        ("chroma", "search_species_text", {"query": "starfish", "top_k": 2}),
        ("chroma", "search_image_captions", {"query": "starfish", "top_k": 2}),
        ("raganything", "raganything_index_status", {}),
    ]

    for server, tool, args in test_tools:
        key = f"{server}.{tool}"
        try:
            started = time.perf_counter()
            result = client.call_tool_sync(server, tool, args)
            elapsed = time.perf_counter() - started
            count = (
                len(result)
                if isinstance(result, list)
                else len(result.get("results", result.get("entities", [])))
                if isinstance(result, dict)
                else 0
            )
            results[key] = {
                "status": "ok",
                "elapsed_ms": int(elapsed * 1000),
                "result_count": count,
            }
        except Exception as error:
            results[key] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    return results


def verify_workflow_routing() -> dict:
    """Verify that workflow.py correctly routes between local and MCP modes."""
    wf_file = ROOT / "src" / "aquabio_mrag" / "workflow.py"
    content = wf_file.read_text(encoding="utf-8")

    checks = {
        "mcp_retrieval_enabled_in_runtime_options": (
            '"mcp_retrieval_enabled"' in content
            and "mcp_retrieval_enabled" in content
        ),
        "text_retriever_mcp_path": "search_text_mcp" in content,
        "image_retriever_mcp_path": "search_image_mcp" in content,
        "multimodal_retriever_mcp_path": "search_multimodal_mcp" in content,
        "pdf_retriever_mcp_path": "search_pdf_chunks_mcp" in content,
        "pdf_retriever_graph_mcp": 'raganything_hybrid_search' in content,
    }
    return {k: "[OK]" if v else "[MISS] MISSING" for k, v in checks.items()}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify MCP integration")
    parser.add_argument("--quick", action="store_true", help="Definitions only")
    parser.add_argument("--live", action="store_true", help="Live tool calls")
    args = parser.parse_args()

    width = 72
    print("=" * width)
    print("  AquaBio-AgentRAG MCP Integration Verification")
    print("=" * width)

    # ── Section 1: Tool definitions ────────────────────────────────
    print("\n[1] MCP Tool Definitions (source code scan)")
    print("-" * 48)
    defs = check_definitions()
    print(f"  Total tools expected: {defs['total']}")
    print(f"  Found in source:      {defs['found']}")
    if defs["missing"]:
        print(f"  [MISS] Missing: {defs['missing']}")
    else:
        print("  [OK] All 14 tools defined in MCP servers")

    # ── Section 2: Call sites ──────────────────────────────────────
    print("\n[2] MCP Tool Call Sites (retrieval_agent.py scan)")
    print("-" * 48)
    calls = check_call_sites()
    for tool_name, info in calls.items():
        server = ALL_TOOLS[tool_name]["server"]
        print(f"  {info['status']:12s} {server:12s} → {tool_name}")
        if info.get("wrong_names_found"):
            for wn in info["wrong_names_found"]:
                print(f"                    [BUG] {wn}")

    # ── Section 3: Bug fixes verification ──────────────────────────
    print("\n[3] Bug Fix Verification")
    print("-" * 48)
    agent_file = ROOT / "src" / "aquabio_mrag" / "retrieval_agent.py"
    agent_code = agent_file.read_text(encoding="utf-8")

    bugs = {
        "search_pdf_images → raganything_image_search": (
            '"search_pdf_images"' not in agent_code
        ),
        "get_source_detail → raganything_source_detail": (
            '"get_source_detail"' not in agent_code
        ),
        "raganything_image_search present": (
            '"raganything_image_search"' in agent_code
        ),
        "raganything_source_detail present": (
            '"raganything_source_detail"' in agent_code
        ),
    }
    for desc, ok in bugs.items():
        print(f"  {'[OK]' if ok else '[MISS]'} {desc}")

    # ── Section 4: Workflow routing ────────────────────────────────
    print("\n[4] LangGraph Workflow MCP Routing")
    print("-" * 48)
    routing = verify_workflow_routing()
    for check, status in routing.items():
        print(f"  {status} {check}")

    # ── Section 5: Node → Tool mapping ─────────────────────────────
    print("\n[5] LangGraph Node → MCP Tool Call Map")
    print("-" * 48)

    node_tool_map = {
        "session_init": [],
        "memory_load": [],
        "followup_resolver": [],
        "router": [],
        "rewrite": [],
        "source_selection": [],
        "react_tool_plan": [],
        "vision": ["generate_image_caption (optional, via direct LLM by default)"],
        "retrieval_node (text)": [
            "search_species_text (mcp_retrieval=True)",
            "MultiSourceRetriever.search() (default local)",
        ],
        "retrieval_node (image)": [
            "search_image_captions (mcp_retrieval=True)",
            "MultiSourceRetriever.image_search() (default local)",
        ],
        "retrieval_node (pdf_image)": [
            "raganything_image_search (MCP, fixed from search_pdf_images)",
            "query_pdf_images() (local fallback)",
            "raganything_entity_images (optional)",
        ],
        "retrieval_node (multimodal)": [
            "search_multimodal (mcp_retrieval=True)",
            "MultiSourceRetriever.multimodal_search() (default local)",
        ],
        "retrieval_node (pdf)": [
            "search_pdf (mcp_retrieval=True for Chroma portion)",
            "raganything_hybrid_search (MCP, always for online mode)",
            "raganything_graph_neighbors (MCP, with graph_entities)",
            "raganything_source_detail (MCP, fixed from get_source_detail)",
            "BookNativeBM25.search() (local BM25)",
            "ChromaMRAGStore.query() (local Chroma)",
        ],
        "answer_node": [],
        "finalize": [],
        "memory_save": [],
    }

    for node, tools in node_tool_map.items():
        if tools:
            print(f"  {node}:")
            for tool in tools:
                print(f"    → {tool}")
        else:
            print(f"  {node}: (no MCP tools)")

    if args.quick:
        return

    # ── Section 6: Live tool listing ───────────────────────────────
    print("\n[6] Live MCP Server Status")
    print("-" * 48)
    for server in ("chroma", "raganything"):
        status = try_list_tools(server)
        print(f"  {server}: {status['status']}")
        if status["status"] == "available":
            print(f"    Tools: {status['tools']}")
        else:
            print(f"    Error: {status.get('error', 'unknown')}")

    if args.live:
        print("\n[7] Live Tool Call Tests")
        print("-" * 48)
        live_results = try_call_tools()
        for key, result in live_results.items():
            if result["status"] == "ok":
                print(
                    f"  [OK] {key}: {result['result_count']} results"
                    f" ({result['elapsed_ms']}ms)"
                )
            else:
                print(f"  [MISS] {key}: {result['error']}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * width)
    print("  Summary")
    print("=" * width)
    print(f"  Total MCP tools:              {defs['total']}")
    called_count = sum(1 for v in calls.values() if v["called_via_mcp"])
    print(f"  Called via MCP in code:       {called_count}")
    print(f"  Available locally (no MCP):   {defs['total'] - called_count}")
    print(f"  Bug fixes applied:            {sum(1 for v in bugs.values() if v)}/{len(bugs)}")
    routing_ok = sum(1 for v in routing.values() if v == "[OK]")
    print(f"  Workflow routing checks:      {routing_ok}/{len(routing)}")

    any_issue = defs["missing"] or any(
        v["wrong_names_found"] for v in calls.values()
    ) or routing_ok < len(routing)

    if any_issue:
        print("\n  [WARN]  Some issues remain — see details above.")
    else:
        print("\n  [OK] All checks passed. MCP integration is complete.")
    print()


if __name__ == "__main__":
    main()
