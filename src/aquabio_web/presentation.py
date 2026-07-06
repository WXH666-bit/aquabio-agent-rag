from __future__ import annotations

from collections import Counter
from typing import Any


REACT_PHASES = (
    (
        "Thought",
        "理解与规划",
        {
            "session_init",
            "memory_load",
            "followup_resolver",
            "router",
            "query_rewrite",
            "source_selection",
            "react_tool_plan",
            "web_plan",
        },
    ),
    (
        "Action",
        "调用模型与工具",
        {
            "vision",
            "retrieval",
            "retry",
        },
    ),
    (
        "Observation",
        "整理检索证据",
        {
            "rerank",
            "context_builder",
        },
    ),
    (
        "Answer",
        "生成、评估与保存",
        {
            "answer_generation",
            "response_guard",
            "evaluation",
            "finalize",
            "memory_save",
            "api",
        },
    ),
)

TOOL_LABELS = {
    "vlm_caption": "视觉理解",
    "text_retriever": "文本向量检索",
    "image_retriever": "图片检索",
    "pdf_image_retriever": "PDF 图片 MCP",
    "multimodal_retriever": "图文联合检索",
    "pdf_retriever": "PDF/图谱混合检索",
}

SOURCE_LABELS = {
    "chroma": "Chroma",
    "web_research": "Web API",
    "image_retrieval": "本地图片",
    "pdf_hybrid": "PDF Hybrid",
    "lightrag_graph": "LightRAG",
    "raganything_pdf_image": "PDF 图片",
    "wikipedia_api": "Wikipedia",
    "wikimedia_commons": "Wikimedia",
}


def trace_phase_summary(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_node: dict[str, list[dict[str, Any]]] = {}
    for event in trace:
        by_node.setdefault(str(event.get("node", "")), []).append(event)
    result = []
    for phase, label, nodes in REACT_PHASES:
        events = [
            event
            for node in nodes
            for event in by_node.get(node, [])
        ]
        result.append(
            {
                "phase": phase,
                "label": label,
                "status": "complete" if events else "pending",
                "event_count": len(events),
                "details": [
                    str(event.get("detail", ""))
                    for event in events[-3:]
                    if event.get("detail")
                ],
            }
        )
    return result


def selected_tools(trace: list[dict[str, Any]]) -> list[str]:
    tools: list[str] = []
    for event in trace:
        if event.get("node") not in {
            "source_selection",
            "react_tool_plan",
        }:
            continue
        detail = str(event.get("detail", ""))
        if "tools=" in detail:
            detail = detail.split("tools=", 1)[1]
        elif ":" in detail:
            detail = detail.split(":", 1)[1]
        for value in detail.split(","):
            name = value.strip()
            if name in TOOL_LABELS and name not in tools:
                tools.append(name)
    return tools


def mcp_activity(
    trace: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    activity = {
        "enabled": False,
        "pdf_images": 0,
        "graph_retrieval": False,
        "details": [],
    }
    for event in trace:
        detail = str(event.get("detail", ""))
        if "mcp" not in detail.casefold() and event.get("node") != "retrieval":
            continue
        if "mcp_pdf_images=" in detail:
            try:
                count = int(detail.split("mcp_pdf_images=", 1)[1].split(",", 1)[0])
            except ValueError:
                count = 0
            activity["pdf_images"] = max(activity["pdf_images"], count)
            activity["enabled"] = True
        if "mcp_graph=" in detail:
            try:
                count = int(detail.split("mcp_graph=", 1)[1].split(",", 1)[0])
            except ValueError:
                count = 0
            activity["graph_retrieval"] = (
                activity["graph_retrieval"] or count > 0
            )
            activity["enabled"] = activity["enabled"] or count > 0
        if "pdf_retriever" in detail or "lightrag" in detail.casefold():
            activity["graph_retrieval"] = True
            activity["enabled"] = True
        activity["details"].append(detail)
    for item in evidence or []:
        source = str(
            item.get("source_system")
            or item.get("metadata", {}).get("retrieval_source")
            or ""
        )
        if source == "lightrag_graph":
            activity["graph_retrieval"] = True
            activity["enabled"] = True
        if "raganything_pdf_image" in source:
            activity["pdf_images"] += 1
            activity["enabled"] = True
    return activity


def evidence_source_counts(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for item in evidence:
        source = str(
            item.get("source_system")
            or item.get("metadata", {}).get("retrieval_source")
            or item.get("source_type")
            or "unknown"
        )
        counts[SOURCE_LABELS.get(source, source)] += 1
    return [
        {"source": source, "count": count}
        for source, count in counts.most_common()
    ]


def architecture_manifest(status: dict[str, Any]) -> list[dict[str, str]]:
    graph = status.get("graph", {})
    warmup = status.get("warmup", {})
    return [
        {
            "layer": "UI",
            "component": "Streamlit",
            "state": "ready",
            "detail": "多会话、附件、任务进度与 Agent 驾驶舱",
        },
        {
            "layer": "API",
            "component": status.get("backend", "FastAPI"),
            "state": "ready" if status.get("status") == "running" else "error",
            "detail": "会话、任务、上传、反馈与资源接口",
        },
        {
            "layer": "Agent",
            "component": status.get("agent", "LangGraph"),
            "state": warmup.get("status", "unknown"),
            "detail": "路由、ReAct、检索子图、答案子图与 SQLite checkpoint",
        },
        {
            "layer": "Retrieval",
            "component": "Chroma + BGE-M3",
            "state": warmup.get("status", "unknown"),
            "detail": status.get("embedding_model", ""),
        },
        {
            "layer": "Graph",
            "component": "LightRAG + NetworkX",
            "state": "ready" if graph.get("storage_valid") else "warning",
            "detail": (
                f"{graph.get('entities', 0)} entities / "
                f"{graph.get('relations', 0)} relations"
            ),
        },
        {
            "layer": "Tools",
            "component": "MCP",
            "state": "ready",
            "detail": ", ".join(status.get("mcp_servers", [])),
        },
    ]
