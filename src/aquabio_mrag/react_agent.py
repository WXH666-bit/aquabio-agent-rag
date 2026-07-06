from __future__ import annotations

import json
from typing import Any

from aquabio.openrouter import OpenRouterClient


TOOL_DESCRIPTIONS = {
    "vlm_caption": "Analyze the supplied underwater image.",
    "text_retriever": "Search species cards and long-form species text.",
    "image_retriever": "Search image captions.",
    "pdf_image_retriever": (
        "Search images extracted from indexed PDFs through the "
        "RAG-Anything MCP server."
    ),
    "multimodal_retriever": "Search linked image and text evidence.",
    "pdf_retriever": "Search Chroma, book BM25 and LightRAG graph evidence.",
}


class ControlledReActPlanner:
    def __init__(self, llm: OpenRouterClient, max_steps: int = 4):
        self.llm = llm
        self.max_steps = max_steps

    def plan(
        self,
        query: str,
        allowed_tools: list[str],
        observations: list[str],
        step: int,
    ) -> list[str]:
        if step >= self.max_steps or not allowed_tools:
            return []
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
            for name in allowed_tools
            if name in TOOL_DESCRIPTIONS
        ]
        calls = self.llm.select_tools(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a controlled retrieval planner. Select only "
                        "tools needed for the current question. You may select "
                        "multiple tools. Do not answer the question."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "step": step + 1,
                            "observations": observations,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools,
        )
        selected = []
        for call in calls:
            name = call.get("name", "")
            if name in allowed_tools and name not in selected:
                selected.append(name)
        return selected
