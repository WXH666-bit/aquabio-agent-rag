from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .image_tools import analyze_quality, create_enhancements
from .openrouter import OpenRouterClient
from .retriever import HybridRetriever


class AquaBioAgent:
    def __init__(self, project_root: str | Path = ".", offline: bool = False):
        self.root = Path(project_root)
        self.settings = Settings.from_env()
        self.client = OpenRouterClient(self.settings, offline=offline)
        self.retriever = HybridRetriever(
            self.root / "data/knowledge",
            self.root / "data/index",
            self.root / "data/vector_db",
        )

    def _offline_answer(self, query: str, contexts: list[dict], warning: str) -> str:
        excerpts = []
        for item in contexts[:4]:
            source = item.get("source") or item.get("dataset_name") or item.get("class_name", "knowledge")
            page = f", p.{item['page']}" if item.get("page") else ""
            excerpts.append(f"- [{source}{page}] {item.get('content', '')[:260]}")
        body = "\n".join(excerpts) or "- 本地知识库尚无可用内容。"
        return f"{warning}\n\n与“{query}”最相关的本地证据：\n{body}"

    def run(self, query: str, image_path: str | None = None) -> dict:
        state = {
            "query": query,
            "route": "multimodal_qa" if image_path else "document_qa",
            "image_path": image_path,
            "retrieval": [],
            "image_quality": None,
            "enhancements": [],
            "vision_analysis": None,
            "tool_trace": [],
            "warnings": [],
            "answer": "",
        }
        retrieval_query = query
        if image_path:
            state["image_quality"] = analyze_quality(image_path)
            state["tool_trace"].append("image_quality")
            state["enhancements"] = create_enhancements(
                image_path, self.root / "data/outputs/enhanced"
            )
            state["tool_trace"].append("image_enhancement")
            if self.client.enabled:
                state["vision_analysis"] = self.client.analyze_image(
                    image_path,
                    "你是水下生物图像分析助手。给出候选类别、可见特征、退化问题和不确定性。"
                    "不要声称输出了可靠边界框，也不要把候选识别说成专用检测器结果。",
                )
                state["tool_trace"].append("openrouter_vision")
                retrieval_query += " " + " ".join(state["vision_analysis"]["possible_species"])
            else:
                state["warnings"].append("未配置 OpenRouter key，已跳过 VLM 图像理解。")

        state["retrieval"] = self.retriever.search(retrieval_query, top_k=7)
        state["tool_trace"].append("hybrid_retrieval")

        if not self.client.enabled:
            state["answer"] = self._offline_answer(
                query,
                state["retrieval"],
                "当前为离线模式，下面只展示检索证据，不生成模型结论。",
            )
            return state

        evidence = []
        for item in state["retrieval"]:
            source = item.get("source") or item.get("dataset_name") or item.get("class_name")
            evidence.append(
                {
                    "source": source,
                    "page": item.get("page"),
                    "source_type": item.get("source_type"),
                    "content": item.get("content"),
                }
            )
        prompt = {
            "user_query": query,
            "image_quality": state["image_quality"],
            "vision_analysis": state["vision_analysis"],
            "retrieved_evidence": evidence,
            "rules": [
                "只根据工具事实和检索证据回答。",
                "VLM 结果称为候选识别，不称为目标检测结果。",
                "不能因视觉质量提高就断言检测性能提高。",
                "引用 PDF 时写出文件名和页码。",
                "信息不足时明确说明不确定性。",
            ],
        }
        state["answer"] = self.client.chat(
            [
                {
                    "role": "system",
                    "content": "你是 AquaBio-AgentRAG，请用简洁中文回答水下生物、图像增强和 PDF 问答问题。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ]
        )
        state["tool_trace"].append("answer_generation")
        if state["retrieval"] and not any(
            str(item.get("source", "")) in state["answer"] for item in state["retrieval"] if item.get("source")
        ):
            state["warnings"].append("生成答案可能未显式标注全部检索来源，请结合证据面板检查。")
        return state
