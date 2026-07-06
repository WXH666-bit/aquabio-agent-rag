from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from aquabio.config import GeminiSettings, Settings
from aquabio.gemini import GeminiVisionClient
from aquabio.openrouter import OpenRouterClient

from .config import MRAGPaths, MRAGSettings
from .conversation import ConversationStore
from .io_utils import read_jsonl
from .models import AquaBioState, EvaluationResult, RouteDecision
from .retrieval import MultiSourceRetriever, RetrievalRequest
from .retrieval_agent import RetrievalAgent
from aquabio_raganything.image_rag import (
    DISTRIBUTION_ROLE,
    SPECIMEN_ROLE,
    asks_for_reference_images,
    requested_page,
    requested_image_roles,
    role_matches,
)
from aquabio_web.web_knowledge import (
    asks_for_related_entity_image,
    needs_web_research,
    plan_web_research,
    search_wikipedia,
)
from .react_agent import ControlledReActPlanner


COMPARISON_MARKERS = ("区别", "区分", "比较", "不同", "versus", " vs ")
SOURCE_MARKERS = ("依据", "来源", "哪篇", "出处", "source")
PDF_MARKERS = ("pdf", "文档", "手册", "报告", "论文", "fao", "noaa", "iucn")
DETECTION_MARKERS = (
    "检测",
    "目标检测",
    "识别目标",
    "检测目标",
    "增强检测",
    "水下检测",
    "物体检测",
    "detect",
    "detection",
    "object detection",
    "find objects",
    "locate",
    "bbox",
    "bounding box",
)
PDF_GRAPH_IMAGE_MARKERS = (
    "书中图片",
    "书里的图片",
    "书里图片",
    "书中图",
    "页码",
    "所在页",
    "分类关系",
    "分类层级",
    "pdf 证据",
    "PDF 证据",
    "图谱",
    "Field Guide",
    "field guide",
)
FOLLOWUP_MARKERS = (
    "刚才",
    "上次",
    "前面",
    "之前",
    "这个生物",
    "那个生物",
    "该生物",
    "它",
    "它的",
    "她",
    "她的",
    "其",
    "刚才",
    "上次",
    "前面",
    "之前",
    "这个生物",
    "那个生物",
    "该生物",
    "它",
    "它的",
    "其",
)
SHORT_MARKERS = ("简短", "简洁", "只回答", "不要回答其他", "仅回答")
COLOR_MARKERS = ("颜色", "什么颜色", "常见颜色", "色彩")
COLOR_WORDS = (
    "橙色",
    "红色",
    "褐色",
    "灰色",
    "棕色",
    "粉色",
    "黄色",
    "白色",
    "黑色",
    "蓝色",
    "绿色",
    "紫色",
    "透明",
    "半透明",
)


def _tool_flow_label(task: str, tools: list[str]) -> str:
    tool_set = set(tools)
    if "vlm_caption" in tool_set and "pdf_image_retriever" in tool_set:
        return "multimodal_pdf_image_graph"
    if "vlm_caption" in tool_set:
        return "multimodal_identification"
    if "pdf_image_retriever" in tool_set and "pdf_retriever" in tool_set:
        return "pdf_image_graph_evidence"
    if "pdf_image_retriever" in tool_set:
        return "pdf_image_lookup"
    if task == "comparison_qa" and "pdf_retriever" in tool_set:
        return "comparison_text_pdf_graph"
    if "pdf_retriever" in tool_set:
        return "pdf_graph_evidence"
    if "image_retriever" in tool_set:
        return "image_caption_lookup"
    if "text_retriever" in tool_set:
        return "text_basic_rag"
    return "no_retrieval"


def _append_trace(state: AquaBioState, value: str) -> list[str]:
    return [*state.get("trace", []), value]


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            result.append(row)
    return result


class AquaBioMRAGWorkflow:
    def __init__(
        self,
        paths: MRAGPaths,
        settings: MRAGSettings,
        offline: bool = False,
    ):
        self.paths = paths
        self.settings = settings
        self.retriever = MultiSourceRetriever(paths, settings)
        self.retrieval_agent = RetrievalAgent(paths.root, self.retriever)
        self.llm = OpenRouterClient(Settings.from_env(), offline=offline)
        self.react_planner = ControlledReActPlanner(self.llm)
        self.gemini = GeminiVisionClient(
            GeminiSettings.from_env(), offline=offline
        )
        self.offline = offline
        self.conversations = ConversationStore(paths.sessions_dir)
        self.species = json.loads(
            paths.species_list.read_text(encoding="utf-8")
        )
        self.species_by_id = {
            item["species_id"]: item for item in self.species
        }
        self.species_aliases: dict[str, str] = {}
        for item in self.species:
            aliases = {
                item["species_id"],
                item.get("english_name", ""),
                item.get("chinese_name", ""),
                item.get("scientific_name", ""),
                *item.get("keywords", []),
            }
            for alias in aliases:
                normalized = str(alias).strip().casefold()
                if normalized:
                    self.species_aliases[normalized] = item["species_id"]
        self.known_images = {
            str((paths.root / row["image_path"]).resolve()).lower(): row
            for row in read_jsonl(paths.knowledge_dir / "image_docs.jsonl")
            if row.get("image_path")
        }
        self.graph = self._build()

    def _normalize_species_ids(self, values: list[str]) -> list[str]:
        result = []
        for raw in values:
            value = str(raw).strip().casefold()
            if not value:
                continue
            species_id = self.species_aliases.get(value)
            if species_id is None:
                for alias, candidate in self.species_aliases.items():
                    if len(alias) >= 2 and (
                        alias in value or value in alias
                    ):
                        species_id = candidate
                        break
            if species_id and species_id not in result:
                result.append(species_id)
        return result

    def session_init_node(self, state: AquaBioState) -> dict:
        session_id = self.conversations.normalize_session_id(
            state.get("session_id", "default")
        )
        return {
            "session_id": session_id,
            "session_initialized": True,
            "trace": _append_trace(state, f"session_init:{session_id}"),
        }

    def memory_load_node(self, state: AquaBioState) -> dict:
        if not state.get("runtime_options", {}).get(
            "memory_enabled", True
        ):
            return {
                "conversation_history": [],
                "memory_summary": {},
                "trace": _append_trace(state, "memory_load:disabled"),
            }
        session = self.conversations.load(
            state.get("session_id", "default")
        )
        history = [
            {
                "turn_index": turn.get("turn_index"),
                "user_query": turn.get("user_query", ""),
                "resolved_query": turn.get("resolved_query", ""),
                "assistant_answer": turn.get("assistant_answer", ""),
                "species_ids": turn.get("species_ids", []),
                "image_path": turn.get("image_path", ""),
            }
            for turn in session.get("turns", [])[-8:]
        ]
        return {
            "conversation_history": history,
            "memory_summary": session.get("summary", {}),
            "trace": _append_trace(
                state, f"memory_load:{len(history)}"
            ),
        }

    def followup_resolver_node(self, state: AquaBioState) -> dict:
        query = state.get("original_query", "").strip()
        summary = state.get("memory_summary", {})
        previous_species = list(summary.get("last_species_ids", []))
        explicit_species = self._normalize_species_ids([query])
        explicit_latin_name = bool(
            re.search(r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b", query)
        )
        # With a newly supplied image, "这个生物" refers to the current image,
        # not to a previous turn.
        followup = (
            not state.get("image_path")
            and not explicit_species
            and not explicit_latin_name
            and any(marker in query for marker in FOLLOWUP_MARKERS)
        )
        response_mode = "normal"
        constraints: list[str] = []
        if any(marker in query for marker in COLOR_MARKERS) and any(
            marker in query for marker in SHORT_MARKERS
        ):
            response_mode = "short_color_only"
            constraints = [
                "只输出常见颜色",
                "不解释、不引用、不添加物种名称",
                "使用一个简短中文句子",
            ]
        elif any(marker in query for marker in SHORT_MARKERS):
            response_mode = "short_answer"
            constraints = ["严格简短回答，不添加无关信息"]

        resolved_species = (
            previous_species if followup else explicit_species
        )
        resolved_query = query
        if followup and previous_species:
            names = [
                self.species_by_id[item]["chinese_name"]
                for item in previous_species
                if item in self.species_by_id
            ]
            resolved_query = (
                f"追问指代的上一轮物种是{'、'.join(names)}。"
                f"用户追问：{query}"
            )

        warnings = list(state.get("warnings", []))
        need_clarification = followup and not previous_species
        clarification_question = ""
        if followup and not previous_species:
            warnings.append(
                "检测到追问指代，但该会话没有可解析的上一轮物种。"
            )
            clarification_question = (
                "当前会话没有可解析的上一轮生物。请先提供图片，"
                "或直接说明要查询的生物名称。"
            )
        return {
            "resolved_query": resolved_query,
            "resolved_species_ids": resolved_species,
            "image_caption": (
                summary.get("last_image_caption", "")
                if followup
                else state.get("image_caption")
            ),
            "followup_detected": followup,
            "need_clarification": need_clarification,
            "clarification_question": clarification_question,
            "response_mode": response_mode,
            "response_constraints": constraints,
            "warnings": warnings,
            "trace": _append_trace(
                state,
                "followup_resolver:"
                f"{followup}:{','.join(resolved_species) or 'none'}:"
                f"{response_mode}",
            ),
        }

    def router_node(self, state: AquaBioState) -> dict:
        query = (
            state.get("resolved_query")
            or state.get("original_query", "")
        ).strip()
        has_image = bool(state.get("image_path"))
        options = state.get("runtime_options", {})
        rag_enabled = options.get("rag_enabled", True)
        vision_enabled = options.get("vision_enabled", True)
        pdf_enabled = options.get("pdf_enabled", True)
        image_search_enabled = options.get(
            "image_search_enabled", True
        )
        latin_query = bool(
            re.search(r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b", query)
        )
        needs_pdf_graph_image = any(
            marker in query for marker in PDF_GRAPH_IMAGE_MARKERS
        ) or (
            latin_query
            and any(
                marker in query.casefold()
                for marker in ("pdf", "page", "evidence", "image")
            )
        )
        if state.get("followup_detected") and state.get(
            "resolved_species_ids"
        ):
            task = "followup_text_qa"
        elif has_image and any(
            marker in query.lower() for marker in DETECTION_MARKERS
        ):
            task = "detection_qa"
        elif has_image and query:
            task = "multimodal_qa"
        elif has_image:
            task = "image_qa"
        elif needs_pdf_graph_image:
            task = "pdf_qa"
        elif any(marker in query.lower() for marker in SOURCE_MARKERS):
            task = "source_trace"
        elif any(marker in query.lower() for marker in PDF_MARKERS):
            task = "pdf_qa"
        elif any(marker in query.lower() for marker in COMPARISON_MARKERS):
            task = "comparison_qa"
        else:
            task = "text_qa"

        decision = RouteDecision(
            task_type=task,
            need_vlm=has_image and vision_enabled,
            need_text_retrieval=rag_enabled and task
            in {
                "text_qa",
                "followup_text_qa",
                "comparison_qa",
                "image_qa",
                "multimodal_qa",
                "detection_qa",
            },
            need_image_retrieval=rag_enabled
            and image_search_enabled
            and (
                task in {"image_qa", "multimodal_qa"}
                or asks_for_reference_images(query)
                or needs_pdf_graph_image
            ),
            need_multimodal_retrieval=rag_enabled and task
            in {"image_qa", "multimodal_qa"},
            need_pdf_retrieval=(
                rag_enabled
                and pdf_enabled
                and task != "followup_text_qa"
            ),
            need_detection=task == "detection_qa",
        )
        return {
            "route": decision.model_dump(),
            "trace": _append_trace(state, f"router:{task}"),
        }

    def rewrite_node(self, state: AquaBioState) -> dict:
        query = (
            state.get("resolved_query")
            or state.get("original_query", "")
        ).strip()
        task = state["route"]["task_type"]
        if task == "followup_text_qa":
            query += "。严格围绕已解析的上一轮物种检索，不扩展到其他物种。"
        elif not query and task == "image_qa":
            query = (
                "识别图片中可能的水下生物，并说明视觉特征、"
                "相似物种和栖息环境。"
            )
        elif task == "comparison_qa":
            query += (
                "。重点检索视觉特征、身体结构、相似物种、"
                "image_recognition_tips 和权威 PDF 描述。"
            )
        elif task == "source_trace":
            query += "。返回物种卡片、图片来源页、PDF 名称和页码。"
        elif task == "pdf_qa":
            query += "。优先使用已登记 PDF 文档并保留页码。"
        else:
            query += "。检索物种定义、视觉特征、栖息地和生态信息。"
        return {
            "rewritten_query": query,
            "trace": _append_trace(state, "query_rewrite"),
        }

    def source_selection_node(self, state: AquaBioState) -> dict:
        route = state["route"]
        tools = []
        if route["need_vlm"]:
            tools.append("vlm_caption")
        if route["need_text_retrieval"]:
            tools.append("text_retriever")
        if route["need_image_retrieval"]:
            tools.append("image_retriever")
        query = (
            state.get("resolved_query")
            or state.get("original_query", "")
        )
        if (
            route["need_image_retrieval"]
            and (
                asks_for_reference_images(query)
                or any(marker in query for marker in PDF_GRAPH_IMAGE_MARKERS)
            )
        ):
            tools.append("pdf_image_retriever")
        if route["need_multimodal_retrieval"]:
            tools.append("multimodal_retriever")
        if route["need_pdf_retrieval"]:
            tools.append("pdf_retriever")
        if route["need_detection"]:
            tools.append("yolov8_detector")
        return {
            "selected_tools": tools,
            "trace": _append_trace(
                state,
                "source_selection:"
                + ",".join(tools)
                + ";flow="
                + _tool_flow_label(route["task_type"], tools),
            ),
        }

    def react_tool_plan_node(self, state: AquaBioState) -> dict:
        routed_tools = list(state.get("selected_tools", []))
        plan = list(routed_tools)
        step = state.get("react_step", 0) + 1
        planning_mode = "deterministic"
        if (
            os.getenv("MRAG_REACT_NATIVE", "false").lower()
            in {"1", "true", "yes", "on"}
            and self.llm.enabled
        ):
            try:
                selected = self.react_planner.plan(
                    state.get("rewritten_query", ""),
                    plan,
                    state.get("tool_observations", []),
                    step - 1,
                )
                if selected:
                    plan = selected
                    planning_mode = "native_function_calling"
            except Exception:
                # Deterministic routing remains the controlled fallback.
                pass
        # Explicit user modalities are routing constraints, not optional
        # suggestions for the LLM planner.
        for required in (
            "vlm_caption",
            "pdf_image_retriever",
        ):
            if required in routed_tools and required not in plan:
                plan.append(required)
        return {
            "tool_plan": plan,
            "selected_tools": plan,
            "tool_observations": state.get("tool_observations", []),
            "react_step": step,
            "trace": _append_trace(
                state,
                f"react_tool_plan:step={step};mode={planning_mode};"
                f"tools={','.join(plan)}",
            ),
        }

    def clarification_node(self, state: AquaBioState) -> dict:
        answer = state.get("clarification_question") or (
            "请说明你指的是哪一种生物。"
        )
        if state.get("hitl_enabled"):
            response = interrupt(
                {
                    "review_type": "clarification",
                    "question": answer,
                    "session_id": state.get("session_id", "default"),
                    "original_query": state.get("original_query", ""),
                }
            )
            response_text = (
                str(response.get("answer", "")).strip()
                if isinstance(response, dict)
                else str(response).strip()
            )
            return {
                "original_query": response_text,
                "resolved_query": response_text,
                "need_clarification": False,
                "human_review": {
                    "status": "resumed",
                    "answer": response_text,
                },
                "trace": _append_trace(
                    state, "human_in_loop:resumed"
                ),
            }
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "trace": _append_trace(state, "human_in_loop:clarification"),
        }

    def vision_node(self, state: AquaBioState) -> dict:
        started = time.perf_counter()
        image_path = state.get("image_path")
        if not image_path:
            return {"trace": _append_trace(state, "vision:skipped")}

        known = self.known_images.get(
            str(Path(image_path).resolve()).lower()
        )
        if known:
            return {
                "image_caption": known["caption"],
                "vision_failed": False,
                "visual_features": known.get(
                    "visual_keywords", []
                ),
                "candidate_species": [known["species_id"]],
                "detected_species_ids": [known["species_id"]],
                "warnings": list(state.get("warnings", [])),
                "trace": _append_trace(
                    state,
                    "vision:catalog_match:"
                    f"{time.perf_counter() - started:.2f}s",
                ),
            }
        vision_provider = os.getenv(
            "AQUABIO_VISION_PROVIDER", "auto"
        ).lower()
        vision_llm = (
            self.llm
            if vision_provider == "qwen"
            else self.gemini if self.gemini.enabled else self.llm
        )
        if not vision_llm.enabled:
            return {
                "image_caption": "图片已上传，但离线模式未调用视觉模型。",
                "vision_failed": True,
                "visual_features": [],
                "candidate_species": [],
                "detected_species_ids": (
                    [known["species_id"]] if known else []
                ),
                "warnings": [
                    *state.get("warnings", []),
                    "离线模式无法生成图片 caption。",
                ],
                "trace": _append_trace(state, "vision:offline"),
            }
        try:
            analysis = vision_llm.analyze_image(
                image_path,
                (
                    "严格分析这张水下图片。输出候选物种、可见形态、"
                    "颜色、环境与不确定性。候选优先限制为系统20类，"
                    "不得虚构检测框。"
                ),
            )
        except Exception as error:
            if known:
                return {
                    "image_caption": known["caption"],
                    "vision_failed": False,
                    "visual_features": known.get(
                        "visual_keywords", []
                    ),
                    "candidate_species": [known["species_id"]],
                    "detected_species_ids": [known["species_id"]],
                    "warnings": [
                        *state.get("warnings", []),
                        "视觉模型分析失败，已使用知识库登记 caption。",
                    ],
                    "trace": _append_trace(
                        state, "vision:catalog_fallback"
                    ),
                }
            return {
                "image_caption": "视觉模型未能生成有效图片描述。",
                "vision_failed": True,
                "visual_features": [],
                "candidate_species": [],
                "detected_species_ids": [],
                "warnings": [
                    *state.get("warnings", []),
                    f"视觉模型分析失败：{error}",
                ],
                "trace": _append_trace(
                    state, f"vision:fallback:{type(error).__name__}"
                ),
            }

        detected = self._normalize_species_ids(
            analysis.get("possible_species", [])
        )
        if known and known["species_id"] not in detected:
            detected.insert(0, known["species_id"])
        warnings = list(state.get("warnings", []))
        if not analysis.get("structured_output", True):
            warnings.extend(analysis.get("limitations", []))
        return {
            "image_caption": analysis["description"],
            "vision_failed": False,
            "visual_features": analysis.get("visible_features", []),
            "candidate_species": analysis.get("possible_species", []),
            "detected_species_ids": detected,
            "warnings": warnings,
            "trace": _append_trace(
                state,
                f"vision:completed:{time.perf_counter() - started:.2f}s"
                if analysis.get("structured_output", True)
                else (
                    "vision:text_fallback:"
                    f"{time.perf_counter() - started:.2f}s"
                ),
            ),
        }

    def detection_node(self, state: AquaBioState) -> dict:
        started = time.perf_counter()
        image_path = state.get("image_path")
        if not image_path:
            return {
                "detection_result": {},
                "detection_comparison": {},
                "detection_visualization_path": "",
                "trace": _append_trace(state, "detection:skipped"),
            }
        try:
            from aquabio.detector import YOLOv8Detector, AgentDetectionOrchestrator
            from aquabio.visualizer import draw_detections

            detector = YOLOv8Detector()
            orchestrator = AgentDetectionOrchestrator(detector)
            output_dir = self.paths.outputs_dir / "detection"
            result = orchestrator.analyze_and_detect(
                image_path, output_dir=output_dir
            )
            vis_path = ""
            if result.get("original_detection", {}).get("detections"):
                vis_path = str(
                    output_dir / f"{Path(image_path).stem}_detected.jpg"
                )
                draw_detections(
                    image_path,
                    result["original_detection"]["detections"],
                    output_path=vis_path,
                )
            if result.get("enhanced_results"):
                best = result.get("best_method", "original")
                if best != "original" and best in result["enhanced_results"]:
                    enh_data = result["enhanced_results"][best]
                    best_vis_path = str(
                        output_dir / f"{Path(image_path).stem}_best_{best}.jpg"
                    )
                    draw_detections(
                        enh_data["enhanced_path"],
                        enh_data["detection"]["detections"],
                        output_path=best_vis_path,
                    )
                    if not vis_path:
                        vis_path = best_vis_path
            comparison = {
                "original": {
                    "path": str(image_path),
                    "detection": result.get("original_detection", {}),
                },
                "enhanced": {
                    method: {
                        "path": data.get("enhanced_path", ""),
                        "detection": data.get("detection", {}),
                    }
                    for method, data in result.get("enhanced_results", {}).items()
                },
            }
            return {
                "detection_result": result.get("original_detection", {}),
                "detection_comparison": comparison,
                "detection_visualization_path": vis_path,
                "trace": _append_trace(
                    state,
                    f"detection:completed:"
                    f"dets={result.get('original_detection', {}).get('num_detections', 0)};"
                    f"best={result.get('best_method', 'original')};"
                    f"elapsed={time.perf_counter() - started:.2f}s",
                ),
            }
        except ImportError:
            return {
                "detection_result": {},
                "detection_comparison": {},
                "detection_visualization_path": "",
                "warnings": [
                    *state.get("warnings", []),
                    "ultralytics 未安装，目标检测功能不可用。请执行 pip install ultralytics。",
                ],
                "trace": _append_trace(state, "detection:ultralytics_missing"),
            }
        except Exception as error:
            return {
                "detection_result": {},
                "detection_comparison": {},
                "detection_visualization_path": "",
                "warnings": [
                    *state.get("warnings", []),
                    f"目标检测失败：{type(error).__name__}: {error}",
                ],
                "trace": _append_trace(
                    state, f"detection:error:{type(error).__name__}"
                ),
            }

    def retrieval_node(self, state: AquaBioState) -> dict:
        started = time.perf_counter()
        task = state["route"]["task_type"]
        query = state["rewritten_query"]
        caption = state.get("image_caption") or ""
        candidates = []
        for source in (
            state.get("resolved_species_ids", []),
            state.get("detected_species_ids", []),
            self._normalize_species_ids(
                state.get("candidate_species", [])
            ),
        ):
            for species_id in source:
                if species_id not in candidates:
                    candidates.append(species_id)
        top_k = self.settings.top_k + 4 * state.get("retry_count", 0)
        use_mcp = state.get("runtime_options", {}).get(
            "mcp_enabled", True
        )

        text_context: list[dict] = []
        web_context: list[dict] = []
        image_context: list[dict] = []
        multimodal_context: list[dict] = []
        pdf_context: list[dict] = []
        text_retrieval_meta: dict[str, Any] = {
            "counts": {},
            "warnings": [],
            "tool_calls": [],
        }
        chroma_image_retrieval_meta: dict[str, Any] = {
            "counts": {},
            "warnings": [],
            "tool_calls": [],
        }
        multimodal_retrieval_meta: dict[str, Any] = {
            "counts": {},
            "warnings": [],
            "tool_calls": [],
        }
        original_query = state.get("original_query", "")
        related_image = asks_for_related_entity_image(original_query)
        web_plan: dict[str, Any] = {}
        web_warnings: list[str] = []
        subject = ""
        if candidates:
            species = self.species_by_id.get(candidates[0], {})
            subject = (
                species.get("english_name")
                or species.get("scientific_name")
                or species.get("chinese_name")
                or candidates[0]
            )
        latin_caption = re.search(
            r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b",
            caption,
        )
        latin_query = re.search(
            r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b",
            original_query,
        )
        if latin_caption and not subject:
            subject = latin_caption.group(0)
        if latin_query and not subject:
            subject = latin_query.group(0)
        if needs_web_research(original_query):
            web_plan = plan_web_research(
                self.llm,
                subject or "marine organism",
                original_query,
            )
            web_context, web_warnings = search_wikipedia(
                web_plan.get("search_query", original_query),
                focus=web_plan.get("focus", original_query),
                top_k=3,
            )
        image_search_text = (
            web_plan.get("image_query", "")
            if related_image
            else caption or query
        )
        image_entity = (
            web_plan.get("image_target", "")
            if related_image
            else (
                self.species_by_id.get(candidates[0], {}).get(
                    "scientific_name", ""
                )
                if candidates
                else (latin_query.group(0) if latin_query else "")
            )
        )
        mcp_retrieval = state.get("runtime_options", {}).get(
            "mcp_retrieval_enabled", False
        )
        if "text_retriever" in state["selected_tools"]:
            if mcp_retrieval:
                text_context, text_retrieval_meta = (
                    self.retrieval_agent.search_text_mcp(
                    query=f"{query}\n{caption}",
                    top_k=top_k,
                    species_ids=candidates or None,
                    )
                )
            else:
                text_context = self.retriever.search(
                    RetrievalRequest(
                        query=f"{query}\n{caption}",
                        task_type=task,
                        top_k=top_k,
                        species_ids=candidates or None,
                        source_types=[
                            "species_card",
                            "species_text_chunk",
                        ],
                    )
                )
        if (
            "image_retriever" in state["selected_tools"]
            and image_search_text
        ):
            if "pdf_image_retriever" not in state["selected_tools"]:
                if mcp_retrieval:
                    image_context, chroma_image_retrieval_meta = (
                        self.retrieval_agent.search_image_mcp(
                            image_search_text,
                            top_k=top_k,
                            species_ids=(
                                None if related_image else candidates or None
                            ),
                        )
                    )
                else:
                    image_context = self.retriever.image_search(
                        image_search_text,
                        top_k=top_k,
                        species_ids=None if related_image else candidates or None,
                    )
            if related_image and image_entity:
                target = image_entity.casefold()
                image_context = [
                    row
                    for row in image_context
                    if target
                    in " ".join(
                        [
                            row.get("content", ""),
                            str(row.get("metadata", {})),
                        ]
                    ).casefold()
                ]
        if "pdf_image_retriever" in state["selected_tools"]:
            if self.offline:
                pdf_image_context = []
                image_retrieval_meta = {
                    "counts": {"pdf_images": 0},
                    "warnings": [
                        "Offline mode does not start the PDF image MCP tool."
                    ],
                }
            else:
                pdf_image_context, image_retrieval_meta = (
                    self.retrieval_agent.search_pdf_images(
                        image_search_text or query,
                        top_k=min(top_k, 8),
                        entity=image_entity,
                        use_mcp=use_mcp,
                    )
                )
            image_context.extend(pdf_image_context)
            if not candidates and image_entity:
                target = image_entity.casefold()
                pdf_only = [
                    row
                    for row in image_context
                    if str(
                        row.get("metadata", {}).get(
                            "retrieval_source", ""
                        )
                    )
                    == "raganything_pdf_image"
                    and target
                    in " ".join(
                        [
                            row.get("content", ""),
                            str(row.get("metadata", {})),
                        ]
                    ).casefold()
                ]
                if pdf_only:
                    image_context = pdf_only
            if related_image and image_entity:
                target = image_entity.casefold()
                image_context = [
                    row
                    for row in image_context
                    if target
                    in " ".join(
                        [
                            row.get("content", ""),
                            str(row.get("metadata", {})),
                        ]
                    ).casefold()
                ]
        else:
            image_retrieval_meta = {"counts": {}, "warnings": []}
        image_query = original_query
        requested_roles = requested_image_roles(image_query)
        if asks_for_reference_images(image_query) and not requested_roles:
            requested_roles = {SPECIMEN_ROLE}
        network_count = 0
        if (
            requested_roles
            and not self.offline
            and state.get("runtime_options", {}).get(
                "image_search_enabled", True
            )
        ):
            local_pdf_image_roles = {
                role
                for role in requested_roles
                if any(
                    role_matches(
                        str(
                            row.get("metadata", {}).get(
                                "image_role", SPECIMEN_ROLE
                            )
                        ),
                        role,
                    )
                    and str(
                        row.get("metadata", {}).get(
                            "retrieval_source", ""
                        )
                    )
                    == "raganything_pdf_image"
                    for row in image_context
                )
            }
            missing_roles = [
                role
                for role in (DISTRIBUTION_ROLE, SPECIMEN_ROLE)
                if role in requested_roles
                and role not in local_pdf_image_roles
                and not any(
                    role_matches(
                        str(
                            row.get("metadata", {}).get(
                                "image_role", SPECIMEN_ROLE
                            )
                        ),
                        role,
                    )
                    for row in image_context
                )
            ]
            if missing_roles:
                from aquabio_web.network_images import (
                    fetch_commons_images,
                    preferred_taxon_query,
                )

                latin_match = re.search(
                    r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b",
                    image_query,
                )
                species = (
                    self.species_by_id.get(candidates[0], {})
                    if candidates
                    else {}
                )
                network_query = (
                    web_plan.get("image_query", "")
                    if related_image
                    else ""
                ) or (
                    preferred_taxon_query(
                        species.get("scientific_name", ""),
                        species.get("english_name", ""),
                    )
                    or (latin_match.group(0) if latin_match else image_query)
                )
                cache_id = (
                    hashlib.sha256(
                        network_query.encode("utf-8")
                    ).hexdigest()[:16]
                    if related_image
                    else candidates[0]
                    if candidates
                    else hashlib.sha256(
                        network_query.encode("utf-8")
                    ).hexdigest()[:16]
                )
                for role in missing_roles:
                    network_rows, network_warnings = fetch_commons_images(
                        self.paths.root,
                        cache_id,
                        network_query,
                        top_k=1,
                        image_role=role,
                    )
                    image_retrieval_meta["warnings"].extend(
                        network_warnings
                    )
                    for row in network_rows:
                        image_context.append(
                            {
                                "id": row["image_id"],
                                "content": row.get("caption", ""),
                                "semantic_similarity": 0.0,
                                "final_score": 0.25,
                                "metadata": {
                                    "source_type": "network_image",
                                    "retrieval_source": "wikimedia_commons",
                                    "modality": "image",
                                    "image_path": row.get(
                                        "image_path", ""
                                    ),
                                    "image_role": row.get(
                                        "image_role", role
                                    ),
                                    "scientific_name": row.get(
                                        "scientific_name", network_query
                                    ),
                                    "common_name": row.get(
                                        "common_name", ""
                                    ),
                                    "source_page": row.get(
                                        "source_page", ""
                                    ),
                                    "license": row.get("license", ""),
                                },
                            }
                        )
                        network_count += 1
        image_retrieval_meta["counts"]["network_images"] = network_count
        if (
            "multimodal_retriever" in state["selected_tools"]
            and caption
        ):
            if mcp_retrieval:
                (
                    multimodal_context,
                    multimodal_retrieval_meta,
                ) = (
                    self.retrieval_agent.search_multimodal_mcp(
                        query,
                        caption,
                        top_k=top_k,
                        species_ids=candidates or None,
                    )
                )
            else:
                multimodal_context = self.retriever.multimodal_search(
                    query,
                    caption,
                    top_k=top_k,
                    species_ids=candidates or None,
                )
        if "pdf_retriever" in state["selected_tools"]:
            if self.offline:
                pdf_context = self.retriever.pdf_search(
                    f"{query}\n{caption}",
                    top_k=top_k,
                    species_ids=candidates or None,
                )
                retrieval_meta = {
                    "counts": {"chroma": len(pdf_context)},
                    "warnings": [],
                }
            elif mcp_retrieval:
                # Full MCP path: Chroma PDF chunks + RAG-Anything hybrid search
                (
                    pdf_context_chroma,
                    pdf_chroma_meta,
                ) = (
                    self.retrieval_agent.search_pdf_chunks_mcp(
                        f"{query}\n{caption}",
                        top_k=top_k,
                        species_ids=candidates or None,
                    )
                )
                pdf_context = list(pdf_context_chroma)
                retrieval_warnings: list[str] = []
                try:
                    graph_payload = self.retrieval_agent.mcp.call_tool_sync(
                        "raganything",
                        "raganything_hybrid_search",
                        {"query": f"{query}\n{caption}", "top_k": top_k},
                    )
                    graph_rows = self.retrieval_agent._graph_evidence(
                        graph_payload
                    )
                    for row in graph_rows:
                        pdf_context.append(
                            {
                                "id": row.id,
                                "content": row.content,
                                "semantic_similarity": row.score,
                                "final_score": row.score,
                                "metadata": {
                                    **row.metadata,
                                    "doc_id": row.doc_id,
                                    "page": row.page,
                                    "doc_title": row.source_file,
                                    "source_type": "lightrag_graph",
                                    "modality": row.modality,
                                    "entity_names": row.entity_names,
                                    "relation_path": row.relation_path,
                                    "retrieval_source": "mcp_lightrag_graph",
                                },
                            }
                        )
                except Exception as error:
                    retrieval_warnings.append(
                        "RAG-Anything hybrid_search MCP unavailable in "
                        f"mcp_retrieval mode: {type(error).__name__}: {error}"
                    )
                retrieval_meta = {
                    "counts": {
                        "chroma": len(pdf_context_chroma),
                        "graph": len(pdf_context) - len(pdf_context_chroma),
                        **pdf_chroma_meta.get("counts", {}),
                    },
                    "warnings": [
                        *pdf_chroma_meta.get("warnings", []),
                        *retrieval_warnings,
                    ],
                    "tool_calls": [
                        *pdf_chroma_meta.get("tool_calls", []),
                    ],
                }
            else:
                pdf_context, retrieval_meta = (
                    self.retrieval_agent.search_pdf(
                        f"{query}\n{caption}",
                        top_k=top_k,
                        species_ids=candidates or None,
                        graph_entities=[
                            name
                            for species_id in candidates
                            for name in (
                                self.species_by_id.get(
                                    species_id, {}
                                ).get("scientific_name", ""),
                                self.species_by_id.get(
                                    species_id, {}
                                ).get("english_name", ""),
                                species_id,
                            )
                            if name
                        ]
                        + (
                            [latin_query.group(0)]
                            if latin_query
                            and latin_query.group(0)
                            not in {
                                name
                                for species_id in candidates
                                for name in (
                                    self.species_by_id.get(
                                        species_id, {}
                                    ).get("scientific_name", ""),
                                    self.species_by_id.get(
                                        species_id, {}
                                    ).get("english_name", ""),
                                    species_id,
                                )
                                if name
                            }
                            else []
                        ),
                        use_mcp=use_mcp,
                    )
                )
        else:
            retrieval_meta = {"counts": {}, "warnings": []}
        observations = [
            f"text_retriever:{len(text_context)}",
            f"web_research:{len(web_context)}",
            f"image_retriever:{len(image_context)}",
            f"multimodal_retriever:{len(multimodal_context)}",
            f"pdf_retriever:{len(pdf_context)}",
        ]
        species_filter = ",".join(candidates) or "none"
        tool_calls = [
            *state.get("tool_calls", []),
            *text_retrieval_meta.get("tool_calls", []),
            *chroma_image_retrieval_meta.get("tool_calls", []),
            *multimodal_retrieval_meta.get("tool_calls", []),
            *retrieval_meta.get("tool_calls", []),
            *image_retrieval_meta.get("tool_calls", []),
        ]
        mcp_trace = [
            (
                "mcp_tool_call:"
                f"{item.get('tool_name')} "
                f"status={item.get('status')} "
                f"latency={item.get('latency_ms')}ms "
                f"results={item.get('result_count', 0)}"
            )
            for item in tool_calls
            if item.get("tool_source") == "mcp"
        ]
        return {
            "text_context": text_context,
            "web_context": web_context,
            "image_context": image_context,
            "multimodal_context": multimodal_context,
            "pdf_context": pdf_context,
            "fused_context": pdf_context,
            "graph_context": [
                row
                for row in pdf_context
                if row.get("metadata", {}).get("retrieval_source")
                == "lightrag_graph"
            ],
            "graph_trace": [
                f"{name}:{count}"
                for name, count in {
                    **text_retrieval_meta["counts"],
                    **chroma_image_retrieval_meta["counts"],
                    **multimodal_retrieval_meta["counts"],
                    **retrieval_meta["counts"],
                    **image_retrieval_meta["counts"],
                }.items()
            ],
            "warnings": [
                *state.get("warnings", []),
                *web_warnings,
                *text_retrieval_meta["warnings"],
                *chroma_image_retrieval_meta["warnings"],
                *multimodal_retrieval_meta["warnings"],
                *retrieval_meta["warnings"],
                *image_retrieval_meta["warnings"],
            ],
            "tool_observations": observations,
            "tool_calls": tool_calls,
            "web_research_plan": web_plan,
            "trace": _append_trace(
                state,
                "retrieval:"
                f"text={len(text_context)},web={len(web_context)},"
                f"image={len(image_context)},"
                f"pair={len(multimodal_context)},pdf={len(pdf_context)},"
                "mcp_graph="
                f"{retrieval_meta['counts'].get('graph', 0)},"
                "mcp_pdf_images="
                f"{image_retrieval_meta['counts'].get('raganything_pdf_images', 0)},"
                f"filter_species={species_filter},"
                f"elapsed={time.perf_counter() - started:.2f}s",
            )
            + mcp_trace
            + (
                [
                    "web_plan:"
                    f"search={web_plan.get('search_query', '')};"
                    f"image={web_plan.get('image_target', '')}"
                ]
                if web_plan
                else []
            ),
        }

    def rerank_node(self, state: AquaBioState) -> dict:
        updates: dict[str, Any] = {}
        total = 0
        for key in (
            "text_context",
            "web_context",
            "image_context",
            "multimodal_context",
            "pdf_context",
        ):
            rows = sorted(
                state.get(key, []),
                key=lambda row: row.get("final_score", 0.0),
                reverse=True,
            )
            updates[key] = rows[: self.settings.top_k]
            total += len(updates[key])
        updates["trace"] = _append_trace(
            state, f"rerank:weighted:{total}"
        )
        return updates

    def context_node(self, state: AquaBioState) -> dict:
        rows = _dedupe(
            [
                *state.get("text_context", []),
                *state.get("web_context", []),
                *state.get("image_context", []),
                *state.get("multimodal_context", []),
                *state.get("pdf_context", []),
            ]
        )
        sections = []
        if state.get("extra_context", "").strip():
            sections.append(
                "[E0] type=uploaded_pdf\n"
                + state["extra_context"].strip()
            )
        for index, row in enumerate(rows[:24], start=1):
            metadata = row["metadata"]
            source = (
                metadata.get("doc_title")
                or metadata.get("source_page")
                or metadata.get("wikipedia_url")
                or row["id"]
            )
            page = metadata.get("page", "")
            source_label = (
                f"{source}, page {page}" if page != "" else str(source)
            )
            sections.append(
                f"[E{index}] id={row['id']} "
                f"species={metadata.get('species_id', '')} "
                f"type={metadata.get('source_type', '')} "
                f"score={row['final_score']}\n"
                f"source={source_label}\n{row['content']}"
            )
        return {
            "final_context": "\n\n".join(sections),
            "trace": _append_trace(
                state, f"context_builder:{len(rows)}"
            ),
        }

    def answer_node(self, state: AquaBioState) -> dict:
        started = time.perf_counter()
        image_uploaded = bool(state.get("image_path"))
        vision_enabled = state.get("runtime_options", {}).get(
            "vision_enabled", True
        )
        if image_uploaded and not vision_enabled:
            answer = (
                "图片附件已经收到，但当前运行模式关闭了图像理解功能，"
                "所以我不能直接识别图片中的生物。请打开右侧“图像理解”"
                "开关后重新提问。"
            )
        elif state["route"]["need_vlm"] and state.get("vision_failed"):
            answer = (
                "视觉模型没有返回有效图片分析，当前无法可靠判断图片"
                "中的生物类别。"
            )
        elif not self.llm.enabled:
            answer = (
                "当前为离线模式，以下是检索证据：\n\n"
                + state.get("final_context", "")[:8000]
            )
        else:
            payload = {
                "task_type": state["route"]["task_type"],
                "original_query": state.get("original_query", ""),
                "resolved_query": state.get("resolved_query", ""),
                "resolved_species_ids": state.get(
                    "resolved_species_ids", []
                ),
                "conversation_history": state.get(
                    "conversation_history", []
                ),
                "image_caption": state.get("image_caption"),
                "image_uploaded": image_uploaded,
                "vision_enabled": vision_enabled,
                "image_analysis_available": bool(
                    state.get("image_caption")
                    or state.get("visual_features")
                    or state.get("candidate_species")
                ),
                "visual_features": state.get("visual_features", []),
                "candidate_species": state.get(
                    "candidate_species", []
                ),
                "response_mode": state.get("response_mode", "normal"),
                "response_constraints": state.get(
                    "response_constraints", []
                ),
                "evidence": state.get("final_context", ""),
                "detection_result": state.get("detection_result", {}),
                "detection_comparison": state.get("detection_comparison", {}),
                "detection_best_method": (
                    state.get("detection_comparison", {})
                    .get("original", {})
                    .get("detection", {})
                    .get("num_detections", 0)
                    if not state.get("detection_comparison", {}).get("enhanced")
                    else "see_comparison"
                ),
                "requirements": [
                    "只依据图片 caption、会话实体和检索证据回答。",
                    "普通回答的关键结论使用[E编号]引用。",
                    "追问必须解析上一轮明确保存的物种，不可重新猜测。",
                    "若 response_constraints 要求短答，必须严格服从。",
                    "回答使用中文。",
                    "如果 image_uploaded 为 true，绝不能说用户没有上传图片；"
                    "只能说明视觉分析是否可用以及你能从 caption/证据判断什么。",
                    "如果用户询问中文名、普通名或名称含义，先检查证据中的 "
                    "scientific_name、common_name、title 和 PDF caption。"
                    "若只有英文 common name，没有权威中文名，不要说完全不知道；"
                    "应给出英文名、中文直译/意译，并明确标注“非正式译名”。",
                ],
            }
            requested_roles = requested_image_roles(
                state.get("original_query", "")
            )
            payload["retrieved_images"] = [
                {
                    "image_id": row.get("id", ""),
                    "image_role": row.get("metadata", {}).get(
                        "image_role", "specimen"
                    ),
                    "scientific_name": row.get("metadata", {}).get(
                        "scientific_name", ""
                    ),
                    "page": row.get("metadata", {}).get("page"),
                    "source_page": row.get("metadata", {}).get(
                        "source_page", ""
                    ),
                }
                for row in state.get("image_context", [])
                if row.get("metadata", {}).get("image_path")
                and (
                    not requested_roles
                    or any(
                        role_matches(
                            str(
                                row.get("metadata", {}).get(
                                    "image_role", ""
                                )
                            ),
                            role,
                        )
                        for role in requested_roles
                    )
                )
            ]
            payload["strict_instructions"] = [
                (
                    "Directly answer the requested species, distribution, "
                    "and requested image roles. Never switch species."
                ),
                (
                    "When retrieved_images is non-empty, state that the "
                    "frontend displays one best-matching image. Never claim "
                    "no image exists and never promise multiple displayed "
                    "images."
                ),
                (
                    "For PDF images, always name the exact taxon shown by "
                    "scientific_name/common_name and mention the PDF page "
                    "or printed page when available. If the user used a "
                    "broad Chinese group name such as 海星, explain which "
                    "specific starfish species the displayed local PDF "
                    "image belongs to."
                ),
                (
                    "When the user requests a distribution map, only an "
                    "image whose role is distribution_map may be described "
                    "as displayed. A specimen or habitat photograph is not "
                    "a distribution map. If retrieved_images is empty, do "
                    "not claim that any image is displayed."
                ),
            ]
            if state.get("web_context"):
                payload["strict_instructions"].append(
                    "Web evidence is available. Use it for the requested "
                    "fact, cite its E-number, and do not claim that the "
                    "knowledge base contains no answer."
                )
            if asks_for_related_entity_image(
                state.get("original_query", "")
            ):
                payload["strict_instructions"].append(
                    "The requested image is of a related organism such as "
                    "a predator. Describe that related organism, not an "
                    "unrelated PDF image and not another subject specimen."
                )
            try:
                answer = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是 AquaBio-MRAG 水下生物专家。"
                                "你必须保持同一 session 的对话上下文，"
                                "并且只使用证据支持具体事实。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload, ensure_ascii=False
                            ),
                        },
                    ],
                    max_tokens=1800,
                    max_continuations=0,
                )
            except Exception as error:
                rows = _dedupe(
                    [
                        *state.get("text_context", []),
                        *state.get("web_context", []),
                        *state.get("image_context", []),
                        *state.get("multimodal_context", []),
                        *state.get("pdf_context", []),
                    ]
                )
                ecology = next(
                    (
                        (index, row)
                        for index, row in enumerate(rows, start=1)
                        if row.get("metadata", {}).get("chunk_type")
                        == "ecology_behavior"
                    ),
                    None,
                )
                habitat = next(
                    (
                        (index, row)
                        for index, row in enumerate(rows, start=1)
                        if row.get("metadata", {}).get("chunk_type")
                        == "habitat"
                    ),
                    None,
                )
                image_count = sum(
                    bool(row.get("metadata", {}).get("image_path"))
                    for row in rows
                )
                species_names = [
                    self.species_by_id[item].get(
                        "chinese_name", item
                    )
                    for item in (
                        state.get("resolved_species_ids", [])
                        or state.get("detected_species_ids", [])
                    )
                    if item in self.species_by_id
                ]
                subject = "、".join(species_names) or "该水下生物"
                lines = [
                    f"上一轮识别的生物是**{subject}**。",
                    "",
                ]
                for label, selected in (
                    ("生活习性", ecology),
                    ("栖息环境", habitat),
                ):
                    if selected:
                        index, row = selected
                        excerpt = " ".join(
                            row.get("content", "").split()
                        )[:260]
                        lines.append(
                            f"- **{label}**：{excerpt}[E{index}]"
                        )
                if image_count:
                    lines.extend(
                        [
                            "",
                            f"已从本地知识库或PDF图片索引找到 "
                            f"**{image_count} 张图片**，见下方图片画廊。",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "",
                            "本地知识库和PDF图片索引暂未找到图片，"
                            "系统将尝试网络图片来源。",
                        ]
                    )
                return {
                    "draft_answer": "\n".join(lines),
                    "generation_failed": False,
                    "provider_fallback": True,
                    "warnings": [
                        *state.get("warnings", []),
                        "Qwen生成超时，已使用本地检索证据生成"
                        "结构化回答。",
                    ],
                    "trace": _append_trace(
                        state,
                        f"answer:qwen_failed:{type(error).__name__}",
                    ),
                }
        return {
            "draft_answer": answer,
            "trace": _append_trace(
                state,
                "answer_generation:"
                f"{self.llm.settings.provider}:"
                f"{self.llm.settings.model}:"
                f"{time.perf_counter() - started:.2f}s",
            ),
        }

    def response_guard_node(self, state: AquaBioState) -> dict:
        mode = state.get("response_mode", "normal")
        answer = state.get("draft_answer", "")
        if mode == "short_color_only":
            source = f"{answer}\n{state.get('final_context', '')}"
            colors = [
                color for color in COLOR_WORDS if color in source
            ]
            if colors:
                answer = "、".join(dict.fromkeys(colors)) + "。"
            else:
                answer = "现有证据未明确给出常见颜色。"
        elif mode == "short_answer":
            first_line = next(
                (
                    line.strip()
                    for line in answer.splitlines()
                    if line.strip()
                ),
                "",
            )
            answer = first_line[:120]
        return {
            "draft_answer": answer,
            "trace": _append_trace(state, f"response_guard:{mode}"),
        }

    def evaluation_node(self, state: AquaBioState) -> dict:
        answer = state.get("draft_answer", "")
        if state.get("provider_fallback"):
            context = state.get("final_context", "")
            passed = bool(answer.strip()) and bool(context.strip())
            result = EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=(
                    "模型服务超时，已使用本地检索证据生成完整降级回答。"
                    if passed
                    else "模型服务超时，且本地检索证据不足。"
                ),
                retry_target="none" if passed else "retrieval",
            )
            return {
                "evaluation_result": result.model_dump(),
                "trace": _append_trace(
                    state, f"evaluation:{passed}:provider_fallback"
                ),
            }
        if state.get("generation_failed"):
            result = EvaluationResult(
                passed=False,
                score=0.0,
                reason="Qwen API 调用失败，当前结果仅包含检索证据。",
                retry_target="none",
            )
            return {
                "evaluation_result": result.model_dump(),
                "trace": _append_trace(
                    state, "evaluation:False:provider_error"
                ),
            }

        context = state.get("final_context", "")
        if state.get("response_mode") in {
            "short_color_only",
            "short_answer",
        }:
            passed = bool(answer.strip()) and bool(context.strip())
            result = EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=(
                    "简短追问已遵守输出约束。"
                    if passed
                    else "简短追问缺少答案或检索证据。"
                ),
                retry_target="none" if passed else "retrieval",
            )
            return {
                "evaluation_result": result.model_dump(),
                "trace": _append_trace(
                    state, f"evaluation:{passed}:terse"
                ),
            }

        options = state.get("runtime_options", {})
        rag_enabled = options.get("rag_enabled", True)
        citation_enabled = options.get("citation_enabled", True)
        has_context = bool(context.strip()) or not rag_enabled
        valid_citations = re.findall(
            r"\[E\d+(?:\s*,\s*E?\d+)*\]",
            answer,
        )
        vision_failed = bool(state.get("vision_failed"))
        has_citation = (
            bool(valid_citations)
            or not citation_enabled
            or not rag_enabled
            or self.offline
            or vision_failed
        )
        has_complete_answer = (
            self.offline
            or vision_failed
            or (
                len(answer.strip()) >= 100
                and not re.search(r"\[E(?:\d+)?$", answer.rstrip())
                and answer.count("[") == answer.count("]")
            )
        )
        has_image_consistency = (
            not state["route"]["need_vlm"]
            or bool(state.get("image_caption"))
            or self.offline
        )
        score = (
            0.20 * bool(answer.strip())
            + 0.25 * has_context
            + 0.20 * has_citation
            + 0.15 * has_image_consistency
            + 0.20 * has_complete_answer
        )
        passed = (
            score >= 0.8
            and has_context
            and has_citation
            and has_complete_answer
            and has_image_consistency
        )
        retry_target = "none"
        reason = "回答具备上下文、引用和输入一致性。"
        if not has_complete_answer:
            retry_target = "answer"
            reason = "模型回答疑似被截断或引用括号不完整。"
        elif not has_context:
            retry_target = "retrieval"
            reason = "未检索到足够上下文。"
        elif not has_image_consistency:
            retry_target = "vision"
            reason = "图片任务缺少有效 caption。"
        elif not has_citation:
            retry_target = "retrieval"
            reason = "回答未引用检索证据。"
        result = EvaluationResult(
            passed=passed,
            score=score,
            reason=reason,
            retry_target=retry_target,
        )
        return {
            "evaluation_result": result.model_dump(),
            "trace": _append_trace(
                state, f"evaluation:{passed}:{score:.2f}"
            ),
        }

    def finalize_node(self, state: AquaBioState) -> dict:
        evaluation = state["evaluation_result"]
        answer = state.get("draft_answer", "")
        if not evaluation["passed"]:
            answer += (
                "\n\n系统提示：答案未完全通过自动评估。"
                + evaluation["reason"]
            )
        return {
            "final_answer": answer,
            "trace": _append_trace(state, "finalize"),
        }

    def memory_save_node(self, state: AquaBioState) -> dict:
        if not state.get("runtime_options", {}).get(
            "memory_enabled", True
        ):
            return {
                "session_file": "",
                "trace": _append_trace(state, "memory_save:disabled"),
            }
        species_ids: list[str] = []
        for source in (
            state.get("resolved_species_ids", []),
            state.get("detected_species_ids", []),
        ):
            for species_id in source:
                if (
                    species_id in self.species_by_id
                    and species_id not in species_ids
                ):
                    species_ids.append(species_id)
        # Retrieval can contain many comparison/background species. It is only
        # a fallback when no subject was resolved from conversation or vision.
        if not species_ids:
            counts: dict[str, int] = {}
            for context_name in (
                "text_context",
                "image_context",
                "multimodal_context",
                "pdf_context",
            ):
                for row in state.get(context_name, []):
                    species_id = row.get("metadata", {}).get("species_id")
                    if species_id in self.species_by_id:
                        counts[species_id] = counts.get(species_id, 0) + 1
            species_ids = [
                item[0]
                for item in sorted(
                    counts.items(),
                    key=lambda pair: pair[1],
                    reverse=True,
                )[:3]
            ]
        if not species_ids:
            species_ids = list(
                state.get("memory_summary", {}).get(
                    "last_species_ids", []
                )
            )
        species_names = [
            self.species_by_id[item]["chinese_name"]
            for item in species_ids
            if item in self.species_by_id
        ]
        evidence_ids = []
        for context_name in (
            "text_context",
            "image_context",
            "multimodal_context",
            "pdf_context",
        ):
            for row in state.get(context_name, []):
                evidence_id = row.get("id")
                if evidence_id and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        previous = state.get("memory_summary", {})
        final_answer = state.get("final_answer", "")
        saved_trace = _append_trace(
            state, f"memory_save:{len(species_ids)}"
        )
        summary = {
            "last_species_ids": species_ids[:5],
            "last_species_names": species_names[:5],
            "last_image_path": (
                state.get("image_path")
                or previous.get("last_image_path", "")
            ),
            "last_image_caption": (
                state.get("image_caption")
                or previous.get("last_image_caption", "")
            ),
            "last_answer_summary": final_answer[:500],
            "last_evidence_ids": evidence_ids[:12],
        }
        path = self.conversations.append_turn(
            state.get("session_id", "default"),
            {
                "user_query": state.get("original_query", ""),
                "resolved_query": state.get("resolved_query", ""),
                "image_path": state.get("image_path") or "",
                "assistant_answer": final_answer,
                "species_ids": species_ids[:5],
                "species_names": species_names[:5],
                "image_caption": state.get("image_caption") or "",
                "response_mode": state.get("response_mode", "normal"),
                "evidence_ids": evidence_ids[:12],
                "trace": saved_trace,
                "warnings": state.get("warnings", []),
            },
            summary,
        )
        return {
            "session_file": str(path),
            "memory_summary": summary,
            "trace": saved_trace,
        }

    def route_after_source_selection(self, state: AquaBioState) -> str:
        if state["route"].get("need_detection"):
            return "detection"
        if state["route"]["need_vlm"] and not state.get("image_caption"):
            return "vision"
        return "retrieval"

    @staticmethod
    def route_after_followup_resolution(state: AquaBioState) -> str:
        return (
            "clarification"
            if state.get("need_clarification")
            else "router"
        )

    @staticmethod
    def route_after_clarification(state: AquaBioState) -> str:
        return "router" if state.get("hitl_enabled") else "memory_save"

    def route_after_evaluation(self, state: AquaBioState) -> str:
        result = state["evaluation_result"]
        if (
            result["passed"]
            or result["retry_target"] == "none"
            or state.get("retry_count", 0) >= self.settings.max_retry
        ):
            return "finalize"
        return result["retry_target"]

    @staticmethod
    def route_after_retrieval(state: AquaBioState) -> str:
        evidence_count = sum(
            len(state.get(name, []))
            for name in (
                "text_context",
                "image_context",
                "multimodal_context",
                "pdf_context",
            )
        )
        if evidence_count == 0 and state.get("react_step", 0) < 2:
            return "react"
        return "answer"

    @staticmethod
    def increment_retry(state: AquaBioState) -> dict:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": _append_trace(state, "retry"),
        }

    def _build_retrieval_subgraph(self):
        graph = StateGraph(AquaBioState)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("build_context", self.context_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "build_context")
        graph.add_edge("build_context", END)
        return graph.compile()

    def _build_answer_subgraph(self):
        graph = StateGraph(AquaBioState)
        graph.add_node("generate", self.answer_node)
        graph.add_node("guard", self.response_guard_node)
        graph.add_node("evaluate", self.evaluation_node)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", "guard")
        graph.add_edge("guard", "evaluate")
        graph.add_edge("evaluate", END)
        return graph.compile()

    def _build(self):
        graph = StateGraph(AquaBioState)
        graph.add_node("session_init", self.session_init_node)
        graph.add_node("memory_load", self.memory_load_node)
        graph.add_node(
            "followup_resolver", self.followup_resolver_node
        )
        graph.add_node("router", self.router_node)
        graph.add_node("rewrite", self.rewrite_node)
        graph.add_node(
            "source_selection", self.source_selection_node
        )
        graph.add_node("react_tool_plan", self.react_tool_plan_node)
        graph.add_node("clarification", self.clarification_node)
        graph.add_node("vision", self.vision_node)
        graph.add_node("detection", self.detection_node)
        graph.add_node(
            "retrieval_agent", self._build_retrieval_subgraph()
        )
        graph.add_node("answer_agent", self._build_answer_subgraph())
        graph.add_node("retry", self.increment_retry)
        graph.add_node("finalize", self.finalize_node)
        graph.add_node("memory_save", self.memory_save_node)

        graph.add_edge(START, "session_init")
        graph.add_edge("session_init", "memory_load")
        graph.add_edge("memory_load", "followup_resolver")
        graph.add_conditional_edges(
            "followup_resolver",
            self.route_after_followup_resolution,
            {
                "clarification": "clarification",
                "router": "router",
            },
        )
        graph.add_conditional_edges(
            "clarification",
            self.route_after_clarification,
            {"router": "router", "memory_save": "memory_save"},
        )
        graph.add_edge("router", "rewrite")
        graph.add_edge("rewrite", "source_selection")
        graph.add_edge("source_selection", "react_tool_plan")
        graph.add_conditional_edges(
            "react_tool_plan",
            self.route_after_source_selection,
            {"vision": "vision", "retrieval": "retrieval_agent", "detection": "detection"},
        )
        graph.add_edge("vision", "retrieval_agent")
        graph.add_edge("detection", "retrieval_agent")
        graph.add_conditional_edges(
            "retrieval_agent",
            self.route_after_retrieval,
            {"react": "react_tool_plan", "answer": "answer_agent"},
        )
        graph.add_conditional_edges(
            "answer_agent",
            self.route_after_evaluation,
            {
                "finalize": "finalize",
                "retrieval": "retry",
                "rewrite": "retry",
                "vision": "retry",
                "answer": "retry",
            },
        )
        graph.add_conditional_edges(
            "retry",
            lambda state: state["evaluation_result"]["retry_target"],
            {
                "retrieval": "retrieval_agent",
                "rewrite": "rewrite",
                "vision": "vision",
                "answer": "answer_agent",
            },
        )
        graph.add_edge("finalize", "memory_save")
        graph.add_edge("memory_save", END)
        if self.offline:
            return graph.compile(checkpointer=MemorySaver())
        checkpoint_path = self.paths.sessions_dir / "langgraph.sqlite"
        self._checkpoint_connection = sqlite3.connect(
            checkpoint_path, check_same_thread=False
        )
        return graph.compile(
            checkpointer=SqliteSaver(self._checkpoint_connection)
        )

    def invoke(
        self,
        query: str = "",
        image_path: str | None = None,
        session_id: str = "default",
        hitl: bool = False,
        options: dict[str, bool] | None = None,
        extra_context: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> AquaBioState:
        if not query.strip() and not image_path:
            raise ValueError("文本和图片不能同时为空。")
        initial: AquaBioState = {
            "session_id": session_id,
            "session_initialized": False,
            "original_query": query,
            "image_path": image_path,
            "conversation_history": [],
            "memory_summary": {},
            "resolved_query": "",
            "resolved_species_ids": [],
            "detected_species_ids": [],
            "followup_detected": False,
            "need_clarification": False,
            "clarification_question": "",
            "response_mode": "normal",
            "response_constraints": [],
            "selected_tools": [],
            "tool_plan": [],
            "tool_observations": [],
            "tool_calls": [],
            "react_step": 0,
            "hitl_enabled": hitl,
            "human_review": {},
            "graph_context": [],
            "fused_context": [],
            "graph_trace": [],
            "text_context": [],
            "web_context": [],
            "image_context": [],
            "multimodal_context": [],
            "pdf_context": [],
            "final_context": "",
            "draft_answer": "",
            "generation_failed": False,
            "provider_fallback": False,
            "evaluation_result": {},
            "final_answer": "",
            "retry_count": 0,
            "runtime_options": {
                "memory_enabled": True,
                "rag_enabled": True,
                "vision_enabled": True,
                "pdf_enabled": True,
                "mcp_enabled": True,
                "mcp_retrieval_enabled": False,
                "image_search_enabled": True,
                "citation_enabled": True,
                "log_enabled": True,
                **(options or {}),
            },
            "extra_context": extra_context,
            "web_research_plan": {},
            "detection_result": {},
            "detection_comparison": {},
            "detection_visualization_path": "",
            "trace": [],
            "warnings": [],
        }
        final_state: AquaBioState = initial
        for snapshot in self.graph.stream(
            initial,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="values",
        ):
            final_state = snapshot
            trace = snapshot.get("trace", [])
            if progress_callback and trace:
                progress_callback(trace[-1])
        return final_state

    def resume(self, session_id: str, answer: str) -> AquaBioState:
        return self.graph.invoke(
            Command(resume={"answer": answer}),
            config={"configurable": {"thread_id": session_id}},
        )

    def pending(self, session_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(
            {"configurable": {"thread_id": session_id}}
        )
        return {
            "session_id": session_id,
            "next": list(snapshot.next),
            "tasks": [str(task) for task in snapshot.tasks],
            "values": snapshot.values,
        }
