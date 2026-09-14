"""Answer generation, output constraints, and evidence evaluation nodes."""
from __future__ import annotations
import json
import re
import time
from .models import AquaBioState, EvaluationResult
from .cancellation import cancellable
from .workflow_utils import _append_trace, _dedupe
from aquabio_raganything.image_rag import requested_image_roles, role_matches
from aquabio_web.web_knowledge import asks_for_related_entity_image

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


class AnswerNodes:
    @cancellable
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
                        "在线模型生成失败，已使用本地检索证据生成"
                        "结构化回答。",
                    ],
                    "trace": _append_trace(
                        state,
                        f"answer:{self.llm.settings.provider}_failed:{type(error).__name__}",
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

    @cancellable
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

    @cancellable
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
                reason="在线模型 API 调用失败，当前结果仅包含检索证据。",
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
                bool(re.search(r"[。！？.!?](?:\s|\[E|$)", answer))
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

    @cancellable
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

