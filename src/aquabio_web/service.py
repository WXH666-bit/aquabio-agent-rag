from __future__ import annotations

import hashlib
import mimetypes
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.conversation import ConversationStore
from aquabio_raganything.image_rag import (
    DISTRIBUTION_ROLE,
    SPECIMEN_ROLE,
    asks_for_reference_images,
    requested_image_roles,
    role_matches,
)
from .schemas import ChatRequest
from .store import WebStore


BOOK_TITLE_BY_FILE = {
    "Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf": (
        "Field Guide to South African Offshore Marine Invertebrates"
    ),
    "FIELD IDENTIFICATION GUIDE TO THE LIVING.pdf": (
        "Field Identification Guide to the Living Marine Resources"
    ),
}


class ChatService:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.paths = MRAGPaths.from_root(self.root)
        self.paths.ensure()
        self.settings = MRAGSettings.from_env()
        self.store = WebStore(self.paths.sessions_dir / "web_app.sqlite")
        self.conversations = ConversationStore(self.paths.sessions_dir)
        self._workflows: dict[bool, Any] = {}
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._task_lock = threading.RLock()
        self._warmup_lock = threading.Lock()
        self._warmup = {
            "status": "not_started",
            "detail": "Models and vector collections are not loaded.",
            "elapsed_seconds": 0.0,
        }

    def workflow(self, offline: bool = False) -> Any:
        with self._lock:
            if offline not in self._workflows:
                from aquabio_mrag.workflow import AquaBioMRAGWorkflow

                self._workflows[offline] = AquaBioMRAGWorkflow(
                    self.paths, self.settings, offline=offline
                )
            return self._workflows[offline]

    @staticmethod
    def _single_best_image(
        images: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        if not images:
            return []
        requested_roles = requested_image_roles(query)
        eligible = images
        if requested_roles:
            matching = [
                image
                for image in images
                if any(
                    role_matches(
                        str(image.get("image_role", "")), role
                    )
                    for role in requested_roles
                )
            ]
            if not matching:
                return []
            eligible = matching

        def priority(image: dict[str, Any]) -> tuple[int, float]:
            source = str(image.get("source", "")).casefold()
            source_rank = 1
            if "raganything_pdf_image" in source or source.startswith(
                "pdf"
            ):
                source_rank = 0
            elif "wikimedia" in source or "network" in source:
                source_rank = 2
            return source_rank, -float(image.get("score", 0.0))

        return [min(eligible, key=priority)]

    @staticmethod
    def _distribution_map_notice(image: dict[str, Any]) -> str:
        source = str(image.get("source", "")).casefold()
        scientific = str(image.get("scientific_name", "")).strip()
        common = str(image.get("common_name", "")).strip()
        book = str(image.get("book_title") or image.get("source_file") or "").strip()
        page = image.get("page")
        printed = image.get("printed_page")
        name = scientific or "当前物种"
        if common:
            name += f"（英文俗名：{common}）"
        if "raganything_pdf_image" in source or source.startswith("pdf"):
            origin = f"已从本地 PDF 图片实体库显示 {name} 的分布图"
        elif "wikimedia" in source or "network" in source:
            origin = (
                "本地 PDF 图片实体库没有命中对应分布图，"
                "已从 Wikimedia Commons 获取并显示"
            )
        else:
            origin = "已从图片知识库检索并显示分布图"
        detail_parts = []
        if book:
            detail_parts.append(f"来源书籍：{book}")
        if page:
            detail_parts.append(f"PDF 页 {page}")
        if printed:
            detail_parts.append(f"印刷页 {printed}")
        detail = "；" + "，".join(detail_parts) if detail_parts else ""
        return (
            f"{origin}{detail}。"
            f"这张图对应的具体物种是 {scientific or name}，"
            "不是上位类群整体分布图。"
        )

    @staticmethod
    def _source_file_to_book_title(source_file: str) -> str:
        return BOOK_TITLE_BY_FILE.get(source_file, source_file)

    @staticmethod
    def _caption_section(caption: str, marker: str) -> str:
        if marker not in caption:
            return ""
        tail = caption.split(marker, 1)[1]
        for next_marker in (
            " Associated diagnostic features:",
            " Associated colour description:",
            " Associated size description:",
            " Associated distribution:",
            " Extracted image dimensions:",
        ):
            if next_marker != marker and next_marker in tail:
                tail = tail.split(next_marker, 1)[0]
        return tail.strip(" .")

    @staticmethod
    def _rule_translate_distribution(text: str) -> str:
        replacements = [
            ("Southern African endemic", "南部非洲特有"),
            ("South African endemic", "南非特有"),
            ("Circumglobal", "环全球分布"),
            ("Oceanic", "大洋性分布"),
            ("oceanic", "大洋性分布"),
            ("Rare endemic", "稀有特有种"),
            ("rare endemic", "稀有特有种"),
            ("West Coast from", "西海岸，水深"),
            ("West Coast, from", "西海岸，水深"),
            ("West and South Coasts of South Africa", "南非西海岸和南海岸"),
            ("West and South Coasts", "西海岸和南海岸"),
            ("both West and South Coasts", "西海岸和南海岸均有记录"),
            ("Both West and South Coasts", "西海岸和南海岸均有记录"),
            ("Both coasts", "两岸均有记录"),
            ("both coasts", "两岸均有记录"),
            ("on both coasts", "两岸均有记录"),
            ("more common on West Coast", "在西海岸更常见"),
            ("but more common on West Coast", "但在西海岸更常见"),
            ("but more common", "但更常见"),
            ("West Coast", "西海岸"),
            ("South Coast", "南海岸"),
            ("up to Port Elizabeth", "一直到伊丽莎白港一带"),
            ("between", "介于"),
            ("and", "和"),
            ("but", "但"),
            ("on both", "在"),
            ("coasts", "海岸"),
            ("from", "从"),
            ("to", "到"),
            ("but has been recorded at", "但也曾记录于"),
            ("deeper than", "深于"),
            ("where sea surface temperature is", "海表温度"),
            ("Depth from", "深度约"),
            ("Recorded from", "记录深度约"),
            ("Generally", "通常"),
            ("generally", "通常"),
            ("Usually", "通常"),
            ("usually", "通常"),
            ("usually deeper than", "通常深于"),
            ("Pelagic", "远洋/水层生活"),
            ("surface to", "从表层到"),
            ("depth", "水深"),
            ("m depth", "米水深"),
            ("m,", "米，"),
            ("m ", "米 "),
            ("m.", "米。"),
        ]
        translated = text
        for source, target in replacements:
            translated = translated.replace(source, target)
        translated = re.sub(r"\b([0-9][0-9\s\u00a0-]*)\s*m\b", r"\1米", translated)
        translated = translated.replace(" .", "。").replace(". ", "。")
        translated = translated.replace("通常 深于", "通常深于")
        translated = translated.replace("西海岸 从", "西海岸，水深")
        return translated

    def _translate_distribution(self, text: str) -> str:
        if not text:
            return ""
        rule_based = self._rule_translate_distribution(text)
        if not re.search(r"[A-Za-z]{4,}", rule_based):
            return rule_based
        try:
            translated = self.workflow().llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是海洋生物资料翻译助手。只把给定英文分布描述"
                            "忠实翻译成简洁中文，不添加新事实，不解释。"
                            "保留学名、经纬度、数字、单位和地名。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=180,
                max_continuations=0,
            ).strip()
            if translated:
                return self._rule_translate_distribution(translated)
        except Exception:
            pass
        return rule_based

    def _local_pdf_images_for_query(
        self, query: str, top_k: int = 6
    ) -> list[dict[str, Any]]:
        if not asks_for_reference_images(query):
            return []
        try:
            from aquabio_raganything.config import (
                RAGAnythingPaths,
                RAGAnythingSettings,
            )
            from aquabio_raganything.image_rag import query_pdf_images

            payload = query_pdf_images(
                RAGAnythingPaths.from_root(self.root),
                RAGAnythingSettings.from_env(),
                query,
                top_k=top_k,
                entity="",
            )
        except Exception:
            return []
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        result = []
        for row in rows:
            metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
            image_path = row.get("image_path", "")
            if not image_path:
                continue
            source_file = (
                row.get("source_file")
                or metadata.get("source_file")
                or metadata.get("doc_title", "")
            )
            result.append(
                {
                    "image_id": row.get("id") or row.get("image_id", ""),
                    "image_path": image_path,
                    "image_url": f"/files/{image_path}",
                    "caption": row.get("caption") or row.get("content", ""),
                    "scientific_name": row.get("scientific_name", ""),
                    "common_name": row.get("common_name", ""),
                    "fb_code": row.get("fb_code") or metadata.get("fb_code", ""),
                    "class": row.get("class") or metadata.get("class", ""),
                    "order": row.get("order") or metadata.get("order", ""),
                    "family": row.get("family") or metadata.get("family", ""),
                    "phylum": row.get("phylum") or metadata.get("phylum", ""),
                    "source_file": source_file,
                    "book_title": self._source_file_to_book_title(
                        source_file
                    ),
                    "page": row.get("page"),
                    "printed_page": row.get("printed_page"),
                    "score": row.get("final_score", 1.0),
                    "image_role": row.get("image_role", SPECIMEN_ROLE),
                    "source": "raganything_pdf_image",
                    "source_page": "",
                    "license": "",
                }
            )
        return result

    def _local_pdf_answer_from_image(
        self, image: dict[str, Any], query: str, candidates: list[dict[str, Any]] | None = None
    ) -> str:
        scientific = image.get("scientific_name") or "该物种"
        common = image.get("common_name") or ""
        caption = image.get("caption") or ""
        role = image.get("image_role") or ""
        page = image.get("page")
        printed = image.get("printed_page")
        book_title = image.get("book_title") or image.get("source_file") or ""
        fb_code = image.get("fb_code") or ""
        role_text = "分布图" if role == DISTRIBUTION_ROLE else "图片"
        name_text = scientific
        if common:
            name_text += f"（英文俗名：{common}）"
        page_text = []
        if book_title:
            page_text.append(f"《{book_title}》")
        if page:
            page_text.append(f"PDF 页 {page}")
        if printed:
            page_text.append(f"印刷页 {printed}")
        distribution = ""
        marker = "Associated distribution:"
        if marker in caption:
            distribution = caption.split(marker, 1)[1].split(
                " Extracted image dimensions:", 1
            )[0].strip()
        distribution_zh = self._translate_distribution(distribution)
        diagnostic = self._caption_section(
            caption, "Associated diagnostic features:"
        )
        lines = [
            f"已从本地 PDF 图片实体库找到并显示 **{name_text}** 的{role_text}。",
        ]
        if page_text:
            lines.append("来源：" + "，".join(page_text) + "。")
        if fb_code:
            lines.append(f"书中编号/缩写：{fb_code}。")
        if distribution:
            lines.append(f"分布记录：{distribution_zh}")
        if diagnostic and len(diagnostic) <= 260:
            lines.append(f"识别要点：{diagnostic}")
        if "中文名" in query:
            if common:
                lines.append(
                    "中文名：当前本地 PDF 证据没有给出权威中文名；"
                    f"英文俗名是 {common}。如果直译，可暂译为"
                    "“薄饼/鹅足海星”或“煎饼/鹅掌海星”，但这不是权威中文名。"
                )
            else:
                lines.append(
                    "中文名：当前本地 PDF 证据没有给出权威中文名。"
                )
        lines.append(f"说明：这张分布图对应的是 **{scientific}** 这个具体种。")
        extra = [
            item for item in (candidates or [])
            if item.get("image_id") != image.get("image_id")
            and role_matches(str(item.get("image_role", "")), DISTRIBUTION_ROLE)
        ][:5]
        if extra:
            lines.append("以下是库中保存的特定种类及其分布信息：")
            for item in extra:
                item_caption = item.get("caption") or ""
                item_dist = ""
                if marker in item_caption:
                    item_dist = item_caption.split(marker, 1)[1].split(
                        " Extracted image dimensions:", 1
                    )[0].strip()
                item_name = item.get("scientific_name", "")
                item_common = item.get("common_name", "")
                if item_common:
                    item_name += f"（{item_common}）"
                item_page = []
                if item.get("printed_page"):
                    item_page.append(f"印刷页 {item.get('printed_page')}")
                if item.get("page"):
                    item_page.append(f"PDF 页 {item.get('page')}")
                suffix = f"；{self._translate_distribution(item_dist)}" if item_dist else ""
                lines.append(
                    f"- {item_name}（{', '.join(item_page)}）{suffix}"
                )
        return "\n\n".join(lines)

    def warmup(self) -> dict[str, Any]:
        with self._warmup_lock:
            if self._warmup["status"] == "ready":
                return dict(self._warmup)
            started = time.perf_counter()
            self._warmup = {
                "status": "running",
                "detail": "Loading LangGraph, BGE-M3 and Chroma collections.",
                "elapsed_seconds": 0.0,
            }
            try:
                from aquabio_mrag.retrieval import RetrievalRequest

                workflow = self.workflow()
                try:
                    workflow.retriever.search(
                        RetrievalRequest(
                            query="seahorse underwater identification",
                            task_type="text_qa",
                            top_k=1,
                        )
                    )
                    workflow.retriever.image_search(
                        "seahorse underwater", top_k=1
                    )
                    workflow.retriever.multimodal_search(
                        "seahorse",
                        "underwater seahorse",
                        top_k=1,
                    )
                    workflow.retriever.pdf_search(
                        "seahorse identification", top_k=1
                    )
                except OSError:
                    pass
                self._warmup = {
                    "status": "ready",
                    "detail": "LangGraph and Chroma are ready (BGE-M3 may be unavailable, using fallback search).",
                    "elapsed_seconds": round(
                        time.perf_counter() - started, 1
                    ),
                }
            except Exception as error:
                self._warmup = {
                    "status": "failed",
                    "detail": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(
                        time.perf_counter() - started, 1
                    ),
                }
                raise
            return dict(self._warmup)

    def save_upload(
        self,
        session_id: str,
        file_name: str,
        content: bytes,
        expected_type: str,
    ) -> dict[str, Any]:
        suffix = Path(file_name).suffix.lower()
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        pdf_suffixes = {".pdf"}
        allowed = image_suffixes if expected_type == "image" else pdf_suffixes
        if suffix not in allowed:
            raise ValueError(f"不支持的文件类型：{suffix}")
        file_id = f"file_{uuid.uuid4().hex}"
        target_dir = self.paths.uploads_dir / expected_type
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{file_id}{suffix}"
        target.write_bytes(content)
        metadata: dict[str, Any] = {
            "mime_type": mimetypes.guess_type(file_name)[0] or "",
        }
        if expected_type == "pdf":
            document = fitz.open(target)
            try:
                metadata["page_count"] = document.page_count
            finally:
                document.close()
        relative = str(target.relative_to(self.root)).replace("\\", "/")
        row = self.store.save_attachment(
            session_id,
            file_id,
            expected_type,
            file_name,
            relative,
            len(content),
            metadata,
        )
        row["url"] = f"/files/{relative}"
        return row

    def _pdf_context(self, attachments: list[dict[str, Any]]) -> str:
        sections = []
        for attachment in attachments:
            if attachment["file_type"] != "pdf":
                continue
            path = self.root / attachment["file_path"]
            document = fitz.open(path)
            try:
                pages = []
                total = 0
                for page_index, page in enumerate(document):
                    text = " ".join(page.get_text("text").split())
                    if not text:
                        continue
                    excerpt = text[:4000]
                    pages.append(
                        f"[UPLOADED_PDF={attachment['file_name']}]"
                        f"[PAGE={page_index + 1}]\n{excerpt}"
                    )
                    total += len(excerpt)
                    if total >= 16000:
                        break
                sections.extend(pages)
            finally:
                document.close()
        return "\n\n".join(sections)

    def _image_path(
        self, attachment: dict[str, Any]
    ) -> str:
        uploaded = (self.root / attachment["file_path"]).resolve()
        original_name = attachment.get("file_name", "")
        if not original_name or not uploaded.is_file():
            return str(uploaded)
        uploaded_hash = hashlib.sha256(
            uploaded.read_bytes()
        ).digest()
        for candidate in self.paths.images_dir.rglob(original_name):
            if (
                candidate.is_file()
                and candidate.stat().st_size == uploaded.stat().st_size
                and hashlib.sha256(candidate.read_bytes()).digest()
                == uploaded_hash
            ):
                return str(candidate.resolve())
        return str(uploaded)

    @staticmethod
    def _trace_rows(values: list[str]) -> list[dict[str, Any]]:
        rows = []
        for index, value in enumerate(values, start=1):
            node, _, detail = value.partition(":")
            rows.append(
                {
                    "step": index,
                    "node": node,
                    "event": "completed",
                    "detail": detail or value,
                }
            )
        return rows

    @staticmethod
    def _looks_like_image_identification(query: str) -> bool:
        normalized = query.casefold()
        markers = (
            "这是什么",
            "这个是什么",
            "这张图",
            "这张图片",
            "图里",
            "是什么生物",
            "什么动物",
            "识别",
            "图片中",
            "上传图片",
            "what is this",
            "identify",
            "which organism",
        )
        if not any(marker in normalized for marker in markers):
            return False
        named_text_markers = (
            "海马",
            "海星",
            "海胆",
            "鲨鱼",
            "鳐鱼",
            "章鱼",
            "鱿鱼",
            "珊瑚",
            "水母",
            "海龟",
            "seahorse",
            "starfish",
            "sea urchin",
            "shark",
            "manta",
            "octopus",
            "squid",
            "coral",
            "jellyfish",
        )
        if any(marker in normalized for marker in named_text_markers):
            return any(
                marker in normalized
                for marker in ("图片", "图里", "上传", "这张图", "this image")
            )
        return True

    def _missing_image_response(
        self,
        request: ChatRequest,
        attachments: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        trace = [
            {
                "step": 1,
                "node": "attachment_guard",
                "event": "completed",
                "detail": "image_identification_query_without_image",
            },
            {
                "step": 2,
                "node": "api",
                "event": "completed",
                "detail": (
                    f"latency_ms={int((time.perf_counter() - started) * 1000)}"
                ),
            },
        ]
        answer = (
            "我这边没有收到可用于识别的图片附件，所以不能判断“这是什么生物”。\n\n"
            "请先在“图片附件”里选择图片，看到它出现在“本轮附件”区域后再提问；"
            "如果图片选择后仍然变红，请重新选择图片或点击“清空本轮附件”后再上传。"
        )
        response = {
            "session_id": request.session_id,
            "turn_id": turn_id,
            "answer": answer,
            "answer_type": "text",
            "images": [],
            "evidence": [],
            "trace": trace,
            "warnings": ["本轮请求没有携带图片附件，已跳过长时间 RAG/MCP 检索。"],
            "memory": {},
            "route": {
                "task_type": "image_identification",
                "need_vlm": True,
                "need_text_retrieval": False,
                "need_image_retrieval": False,
                "need_multimodal_retrieval": False,
                "need_pdf_retrieval": False,
            },
            "model": {
                "provider": "not_called",
                "name": "",
                "called": False,
                "fallback": True,
            },
            "pending_review": False,
        }
        if request.options.log_enabled:
            self.store.save_turn(
                request.session_id,
                turn_id,
                request.query,
                answer,
                attachments,
                [],
                trace,
                assistant_attachments=[],
            )
        return response

    def _evidence_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        seen = set()
        contexts = (
            ("text_context", "chroma"),
            ("web_context", "web_research"),
            ("image_context", "image_retrieval"),
            ("multimodal_context", "chroma"),
            ("pdf_context", "pdf_hybrid"),
        )
        for context_name, default_source in contexts:
            for row in state.get(context_name, []):
                row_id = row.get("id", "")
                if not row_id or row_id in seen:
                    continue
                seen.add(row_id)
                metadata = row.get("metadata", {})
                relative = metadata.get("image_path", "")
                rows.append(
                    {
                        "evidence_id": f"E{len(rows) + 1}",
                        "id": row_id,
                        "source_system": metadata.get(
                            "retrieval_source", default_source
                        ),
                        "source_type": metadata.get(
                            "source_type", context_name
                        ),
                        "title": (
                            metadata.get("scientific_name")
                            or metadata.get("title")
                            or metadata.get("doc_title")
                            or row_id
                        ),
                        "content": row.get("content", ""),
                        "score": row.get("final_score", 0.0),
                        "page": metadata.get("page"),
                        "printed_page": metadata.get("printed_page"),
                        "image_path": relative,
                        "image_url": (
                            f"/files/{relative}" if relative else ""
                        ),
                        "metadata": metadata,
                    }
                )
        if state.get("extra_context"):
            rows.insert(
                0,
                {
                    "evidence_id": "E0",
                    "id": "uploaded_pdf",
                    "source_system": "uploaded_file",
                    "source_type": "uploaded_pdf",
                    "title": "本轮上传 PDF",
                    "content": state["extra_context"][:3000],
                    "score": 1.0,
                    "page": None,
                    "image_path": "",
                    "image_url": "",
                    "metadata": {},
                },
            )
        return rows

    def chat(
        self,
        request: ChatRequest,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        print(
            f"[CHAT START] session={request.session_id} "
            f"attachments={len(request.attachments)} "
            f"query={request.query[:120]!r}",
            flush=True,
        )
        attachments = [
            self.store.get_attachment(item.file_id)
            for item in request.attachments
        ]
        image = next(
            (
                self._image_path(item)
                for item in attachments
                if item["file_type"] == "image"
            ),
            None,
        )
        started = time.perf_counter()
        if (
            not image
            and self._looks_like_image_identification(request.query)
            and not re.search(r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b", request.query)
        ):
            return self._missing_image_response(request, attachments, started)
        pdf_context = (
            self._pdf_context(attachments)
            if request.options.pdf_enabled
            else ""
        )
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        state = self.workflow().invoke(
            request.query,
            image,
            session_id=request.session_id,
            hitl=request.options.hitl_enabled,
            options=request.options.model_dump(
                exclude={"hitl_enabled"}
            ),
            extra_context=pdf_context,
            progress_callback=progress_callback,
        )
        trace = self._trace_rows(state.get("trace", []))
        trace.append(
            {
                "step": len(trace) + 1,
                "node": "api",
                "event": "completed",
                "detail": (
                    f"latency_ms={int((time.perf_counter() - started) * 1000)}"
                ),
            }
        )
        evidence = self._evidence_rows(state)
        images = [
            {
                "image_id": item["id"],
                "image_path": item["image_path"],
                "image_url": item["image_url"],
                "caption": item["content"],
                "scientific_name": item["metadata"].get(
                    "scientific_name", ""
                ),
                "common_name": item["metadata"].get("common_name", ""),
                "fb_code": item["metadata"].get("fb_code", ""),
                "class": item["metadata"].get("class", ""),
                "order": item["metadata"].get("order", ""),
                "family": item["metadata"].get("family", ""),
                "phylum": item["metadata"].get("phylum", ""),
                "source_file": item["metadata"].get("source_file", "")
                or item["metadata"].get("doc_title", ""),
                "book_title": self._source_file_to_book_title(
                    item["metadata"].get("source_file", "")
                    or item["metadata"].get("doc_title", "")
                ),
                "page": item.get("page"),
                "printed_page": item.get("printed_page")
                or item["metadata"].get("printed_page"),
                "score": item.get("score", 0.0),
                "image_role": item["metadata"].get(
                    "image_role", SPECIMEN_ROLE
                ),
                "source": item.get("source_system", ""),
                "source_page": item["metadata"].get("source_page", ""),
                "license": item["metadata"].get("license", ""),
            }
            for item in evidence
            if item.get("image_path")
        ]
        local_pdf_images = self._local_pdf_images_for_query(
            request.query, top_k=8
        )
        for image in local_pdf_images:
            if not any(
                existing.get("image_id") == image.get("image_id")
                for existing in images
            ):
                images.append(image)
        response_warnings = list(state.get("warnings", []))
        network_images_added = []
        requested_roles = requested_image_roles(request.query)
        roles_to_fetch = (
            requested_roles
            if requested_roles
            else (
                {SPECIMEN_ROLE}
                if asks_for_reference_images(request.query)
                else set()
            )
        )
        missing_roles = {
            role
            for role in roles_to_fetch
            if not any(
                role_matches(
                    str(image.get("image_role", "")), role
                )
                and (
                    "raganything_pdf_image"
                    in str(image.get("source", "")).casefold()
                    or str(image.get("source", "")).casefold().startswith(
                        "pdf"
                    )
                )
                for image in images
            )
            if not any(
                role_matches(
                    str(image.get("image_role", "")), role
                )
                for image in images
            )
        }
        if (
            missing_roles
            and request.options.image_search_enabled
            and asks_for_reference_images(request.query)
            and not state.get("need_clarification")
        ):
            from .network_images import (
                fetch_commons_images,
                preferred_taxon_query,
            )

            species_ids = (
                state.get("resolved_species_ids", [])
                or state.get("detected_species_ids", [])
            )
            species = (
                self.workflow().species_by_id.get(species_ids[0], {})
                if species_ids
                else {}
            )
            network_query = (
                preferred_taxon_query(
                    species.get("scientific_name", ""),
                    species.get("english_name", ""),
                )
                or (
                    re.search(
                        r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b",
                        request.query,
                    ).group(0)
                    if re.search(
                        r"\b[A-Z][a-z]{2,}\s+[a-z][a-z.-]{2,}\b",
                        request.query,
                    )
                    else ""
                )
                or next(
                    (
                        item["metadata"].get("scientific_name", "")
                        for item in evidence
                        if item["metadata"].get("scientific_name")
                    ),
                    "",
                )
                or request.query
            )
            cache_species_id = (
                species_ids[0]
                if species_ids
                else hashlib.sha256(
                    network_query.encode("utf-8")
                ).hexdigest()[:16]
            )
            for role in (DISTRIBUTION_ROLE, SPECIMEN_ROLE):
                if role not in missing_roles:
                    continue
                network_images, network_warnings = fetch_commons_images(
                    self.root,
                    cache_species_id,
                    network_query,
                    top_k=1,
                    image_role=role,
                )
                images.extend(network_images)
                network_images_added.extend(network_images)
                response_warnings.extend(network_warnings)
        images = self._single_best_image(images, request.query)
        network_images_added = [
            image
            for image in network_images_added
            if any(
                image.get("image_id") == selected.get("image_id")
                for selected in images
            )
        ]
        answer_type = "image_gallery" if images else "text"
        final_answer = state.get("final_answer", "")
        if (
            images
            and str(images[0].get("source", "")).casefold()
            == "raganything_pdf_image"
            and (
                "没有找到" in final_answer
                or "没有检索到" in final_answer
                or not final_answer.strip()
            )
        ):
            final_answer = self._local_pdf_answer_from_image(
                images[0], request.query, local_pdf_images
            )
            trace.append(
                {
                    "step": len(trace) + 1,
                    "node": "local_pdf_image_guard",
                    "event": "completed",
                    "detail": (
                        f"selected={images[0].get('image_id', '')};"
                        f"role={images[0].get('image_role', '')};"
                        "reason=explicit_query_override"
                    ),
                }
            )
        elif (
            images
            and str(images[0].get("source", "")).casefold()
            == "raganything_pdf_image"
            and requested_roles == {DISTRIBUTION_ROLE}
        ):
            final_answer = self._local_pdf_answer_from_image(
                images[0], request.query, local_pdf_images
            )
        if (
            requested_roles == {DISTRIBUTION_ROLE}
            and images
            and str(images[0].get("source", "")).casefold()
            != "raganything_pdf_image"
        ):
            final_answer = (
                self._distribution_map_notice(images[0])
                + "\n\n"
                + final_answer
            )
            trace.append(
                {
                    "step": len(trace) + 1,
                    "node": "image_role_guard",
                    "event": "completed",
                    "detail": (
                        "requested=distribution_map;"
                        f"selected={images[0].get('image_role', '')};"
                        f"source={images[0].get('source', '')}"
                    ),
                }
            )
        if (
            requested_roles == {DISTRIBUTION_ROLE}
            and not images
        ):
            final_answer = (
                "没有找到与当前物种匹配且经过校验的分布图。"
                "系统没有使用普通生物照片冒充分布图。\n\n"
                + final_answer
            )
        if network_images_added:
            try:
                final_answer = self.workflow().llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是 AquaBio 水下生物助手。请根据已经实际"
                                "完成的工具结果改写最终回答，使用中文，直接"
                                "回答问题。不得声称无法联网或没有图片。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "用户问题：\n"
                                f"{request.query}\n\n"
                                "先前草稿：\n"
                                f"{final_answer}\n\n"
                                "已经实际下载并将在前端画廊显示的图片：\n"
                                + "\n".join(
                                    (
                                        f"- role={image.get('image_role')}; "
                                        f"caption={image.get('caption')}; "
                                        f"source={image.get('source_page')}"
                                    )
                                    for image in network_images_added
                                )
                                + "\n\n请保留草稿中有证据支持的生物学事实，"
                                "明确说明本地缺图后已从 Wikimedia Commons "
                                "补充图片，并提示图片已显示在画廊中。"
                            ),
                        },
                    ],
                    max_tokens=1200,
                    max_continuations=0,
                )
                trace.append(
                    {
                        "step": len(trace) + 1,
                        "node": "network_image_answer_rewrite",
                        "event": "completed",
                        "detail": (
                            f"{self.workflow().llm.settings.provider}:"
                            f"{self.workflow().llm.settings.model}"
                        ),
                    }
                )
            except Exception as error:
                final_answer = (
                    "本地知识库未覆盖全部所需图片，系统已从 "
                    "Wikimedia Commons 补充分布图和生物实例图，"
                    "并已显示在图片画廊中。\n\n"
                    + final_answer
                )
                response_warnings.append(
                    "联网图片已获取，但最终答案改写失败："
                    f"{type(error).__name__}: {error}"
                )
        response = {
            "session_id": request.session_id,
            "turn_id": turn_id,
            "answer": final_answer,
            "answer_type": answer_type,
            "images": images,
            "evidence": evidence,
            "trace": trace,
            "warnings": response_warnings,
            "memory": state.get("memory_summary", {}),
            "route": state.get("route", {}),
            "model": {
                "provider": self.workflow().llm.settings.provider,
                "name": self.workflow().llm.settings.model,
                "called": self.workflow().llm.enabled,
                "fallback": bool(state.get("provider_fallback")),
            },
            "pending_review": bool(state.get("__interrupt__")),
            "detection": state.get("detection_result", {}),
            "detection_comparison": state.get("detection_comparison", {}),
            "detection_visualization_url": "",
        }
        det_vis_path = state.get("detection_visualization_path", "")
        if det_vis_path:
            try:
                vis_relative = str(Path(det_vis_path).relative_to(self.root)).replace("\\", "/")
                response["detection_visualization_url"] = f"/files/{vis_relative}"
            except ValueError:
                pass
        if request.options.log_enabled:
            assistant_attachments = [
                {
                    **image,
                    "file_type": "image",
                    "type": "image",
                    "url": image.get("image_url", ""),
                    "file_path": image.get("image_path", ""),
                    "file_name": image.get("image_id", ""),
                }
                for image in images
            ]
            self.store.save_turn(
                request.session_id,
                turn_id,
                request.query,
                response["answer"],
                attachments,
                evidence,
                trace,
                assistant_attachments=assistant_attachments,
            )
        print(
            f"[CHAT DONE] session={request.session_id} "
            f"turn={turn_id} evidence={len(evidence)} "
            f"images={len(images)} warnings={len(response['warnings'])}",
            flush=True,
        )
        return response

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def submit_chat(self, request: ChatRequest) -> dict[str, Any]:
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        task = {
            "task_id": task_id,
            "session_id": request.session_id,
            "status": "queued",
            "stage": "排队等待",
            "detail": "请求已进入后台任务队列。",
            "created_at": self._utc_now(),
            "started_at": "",
            "finished_at": "",
            "elapsed_seconds": 0,
            "result": None,
            "error": "",
            "cancel_requested": False,
        }
        with self._task_lock:
            self._tasks[task_id] = task
        thread = threading.Thread(
            target=self._run_chat_task,
            args=(task_id, request),
            daemon=True,
            name=f"aquabio-{task_id}",
        )
        thread.start()
        return self.chat_task(task_id)

    def _set_task(self, task_id: str, **values: Any) -> None:
        with self._task_lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(values)

    def _run_chat_task(
        self, task_id: str, request: ChatRequest
    ) -> None:
        started = time.monotonic()
        self._set_task(
            task_id,
            status="running",
            stage="初始化 Agent",
            detail="正在加载会话、LangGraph 和模型配置。",
            started_at=self._utc_now(),
        )
        try:
            self._set_task(
                task_id,
                stage="执行 LangGraph",
                detail=(
                    "正在进行路由、视觉分析、向量检索、MCP 调用和答案生成。"
                ),
            )
            def report(value: str) -> None:
                node, _, detail = value.partition(":")
                self._set_task(
                    task_id,
                    stage=f"LangGraph: {node}",
                    detail=detail or value,
                )

            result = self.chat(
                request,
                progress_callback=report,
            )
            if self._tasks[task_id].get("cancel_requested"):
                self._set_task(
                    task_id,
                    status="cancelled",
                    stage="已取消",
                    detail="结果已完成，但前端已取消接收。",
                    result=None,
                )
            else:
                self._set_task(
                    task_id,
                    status="completed",
                    stage="完成",
                    detail="答案、证据和执行轨迹已生成。",
                    result=result,
                )
        except Exception as error:
            print(
                f"[CHAT ERROR] task={task_id} "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            self._set_task(
                task_id,
                status="failed",
                stage="执行失败",
                detail=f"{type(error).__name__}: {error}",
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            self._set_task(
                task_id,
                finished_at=self._utc_now(),
                elapsed_seconds=round(time.monotonic() - started, 1),
            )

    def chat_task(self, task_id: str) -> dict[str, Any]:
        with self._task_lock:
            if task_id not in self._tasks:
                raise KeyError(task_id)
            task = dict(self._tasks[task_id])
        if task["status"] in {"queued", "running"}:
            started_at = task.get("started_at")
            if started_at:
                started = datetime.fromisoformat(started_at)
                task["elapsed_seconds"] = round(
                    (
                        datetime.now(timezone.utc) - started
                    ).total_seconds(),
                    1,
                )
        return task

    def chat_tasks(self) -> list[dict[str, Any]]:
        with self._task_lock:
            task_ids = list(self._tasks)
        return [
            self.chat_task(task_id)
            for task_id in reversed(task_ids)
        ]

    def cancel_chat_task(self, task_id: str) -> dict[str, Any]:
        with self._task_lock:
            if task_id not in self._tasks:
                raise KeyError(task_id)
            if self._tasks[task_id]["status"] in {"queued", "running"}:
                self._tasks[task_id]["cancel_requested"] = True
                self._tasks[task_id]["stage"] = "正在取消"
                self._tasks[task_id]["detail"] = (
                    "已停止前端等待；当前 Python 调用结束后将丢弃结果。"
                )
        return self.chat_task(task_id)

    def status(self) -> dict[str, Any]:
        from aquabio_raganything.config import RAGAnythingPaths
        from aquabio_raganything.query_adapter import index_status

        try:
            graph_status = index_status(
                RAGAnythingPaths.from_root(self.root)
            )
        except Exception:
            graph_status = {"error": "graph status unavailable"}
        return {
            "status": "running",
            "project": "AquaBio-AgentRAG",
            "backend": "FastAPI",
            "agent": "LangGraph",
            "python_runtime": {
                "executable": sys.executable,
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
                "expected_prefix": str(self.root / ".venv"),
                "uses_project_venv": Path(sys.prefix).resolve()
                == (self.root / ".venv").resolve(),
            },
            "embedding_model": self.settings.embedding_model,
            "text_collection": self.settings.collection_name,
            "mcp_servers": ["chroma", "raganything"],
            "retrieval_policy": {
                "version": "2026-06-13-local-pdf-image-first",
                "pdf_images": "local_pdf_registry_first",
                "fallback": "network_only_after_local_miss",
            },
            "graph": graph_status,
            "sessions": len(self.store.list_sessions()),
            "warmup": dict(self._warmup),
            "model": {
                "provider": "",
                "name": "",
            },
            "feedback": self.store.feedback_stats(),
        }
        try:
            result["model"] = {
                "provider": self.workflow().llm.settings.provider,
                "name": self.workflow().llm.settings.model,
            }
        except Exception:
            pass

    def architecture(self) -> dict[str, Any]:
        from .presentation import architecture_manifest

        status = self.status()
        return {
            "layers": architecture_manifest(status),
            "workflow": {
                "type": "LangGraph StateGraph",
                "pattern": "controlled ReAct + retrieval subgraph + answer subgraph",
                "checkpoint": "data/mrag/sessions/langgraph.sqlite",
                "phases": ["Thought", "Action", "Observation", "Answer"],
            },
            "retrieval": {
                "semantic": "BGE-M3 + Chroma",
                "graph": "RAG-Anything + LightRAG + NetworkX",
                "images": "PDF image entity index + local image index",
                "web": "Wikipedia API + Wikimedia Commons",
            },
            "mcp": {
                "servers": status.get("mcp_servers", []),
                "transport": {
                    "chroma": "stdio",
                    "raganything": "streamable-http",
                },
            },
        }

    def mcp_tools(self) -> dict[str, Any]:
        from aquabio_mrag.mcp_client import project_mcp_client

        client = project_mcp_client(self.root)
        result = {}
        for server in ("chroma", "raganything"):
            try:
                result[server] = {
                    "status": "available",
                    "tools": client.list_tools_sync(server),
                }
            except Exception as error:
                result[server] = {
                    "status": "unavailable",
                    "error": str(error),
                    "tools": [],
                }
        return result
