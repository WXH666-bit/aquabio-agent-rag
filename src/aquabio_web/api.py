from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    SessionCreate,
    SessionUpdate,
)
from .service import ChatService


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ChatService(ROOT)
app = FastAPI(
    title="AquaBio AgentRAG 水下生物智能问答 API",
    version="1.0.0",
    description=(
        "基于 LangGraph Agent + BGE-M3 + Chroma 的水下生物智能问答后端。\n\n"
        "## 功能\n"
        "- 图文联合问答（VLM 视觉理解 + RAG 检索增强）\n"
        "- YOLOv8 水下目标检测与增强对比\n"
        "- 多轮会话记忆与追问解析\n"
        "- PDF 知识图谱与实体检索\n"
        "- MCP 协议工具调用\n"
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://localhost:8510"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="健康检查", tags=["系统"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/system/status", summary="系统状态", tags=["系统"])
def system_status() -> dict:
    return SERVICE.status()


@app.post("/api/system/warmup", summary="预热模型与向量库", tags=["系统"])
def system_warmup() -> dict:
    return SERVICE.warmup()


@app.get("/api/mcp/tools", summary="MCP工具列表", tags=["系统"])
def mcp_tools() -> dict:
    return SERVICE.mcp_tools()


@app.get("/api/system/architecture", summary="系统架构信息", tags=["系统"])
def system_architecture() -> dict:
    try:
        return SERVICE.architecture()
    except Exception as error:
        return {"layers": [], "error": f"{type(error).__name__}: {error}"}


@app.get("/api/feedback/stats", summary="反馈统计", tags=["反馈"])
def feedback_stats() -> dict:
    return SERVICE.store.feedback_stats()


@app.post("/api/feedback", summary="提交反馈", tags=["反馈"])
def save_feedback(request: FeedbackRequest) -> dict:
    return SERVICE.store.save_feedback(**request.model_dump())


@app.post("/api/chat", response_model=ChatResponse, summary="同步聊天", tags=["聊天"])
def chat(request: ChatRequest) -> dict:
    try:
        return SERVICE.chat(request)
    except KeyError as error:
        raise HTTPException(404, f"附件不存在：{error}") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/chat/tasks", summary="提交异步聊天任务", tags=["聊天"])
def submit_chat_task(request: ChatRequest) -> dict:
    return SERVICE.submit_chat(request)


@app.get("/api/chat/tasks", summary="聊天任务列表", tags=["聊天"])
def list_chat_tasks() -> list[dict]:
    return SERVICE.chat_tasks()


@app.get("/api/chat/tasks/{task_id}", summary="查询聊天任务", tags=["聊天"])
def get_chat_task(task_id: str) -> dict:
    try:
        return SERVICE.chat_task(task_id)
    except KeyError as error:
        raise HTTPException(404, "聊天任务不存在") from error


@app.delete("/api/chat/tasks/{task_id}", summary="取消聊天任务", tags=["聊天"])
def cancel_chat_task(task_id: str) -> dict:
    try:
        return SERVICE.cancel_chat_task(task_id)
    except KeyError as error:
        raise HTTPException(404, "聊天任务不存在") from error


@app.post("/api/chat/stream", summary="SSE流式聊天", tags=["聊天"])
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def events():
        yield "event: node_start\ndata: {\"node\":\"agent\"}\n\n"
        try:
            result = SERVICE.chat(request)
            payload = json.dumps(result, ensure_ascii=False)
            yield f"event: final\ndata: {payload}\n\n"
        except Exception as error:
            payload = json.dumps(
                {"error": str(error)}, ensure_ascii=False
            )
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/uploads/image", summary="上传图片", tags=["上传"])
async def upload_image(
    session_id: str = Form(...), file: UploadFile = File(...)
) -> dict:
    try:
        return SERVICE.save_upload(
            session_id,
            file.filename or "image.jpg",
            await file.read(),
            "image",
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/uploads/pdf", summary="上传PDF", tags=["上传"])
async def upload_pdf(
    session_id: str = Form(...), file: UploadFile = File(...)
) -> dict:
    try:
        return SERVICE.save_upload(
            session_id,
            file.filename or "document.pdf",
            await file.read(),
            "pdf",
        )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/sessions", summary="创建会话", tags=["会话"])
def create_session(request: SessionCreate) -> dict:
    return SERVICE.store.create_session(request.title, request.session_id)


@app.get("/api/sessions", summary="会话列表", tags=["会话"])
def list_sessions(search: str = "") -> list[dict]:
    return SERVICE.store.list_sessions(search)


@app.get("/api/sessions/{session_id}", summary="获取会话", tags=["会话"])
def get_session(session_id: str) -> dict:
    try:
        return SERVICE.store.get_session(session_id)
    except KeyError as error:
        raise HTTPException(404, "会话不存在") from error


@app.patch("/api/sessions/{session_id}", summary="更新会话", tags=["会话"])
def update_session(session_id: str, request: SessionUpdate) -> dict:
    try:
        return SERVICE.store.update_session(
            session_id, **request.model_dump()
        )
    except KeyError as error:
        raise HTTPException(404, "会话不存在") from error


@app.delete("/api/sessions/{session_id}", summary="删除会话", tags=["会话"])
def delete_session(session_id: str) -> dict:
    removed = SERVICE.store.delete_session(session_id)
    SERVICE.conversations.clear(session_id)
    return {"deleted": removed, "session_id": session_id}


@app.get("/api/sessions/{session_id}/export", summary="导出会话", tags=["会话"])
def export_session(session_id: str) -> FileResponse:
    try:
        session = SERVICE.store.get_session(session_id)
    except KeyError as error:
        raise HTTPException(404, "会话不存在") from error
    export_dir = SERVICE.paths.sessions_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"{session_id}.json"
    target.write_text(
        json.dumps(session, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FileResponse(
        target,
        media_type="application/json",
        filename=target.name,
    )


@app.get(
    "/api/sessions/{session_id}/turns/{turn_id}/evidence",
    summary="获取轮次证据", tags=["会话"],
)
def turn_evidence(session_id: str, turn_id: str) -> list[dict]:
    return SERVICE.store.evidence_for_turn(session_id, turn_id)


@app.get("/api/resources/{file_type}", summary="资源列表", tags=["上传"])
def list_resources(
    file_type: str, session_id: str | None = None
) -> list[dict]:
    mapping = {"images": "image", "pdfs": "pdf"}
    if file_type not in mapping:
        raise HTTPException(404, "资源类型不存在")
    return SERVICE.store.list_attachments(
        session_id=session_id, file_type=mapping[file_type]
    )


@app.get("/files/{relative_path:path}", summary="本地文件访问", tags=["文件"])
def local_file(relative_path: str) -> FileResponse:
    target = (ROOT / relative_path).resolve()
    if ROOT not in target.parents or not target.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(target)


@app.post("/api/detection", summary="水下目标检测", tags=["检测"])
def detect_objects(
    session_id: str = Form(...), file: UploadFile = File(...)
) -> dict:
    try:
        upload = SERVICE.save_upload(
            session_id,
            file.filename or "image.jpg",
            file.read(),
            "image",
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    try:
        from aquabio.detector import YOLOv8Detector, AgentDetectionOrchestrator
        from aquabio.visualizer import draw_detections

        image_path = str(ROOT / upload["file_path"])
        detector = YOLOv8Detector()
        orchestrator = AgentDetectionOrchestrator(detector)
        output_dir = ROOT / "data" / "outputs" / "detection"
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
        vis_relative = ""
        if vis_path and Path(vis_path).exists():
            vis_relative = str(Path(vis_path).relative_to(ROOT)).replace(
                "\\", "/"
            )
        return {
            "session_id": session_id,
            "upload": upload,
            "detection": result.get("original_detection", {}),
            "enhanced_results": result.get("enhanced_results", {}),
            "best_method": result.get("best_method", "original"),
            "recommendation": result.get("recommendation", ""),
            "visualization_url": f"/files/{vis_relative}" if vis_relative else "",
        }
    except ImportError as error:
        raise HTTPException(
            503, f"目标检测依赖未安装：{error}"
        ) from error
    except Exception as error:
        raise HTTPException(500, f"检测失败：{error}") from error
