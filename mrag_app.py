from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from aquabio.config import load_env
from aquabio_web.presentation import (
    TOOL_LABELS,
    evidence_source_counts,
    mcp_activity,
    selected_tools,
    trace_phase_summary,
)


ROOT = Path(__file__).resolve().parent
load_env(ROOT / ".env")
API_URL = os.getenv(
    "AQUABIO_API_URL", "http://127.0.0.1:8000"
).strip().rstrip("/")

st.set_page_config(
    page_title="AquaBio AgentRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {
        --aqua-navy: #123047;
        --aqua-blue: #167d9a;
        --aqua-cyan: #38b6c7;
        --aqua-mist: #eef8fa;
        --aqua-line: rgba(18, 48, 71, 0.12);
    }
    .stApp {
        background:
            radial-gradient(circle at 75% 0%, rgba(56,182,199,.12), transparent 30%),
            linear-gradient(180deg, #f8fcfd 0%, #ffffff 45%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        min-width: 330px;
        max-width: 330px;
        background: linear-gradient(180deg, #123047 0%, #17445b 100%);
        color: white;
        border-right: 0;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,.94);
    }
    [data-testid="stSidebar"] input {
        color: #123047 !important;
        background: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: white !important;
        color: #123047 !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] svg {
        fill: #123047 !important;
    }
    [data-testid="stSidebar"] .stSelectbox li {
        color: #123047 !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: rgba(255,255,255,.94) !important;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] {
        color: rgba(255,255,255,.94) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        text-align: left;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 12px;
        min-height: 48px;
        white-space: pre-line;
        background: rgba(255,255,255,.08);
        color: white;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(255,255,255,.45);
        background: rgba(255,255,255,.15);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 12px;
        background: rgba(255,255,255,.05);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--aqua-line);
        border-radius: 16px;
        padding: 0.6rem 0.85rem;
        background: rgba(255,255,255,.86);
        box-shadow: 0 8px 24px rgba(18,48,71,.045);
    }
    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--aqua-line);
        border-radius: 14px;
        padding: .75rem 1rem;
        background: rgba(255,255,255,.82);
    }
    .aquabio-hero {
        border: 1px solid var(--aqua-line);
        border-radius: 22px;
        padding: 1.25rem 1.45rem;
        margin: .25rem 0 1rem 0;
        color: white;
        background:
            linear-gradient(115deg, rgba(18,48,71,.98), rgba(22,125,154,.92)),
            radial-gradient(circle at right top, #38b6c7, transparent 45%);
        box-shadow: 0 18px 45px rgba(18,48,71,.14);
    }
    .aquabio-hero h1 {
        margin: 0;
        color: white;
        font-size: 2rem;
    }
    .aquabio-hero p {
        margin: .45rem 0 0;
        color: rgba(255,255,255,.82);
    }
    .flow-card {
        border: 1px solid var(--aqua-line);
        border-radius: 14px;
        padding: .75rem;
        min-height: 112px;
        background: white;
    }
    .flow-card.complete {
        border-top: 4px solid var(--aqua-cyan);
    }
    .flow-title {
        color: var(--aqua-navy);
        font-weight: 700;
        font-size: .92rem;
    }
    .flow-label {
        color: #607786;
        font-size: .78rem;
        margin-top: .25rem;
    }
    .tool-chip {
        display: inline-block;
        border: 1px solid rgba(22,125,154,.22);
        background: var(--aqua-mist);
        color: var(--aqua-blue);
        border-radius: 999px;
        padding: .25rem .55rem;
        margin: .18rem .14rem .18rem 0;
        font-size: .76rem;
        font-weight: 600;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #35c58b;
        margin-right: 6px;
    }
    .architecture-layer {
        border-left: 3px solid var(--aqua-cyan);
        border-radius: 0 10px 10px 0;
        padding: .48rem .65rem;
        margin: .4rem 0;
        background: rgba(232,247,249,.72);
        color: var(--aqua-navy);
        line-height: 1.25;
    }
    .architecture-layer div {
        color: var(--aqua-blue);
        font-size: .8rem;
        margin-top: .12rem;
    }
    .architecture-layer small {
        color: #607786;
        font-size: .72rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODE_PRESETS = {
    "标准 Agent": {
        "memory_enabled": True,
        "rag_enabled": True,
        "vision_enabled": True,
        "pdf_enabled": True,
        "image_search_enabled": True,
        "mcp_enabled": True,
        "mcp_retrieval_enabled": False,
        "citation_enabled": True,
        "log_enabled": True,
    },
    "全 MCP 模式": {
        "memory_enabled": True,
        "rag_enabled": True,
        "vision_enabled": True,
        "pdf_enabled": True,
        "image_search_enabled": True,
        "mcp_enabled": True,
        "mcp_retrieval_enabled": True,
        "citation_enabled": True,
        "log_enabled": True,
    },
    "快速文本 RAG": {
        "memory_enabled": True,
        "rag_enabled": True,
        "vision_enabled": False,
        "pdf_enabled": False,
        "image_search_enabled": False,
        "mcp_enabled": False,
        "mcp_retrieval_enabled": False,
        "citation_enabled": True,
        "log_enabled": True,
    },
    "图谱研究": {
        "memory_enabled": True,
        "rag_enabled": True,
        "vision_enabled": False,
        "pdf_enabled": True,
        "image_search_enabled": True,
        "mcp_enabled": True,
        "mcp_retrieval_enabled": False,
        "citation_enabled": True,
        "log_enabled": True,
    },
    "图像识别": {
        "memory_enabled": True,
        "rag_enabled": True,
        "vision_enabled": True,
        "pdf_enabled": True,
        "image_search_enabled": True,
        "mcp_enabled": True,
        "mcp_retrieval_enabled": False,
        "citation_enabled": True,
        "log_enabled": True,
    },
    "目标检测": {
        "memory_enabled": True,
        "rag_enabled": True,
        "vision_enabled": True,
        "pdf_enabled": False,
        "image_search_enabled": False,
        "mcp_enabled": False,
        "mcp_retrieval_enabled": False,
        "citation_enabled": True,
        "log_enabled": True,
        "detection_enabled": True,
    },
}


def api(
    method: str,
    path: str,
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> Any:
    response = requests.request(
        method, f"{API_URL}{path}", timeout=timeout, **kwargs
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    content_type = response.headers.get("content-type", "")
    return response.json() if "json" in content_type else response.content


def create_session(title: str = "新会话") -> dict[str, Any]:
    return api("POST", "/api/sessions", json={"title": title})


def ensure_state() -> None:
    st.session_state.setdefault("session_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("attachments", [])
    st.session_state.setdefault("last_result", {})
    st.session_state.setdefault("upload_nonce", 0)
    st.session_state.setdefault("agent_mode", "标准 Agent")
    st.session_state.setdefault("last_agent_mode", "")
    st.session_state.setdefault("feedback_sent", {})
    if not st.session_state.session_id:
        try:
            sessions = api("GET", "/api/sessions", timeout=10)
        except Exception:
            sessions = []
        previous = next(
            (
                item
                for item in sessions
                if int(item.get("message_count", 0)) > 0
            ),
            None,
        )
        if previous:
            load_session(previous["session_id"])
        else:
            try:
                session = create_session()
                st.session_state.session_id = session["session_id"]
            except Exception:
                st.session_state.session_id = "default"


def load_session(session_id: str) -> None:
    session = api("GET", f"/api/sessions/{session_id}")
    st.session_state.session_id = session_id
    st.session_state.messages = [
        {
            "role": item["role"],
            "content": item["content"],
            "attachments": item.get("attachments", []),
        }
        for item in session.get("messages", [])
    ]
    st.session_state.attachments = []
    st.session_state.last_result = {}


def upload_file(uploaded: Any, file_type: str) -> dict[str, Any]:
    return api(
        "POST",
        f"/api/uploads/{file_type}",
        data={"session_id": st.session_state.session_id},
        files={
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type,
            )
        },
    )


def stage_selected_files(
    uploaded_images: list[Any] | None,
    uploaded_pdfs: list[Any] | None,
) -> list[dict[str, Any]]:
    staged = list(st.session_state.attachments)
    existing = {
        (
            item.get("file_type", ""),
            item.get("file_name", ""),
            int(item.get("size_bytes", 0)),
        )
        for item in staged
    }
    for file_type, uploads in (
        ("image", uploaded_images or []),
        ("pdf", uploaded_pdfs or []),
    ):
        for uploaded in uploads:
            signature = (
                file_type,
                uploaded.name,
                len(uploaded.getvalue()),
            )
            if signature in existing:
                continue
            item = upload_file(uploaded, file_type)
            staged.append(item)
            existing.add(signature)
    st.session_state.attachments = staged
    return staged


def absolute_url(path: str) -> str:
    if not path:
        return ""
    return path if path.startswith("http") else f"{API_URL}{path}"


def render_attachment(item: dict[str, Any]) -> None:
    if item.get("file_type") == "image" or item.get("type") == "image":
        url = item.get("url") or f"/files/{item.get('file_path', '')}"
        role_labels = {
            "distribution_map": "分布图",
            "specimen_overview": "生物实例图",
            "specimen_detail": "生物细节图",
            "specimen": "生物实例图",
        }
        label = role_labels.get(
            item.get("image_role", ""), item.get("file_name", "")
        )
        name = (
            item.get("scientific_name")
            or item.get("common_name")
            or ""
        )
        caption = " · ".join(value for value in (label, name) if value)
        st.image(
            absolute_url(url),
            width=240,
            caption=caption or None,
        )
    else:
        st.caption(f"PDF：{item.get('file_name', item.get('file_id', ''))}")


def render_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(
            "assistant" if message["role"] == "assistant" else "user"
        ):
            for attachment in message.get("attachments", []):
                render_attachment(attachment)
            st.markdown(message["content"])


def render_react_flow(result: dict[str, Any]) -> None:
    phases = trace_phase_summary(result.get("trace", []))
    columns = st.columns(4, gap="small")
    for column, phase in zip(columns, phases):
        with column:
            details = "<br>".join(
                html.escape(item[:72]) for item in phase["details"][-2:]
            )
            st.markdown(
                (
                    f'<div class="flow-card {phase["status"]}">'
                    f'<div class="flow-title">{phase["phase"]}</div>'
                    f'<div class="flow-label">{phase["label"]}</div>'
                    f'<div class="flow-label">'
                    f'{phase["event_count"]} 个执行事件</div>'
                    f'<div class="flow-label">{details}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
    tools = selected_tools(result.get("trace", []))
    if tools:
        st.markdown(
            "".join(
                f'<span class="tool-chip">{html.escape(TOOL_LABELS[tool])}</span>'
                for tool in tools
            ),
            unsafe_allow_html=True,
        )


def render_feedback(result: dict[str, Any]) -> None:
    turn_id = result.get("turn_id", "")
    if not turn_id:
        return
    sent = st.session_state.feedback_sent.get(turn_id)
    if sent:
        st.caption("本轮反馈已记录，感谢你的评价。")
        return
    st.caption("这个回答是否解决了问题？")
    good, neutral, bad = st.columns(3)
    choices = (
        (good, "有帮助", 1),
        (neutral, "一般", 0),
        (bad, "需改进", -1),
    )
    for column, label, rating in choices:
        if column.button(
            label,
            key=f"feedback_{turn_id}_{rating}",
            use_container_width=True,
        ):
            api(
                "POST",
                "/api/feedback",
                json={
                    "session_id": result["session_id"],
                    "turn_id": turn_id,
                    "rating": rating,
                    "comment": "",
                },
            )
            st.session_state.feedback_sent[turn_id] = rating
            st.rerun()


try:
    health = api("GET", "/api/health", timeout=5)
    backend_online = health.get("status") == "ok"
except Exception:
    backend_online = False

if not backend_online:
    st.error(
        "FastAPI 后端未运行。请先执行：\n\n"
        "`python -m uvicorn api_app:app --host 127.0.0.1 --port 8000`"
    )
    st.stop()

ensure_state()

with st.sidebar:
    st.title("AquaBio")
    st.caption("Marine Intelligence Workspace")
    agent_mode = st.radio(
        "运行模式",
        list(MODE_PRESETS),
        horizontal=True,
        help="为不同任务预设检索器、视觉模型和 MCP 工具。",
    )
    st.session_state.agent_mode = agent_mode
    if st.session_state.get("last_agent_mode", "") != agent_mode:
        for option_name, value in MODE_PRESETS[agent_mode].items():
            st.session_state[f"option_{option_name}"] = value
        st.session_state.last_agent_mode = agent_mode
    if st.button("＋ 新建会话", use_container_width=True):
        session = create_session()
        st.session_state.session_id = session["session_id"]
        st.session_state.messages = []
        st.session_state.attachments = []
        st.session_state.last_result = {}
        st.rerun()

    search = st.text_input("搜索会话", placeholder="标题或会话 ID")
    try:
        all_sessions = api(
            "GET", "/api/sessions", params={"search": search}, timeout=10
        )
    except Exception:
        all_sessions = []
    sessions = [
        item
        for item in all_sessions
        if int(item.get("turn_count", 0)) > 0
        or item["session_id"] == st.session_state.session_id
    ]
    st.caption(f"会话记录 · {len(sessions)}")
    with st.container(height=430, border=True):
        for session in sessions:
            label = session["title"]
            if session.get("is_favorite"):
                label = f"★ {label}"
            session_col, menu_col = st.columns([0.82, 0.18], gap="small")
            with session_col:
                if st.button(
                    f"{label}\n{session.get('turn_count', 0)} 轮对话",
                    key=f"session_{session['session_id']}",
                    use_container_width=True,
                ):
                    load_session(session["session_id"])
                    st.rerun()
            with menu_col:
                with st.popover("⋯", use_container_width=True):
                    renamed = st.text_input(
                        "重命名",
                        value=session["title"],
                        key=f"rename_{session['session_id']}",
                    )
                    if st.button(
                        "保存",
                        key=f"save_rename_{session['session_id']}",
                        use_container_width=True,
                    ):
                        api(
                            "PATCH",
                            f"/api/sessions/{session['session_id']}",
                            json={"title": renamed},
                        )
                        st.rerun()
                    favorite_text = (
                        "取消收藏"
                        if session.get("is_favorite")
                        else "收藏"
                    )
                    if st.button(
                        favorite_text,
                        key=f"favorite_{session['session_id']}",
                        use_container_width=True,
                    ):
                        api(
                            "PATCH",
                            f"/api/sessions/{session['session_id']}",
                            json={
                                "is_favorite": not session.get(
                                    "is_favorite", False
                                )
                            },
                        )
                        st.rerun()

    st.divider()
    current = next(
        (
            item
            for item in sessions
            if item["session_id"] == st.session_state.session_id
        ),
        None,
    )
    if current:
        with st.expander("当前会话管理"):
            title = st.text_input("会话标题", value=current["title"])
            col_a, col_b = st.columns(2)
            if col_a.button("保存标题", use_container_width=True):
                api(
                    "PATCH",
                    f"/api/sessions/{st.session_state.session_id}",
                    json={"title": title},
                )
                st.rerun()
            favorite_label = (
                "取消收藏" if current["is_favorite"] else "收藏"
            )
            if col_b.button(favorite_label, use_container_width=True):
                api(
                    "PATCH",
                    f"/api/sessions/{st.session_state.session_id}",
                    json={"is_favorite": not current["is_favorite"]},
                )
                st.rerun()
            st.link_button(
                "导出会话 JSON",
                f"{API_URL}/api/sessions/"
                f"{st.session_state.session_id}/export",
                use_container_width=True,
            )
            if st.button(
                "删除会话",
                type="secondary",
                use_container_width=True,
            ):
                api(
                    "DELETE",
                    f"/api/sessions/{st.session_state.session_id}",
                )
                session = create_session()
                st.session_state.session_id = session["session_id"]
                st.session_state.messages = []
                st.rerun()

try:
    status = api("GET", "/api/system/status", timeout=30)
except Exception:
    status = {}
graph_status = status.get("graph", {})
model_status = status.get("model", {})
retrieval_policy = status.get("retrieval_policy", {})
st.markdown(
    """
    <div class="aquabio-hero">
      <h1>AquaBio AgentRAG</h1>
      <p>面向水下生物识别、PDF 知识图谱与多模态研究的本地 Agent 工作台</p>
    </div>
    """,
    unsafe_allow_html=True,
)
metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("运行状态", "Ready")
metric_b.metric(
    "图谱规模",
    f"{graph_status.get('graph_nodes', 0)} / {graph_status.get('graph_edges', 0)}",
    help="实体数 / 关系数",
)
metric_c.metric("模型", model_status.get("name", "qwen3.7-plus"))
metric_d.metric("当前模式", agent_mode)

chat_column, panel_column = st.columns([2.2, 1], gap="large")

with panel_column:
    st.subheader("Agent 驾驶舱")
    st.caption(
        "PDF 图片检索："
        + (
            "本地实体索引优先"
            if retrieval_policy.get("pdf_images") == "local_pdf_registry_first"
            else "状态未知"
        )
    )
    st.markdown(
        f'<span class="status-dot"></span>'
        f'{html.escape(model_status.get("provider", "qwen"))} / '
        f'{html.escape(model_status.get("name", ""))}',
        unsafe_allow_html=True,
    )
    with st.expander("能力开关", expanded=False):
        memory_enabled = st.toggle(
            "多轮记忆", key="option_memory_enabled"
        )
        rag_enabled = st.toggle("RAG 检索", key="option_rag_enabled")
        vision_enabled = st.toggle(
            "图像理解", key="option_vision_enabled"
        )
        pdf_enabled = st.toggle(
            "PDF 解析与检索", key="option_pdf_enabled"
        )
        image_search_enabled = st.toggle(
            "实体样例图片", key="option_image_search_enabled"
        )
        mcp_enabled = st.toggle(
            "MCP 工具调用", key="option_mcp_enabled"
        )
        mcp_retrieval_enabled = st.toggle(
            "MCP 检索（Chrom+RAG-Anything）",
            key="option_mcp_retrieval_enabled",
            help=(
                "启用后，文本/图片/PDF 检索全部通过 MCP 协议调用工具，"
                "可展示完整的 LangGraph Node → MCP Tool 调用链路。"
            ),
        )
        citation_enabled = st.toggle(
            "证据引用", key="option_citation_enabled"
        )
        detection_enabled = st.toggle(
            "目标检测", key="option_detection_enabled", value=True,
            help="启用 YOLOv8 目标检测，对上传图片进行水下目标检测与增强对比。",
        )
        log_enabled = st.toggle(
            "保存消息与证据", key="option_log_enabled"
        )

    with st.expander("系统架构与真实组件", expanded=False):
        architecture = api(
            "GET", "/api/system/architecture", timeout=30
        )
        for layer in architecture.get("layers", []):
            st.markdown(
                f"""
                <div class="architecture-layer">
                  <strong>{html.escape(str(layer.get("layer", "")))}</strong>
                  <div>{html.escape(str(layer.get("component", "")))}</div>
                  <small>{html.escape(str(layer.get("detail", "")))}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        workflow = architecture.get("workflow", {})
        st.caption(
            f"工作流：{workflow.get('type', 'unknown')} · "
            f"{workflow.get('pattern', 'ReAct')}"
        )

    result = st.session_state.last_result
    if result:
        st.divider()
        tab_flow, tab_evidence, tab_mcp, tab_memory = st.tabs(
            ["ReAct 流程", "证据", "MCP", "记忆"]
        )
        with tab_flow:
            render_react_flow(result)
            route = result.get("route", {})
            st.caption(
                "路由任务："
                + route.get("task_type", "unknown")
                + " · 轨迹为可审计执行事件，不是模型私有思维链。"
            )
            with st.expander("完整 LangGraph 事件"):
                for item in result.get("trace", []):
                    st.markdown(
                        f"**{item['step']}. {item['node']}**  \n"
                        f"{item['event']} · `{item['detail']}`"
                    )
        with tab_evidence:
            source_rows = evidence_source_counts(
                result.get("evidence", [])
            )
            if source_rows:
                st.bar_chart(
                    {row["source"]: row["count"] for row in source_rows}
                )
            for item in result.get("evidence", [])[:16]:
                title = (
                    f"{item['evidence_id']} · {item['title']}"
                    f" · {item.get('score', 0):.3f}"
                )
                with st.expander(title):
                    if item.get("image_url"):
                        st.image(absolute_url(item["image_url"]))
                    if item.get("page") is not None:
                        st.caption(f"PDF page {item['page']}")
                    st.write(item["content"])
                    relation = item.get("metadata", {}).get(
                        "relation_path"
                    )
                    if relation:
                        st.code(str(relation))
        with tab_mcp:
            activity = mcp_activity(
                result.get("trace", []),
                result.get("evidence", []),
            )
            mcp_a, mcp_b = st.columns(2)
            mcp_a.metric("PDF 图片命中", activity["pdf_images"])
            mcp_b.metric(
                "图谱检索",
                "已参与" if activity["graph_retrieval"] else "未使用",
            )
            if st.button("刷新 MCP 工具", use_container_width=True):
                st.session_state.mcp_tools = api(
                    "GET", "/api/mcp/tools", timeout=300
                )
            if st.session_state.get("mcp_tools"):
                st.json(st.session_state.mcp_tools)
            else:
                st.info("点击按钮读取两个 MCP Server 的真实工具列表。")
        with tab_memory:
            st.json(result.get("memory", {}))
            st.caption(
                "会话消息、证据与轨迹：web_app.sqlite；会话长期记忆："
                "data/mrag/sessions/{session_id}.json；LangGraph checkpoint："
                "data/mrag/sessions/langgraph.sqlite。"
            )
        st.divider()
        render_feedback(result)
    else:
        st.info(
            "提交问题后，这里会显示真实的 ReAct 阶段、LangGraph 节点、"
            "MCP 工具和证据来源。"
        )

with chat_column:
    render_chat()

    if not st.session_state.messages:
        st.info(
            "可直接提问，也可以上传图片识别生物、上传 PDF 临时问答，"
            "或查询已建立的 PDF 图谱。"
        )
    st.caption("快捷任务")
    quick_prompt = ""
    quick_columns = st.columns(5, gap="small")
    quick_tasks = (
        "识别上传图片中的生物，并说明判断依据",
        "说明这个生物的栖息地、分布和生活习性",
        "从 PDF 图谱中查找外观特征并给出来源页码",
        "比较刚才的生物与相似物种，并列出区别",
        "检测上传图片中的水下目标并对比增强效果",
    )
    for column, quick_task in zip(quick_columns, quick_tasks):
        if column.button(
            quick_task,
            key=f"quick_{abs(hash(quick_task))}",
            use_container_width=True,
        ):
            quick_prompt = quick_task

    attachment_tab, pdf_tab = st.tabs(["图片附件", "PDF 附件"])
    with attachment_tab:
        uploaded_images = st.file_uploader(
            "上传水下生物图片",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"image_upload_{st.session_state.upload_nonce}",
            help="支持 JPG、PNG、WEBP；图片会交给视觉模型和多模态检索。",
        )
    with pdf_tab:
        uploaded_pdfs = st.file_uploader(
            "上传临时问答 PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_upload_{st.session_state.upload_nonce}",
            help="上传文件用于本轮临时解析，不会自动写入长期图谱索引。",
        )
    upload_col, clear_col = st.columns([1, 1])
    try:
        if uploaded_images or uploaded_pdfs:
            stage_selected_files(uploaded_images, uploaded_pdfs)
    except Exception as error:
        st.error(f"附件上传失败：{error}")
    if upload_col.button("加入本轮附件", use_container_width=True):
        try:
            stage_selected_files(uploaded_images, uploaded_pdfs)
        except Exception as error:
            st.error(f"附件上传失败：{error}")
            st.stop()
        st.rerun()
    if clear_col.button("清空本轮附件", use_container_width=True):
        st.session_state.attachments = []
        st.session_state.upload_nonce += 1
        st.rerun()

    if st.session_state.attachments:
        st.caption("本轮附件")
        attachment_columns = st.columns(4)
        for index, item in enumerate(st.session_state.attachments):
            with attachment_columns[index % 4]:
                render_attachment(item)

    typed_prompt = st.chat_input(
        "输入问题，例如：给我 Jasus lalandii 的图片并说明识别特征"
    )
    prompt = typed_prompt or quick_prompt or None
    if prompt is not None:
        attachments = stage_selected_files(
            uploaded_images, uploaded_pdfs
        )
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "attachments": attachments,
            }
        )
        with st.chat_message("user"):
            for item in attachments:
                render_attachment(item)
            st.markdown(prompt)

        payload = {
            "session_id": st.session_state.session_id,
            "query": prompt,
            "attachments": [
                {"file_id": item["file_id"], "type": item["file_type"]}
                for item in attachments
            ],
            "options": {
                "memory_enabled": memory_enabled,
                "rag_enabled": rag_enabled,
                "vision_enabled": vision_enabled,
                "pdf_enabled": pdf_enabled,
                "mcp_enabled": mcp_enabled,
                "mcp_retrieval_enabled": mcp_retrieval_enabled,
                "image_search_enabled": image_search_enabled,
                "citation_enabled": citation_enabled,
                "log_enabled": log_enabled,
                "hitl_enabled": False,
                "detection_enabled": detection_enabled,
            },
        }
        with st.chat_message("assistant"):
            try:
                task = api(
                    "POST",
                    "/api/chat/tasks",
                    json=payload,
                    timeout=30,
                )
            except Exception as error:
                st.error(f"无法创建后台任务：{error}")
                st.stop()

            task_id = task["task_id"]
            status_box = st.status(
                "Agent 后台任务已启动", expanded=True
            )
            progress_text = status_box.empty()
            timeout_seconds = int(
                os.getenv("AQUABIO_CHAT_UI_TIMEOUT", "360")
            )
            result = None
            while True:
                try:
                    task = api(
                        "GET",
                        f"/api/chat/tasks/{task_id}",
                        timeout=15,
                    )
                except Exception as error:
                    status_box.update(
                        label="无法读取后台状态",
                        state="error",
                        expanded=True,
                    )
                    progress_text.error(str(error))
                    st.stop()
                elapsed = float(task.get("elapsed_seconds", 0))
                progress_text.markdown(
                    f"**阶段：{task.get('stage', '')}**  \n"
                    f"{task.get('detail', '')}  \n"
                    f"已耗时：`{elapsed:.1f}` 秒  \n"
                    f"任务 ID：`{task_id}`"
                )
                if task["status"] == "completed":
                    result = task["result"]
                    status_box.update(
                        label=f"Agent 已完成，共 {elapsed:.1f} 秒",
                        state="complete",
                        expanded=False,
                    )
                    break
                if task["status"] in {"failed", "cancelled"}:
                    status_box.update(
                        label=task.get("stage", "任务失败"),
                        state="error",
                        expanded=True,
                    )
                    progress_text.error(
                        task.get("error")
                        or task.get("detail")
                        or "后台任务未完成。"
                    )
                    st.stop()
                if elapsed >= timeout_seconds:
                    try:
                        api(
                            "DELETE",
                            f"/api/chat/tasks/{task_id}",
                            timeout=10,
                        )
                    except Exception:
                        pass
                    status_box.update(
                        label="请求超时，已停止前端等待",
                        state="error",
                        expanded=True,
                    )
                    progress_text.error(
                        f"请求已运行超过 {timeout_seconds} 秒，已向后端发送取消请求。"
                        "请检查可见的 API 后台窗口；如果正在等待大模型或 MCP 网络响应，"
                        "建议先关闭“RAG-Anything MCP”或切换到较快模式后重试。"
                    )
                    st.stop()
                time.sleep(1)

            if result.get("images"):
                image = result["images"][0]
                st.subheader("最佳匹配图片")
                source_label = {
                    "raganything_pdf_image": "PDF 原图",
                    "image_retrieval": "本地图库",
                    "wikimedia_commons": "Wikimedia Commons",
                }.get(image.get("source", ""), image.get("source", ""))
                caption_parts = [
                    image.get("scientific_name")
                    or image.get("common_name")
                    or image["image_id"],
                    source_label,
                ]
                if image.get("page") is not None:
                    caption_parts.append(f"PDF page {image['page']}")
                st.image(
                    absolute_url(image["image_url"]),
                    caption=" · ".join(
                        value for value in caption_parts if value
                    ),
                    width=420,
                )
            model = result.get("model", {})
            if model:
                st.caption(
                    "本次回答模型："
                    f"{model.get('provider', '')} / "
                    f"{model.get('name', '')}；"
                    f"实际调用：{'是' if model.get('called') else '否'}；"
                    f"本地降级：{'是' if model.get('fallback') else '否'}"
                )
            st.markdown(result["answer"])
            for warning in result.get("warnings", []):
                st.warning(warning)
            detection = result.get("detection", {})
            if detection and detection.get("detections"):
                st.subheader("目标检测结果")
                det_cols = st.columns(3)
                det_cols[0].metric("检测数量", detection.get("num_detections", 0))
                if detection["detections"]:
                    top_conf = max(
                        d["confidence"] for d in detection["detections"]
                    )
                    det_cols[1].metric("最高置信度", f"{top_conf:.2%}")
                    class_names = list(
                        {d["class_name"] for d in detection["detections"]}
                    )
                    det_cols[2].metric("检测类别", ", ".join(class_names[:3]))
                with st.expander("检测框详情"):
                    for det_item in detection["detections"]:
                        st.markdown(
                            f"- **{det_item['class_name']}** "
                            f"置信度: {det_item['confidence']:.2%} "
                            f"位置: {det_item['bbox']}"
                        )
                det_vis_url = result.get("detection_visualization_url", "")
                if det_vis_url:
                    st.image(
                        absolute_url(det_vis_url),
                        caption="目标检测可视化",
                        width=520,
                    )
                comparison = result.get("detection_comparison", {})
                if comparison.get("enhanced"):
                    st.subheader("增强-检测对比")
                    for method, data in comparison["enhanced"].items():
                        method_labels = {
                            "white_balance": "白平衡",
                            "clahe": "CLAHE",
                            "white_balance_clahe": "白平衡+CLAHE",
                            "gamma": "Gamma校正",
                        }
                        label = method_labels.get(method, method)
                        det_data = data.get("detection", {})
                        count = det_data.get("num_detections", 0)
                        conf = (
                            max(d["confidence"] for d in det_data["detections"])
                            if det_data.get("detections")
                            else 0
                        )
                        st.caption(
                            f"{label}: {count} 个目标, 最高置信度 {conf:.2%}"
                        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "attachments": [
                    {
                        **image,
                        "file_type": "image",
                        "type": "image",
                        "url": image.get("image_url", ""),
                        "file_path": image.get("image_path", ""),
                        "file_name": image.get("image_id", ""),
                    }
                    for image in result.get("images", [])
                ],
            }
        )
        st.session_state.last_result = result
        st.session_state.attachments = []
        st.session_state.upload_nonce += 1
        st.rerun()
