from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aquabio.agent import AquaBioAgent
from aquabio.pdf_ingest import ingest_directory
from aquabio.vector_store import LocalVectorStore


st.set_page_config(page_title="AquaBio AgentRAG", layout="wide")
st.title("AquaBio Agentic RAG")
st.caption("PDF 问答 + 水下图像候选识别 + 图像增强 + 可追踪工具链")

with st.sidebar:
    st.subheader("知识库")
    pdfs = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("写入 PDF 知识库", disabled=not pdfs):
        pdf_dir = ROOT / "data/pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in pdfs:
            (pdf_dir / uploaded.name).write_bytes(uploaded.getvalue())
        count = ingest_directory(pdf_dir, ROOT / "data/index/pdf_chunks.jsonl")
        manifest = LocalVectorStore(ROOT / "data/vector_db").build(
            ROOT / "data/knowledge", ROOT / "data/index"
        )
        st.success(
            f"已生成 {count} 个 PDF chunk，并写入 {manifest['vector_count']} 条向量。"
        )
    st.info("上传 PDF 后会自动重建持久化向量库；修改 JSONL 卡片后需手动重建。")

query = st.text_area("问题", "水下图像为什么会偏蓝？")
image = st.file_uploader("可选：上传水下图片", type=["jpg", "jpeg", "png", "webp"])

if st.button("运行 Agent", type="primary"):
    image_path = None
    if image:
        suffix = Path(image.name).suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(image.getvalue())
        temp.close()
        image_path = temp.name
    with st.spinner("正在执行检索和工具调用..."):
        state = AquaBioAgent(ROOT).run(query, image_path)
    st.subheader("回答")
    st.write(state["answer"])
    if state["warnings"]:
        for warning in state["warnings"]:
            st.warning(warning)
    left, right = st.columns(2)
    with left:
        st.subheader("工具轨迹")
        st.json(state["tool_trace"])
        if state["image_quality"]:
            st.subheader("图像质量")
            st.json(state["image_quality"])
        if state["vision_analysis"]:
            st.subheader("VLM 候选识别")
            st.json(state["vision_analysis"])
    with right:
        st.subheader("检索证据")
        for item in state["retrieval"]:
            label = item.get("source") or item.get("dataset_name") or item.get("class_name")
            if item.get("page"):
                label = f"{label} - p.{item['page']}"
            with st.expander(f"{label} | score={item['score']}"):
                st.write(item.get("content"))
        if state["enhancements"]:
            st.subheader("增强候选")
            for item in state["enhancements"]:
                st.image(item["path"], caption=item["method"])
