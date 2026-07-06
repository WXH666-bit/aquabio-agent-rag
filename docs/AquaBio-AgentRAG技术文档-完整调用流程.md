# AquaBio-AgentRAG 技术文档：从 RAG 建库到 LangGraph ReAct 问答完整调用流程

## 目录

1. [项目总览](#1-项目总览)
2. [系统架构分层](#2-系统架构分层)
3. [阶段一：离线 RAG 知识库构建（完整顺序）](#3-阶段一离线-rag-知识库构建完整顺序)
4. [阶段二：在线 LangGraph ReAct 问答流程](#4-阶段二在线-langgraph-react-问答流程)
5. [用户提问 → 模块调用完整链路（逐节点）](#5-用户提问--模块调用完整链路)
6. [关键数据结构](#6-关键数据结构)
7. [配置文件说明](#7-配置文件说明)
8. [启动与运行命令](#8-启动与运行命令)

---

## 1. 项目总览

**AquaBio-AgentRAG** 是一个面向水下生物识别的多模态 Agentic RAG 系统，核心能力：

- **20 类水下生物** 的多模态知识库（文本 + 图片 + 图文配对 + PDF）
- **BGE-M3 + Chroma** 向量检索，含加权重排序
- **LangGraph StateGraph** 驱动的 ReAct Agent 工作流
- **双 MCP Server**（Chroma stdio + RAG-Anything streamable-http）
- **PDF 图片实体绑定**（PymuPDF 提取 → 角色分类 → 实体名索引 → Chroma 向量库）
- **FastAPI + Streamlit** 前后端分离架构

### 代码包组织（4 个包，`src/` 下）

| 包 | 版本 | 职责 |
|---|---|---|
| `aquabio` | v0.1 | 基础层：LLM 封装（OpenRouter/Gemini/Qwen）、PDF 解析、配置加载 |
| `aquabio_mrag` | v0.2 | 核心 RAG 管道：LangGraph 工作流、多源检索器、向量库、会话管理 |
| `aquabio_raganything` | — | RAG-Anything 集成：PDF 解析编排、图片实体绑定、LightRAG 图谱、PDF 图片向量库 |
| `aquabio_web` | — | 应用层：FastAPI（20+ 端点）、ChatService 业务逻辑、Wikipedia/Commons 集成 |

---

## 2. 系统架构分层

```
┌─────────────────────────────────────────────────────┐
│  Streamlit 前端 (mrag_app.py)                        │
│  多会话聊天 UI + 图片/PDF 上传 + Agent 驾驶舱          │
└──────────────────────┬──────────────────────────────┘
                       │ POST /api/chat/tasks
┌──────────────────────▼──────────────────────────────┐
│  FastAPI 后端 (src/aquabio_web/api.py)               │
│  20+ 端点：chat / sessions / uploads / feedback       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  ChatService (src/aquabio_web/service.py)            │
│  请求预处理 → 调用 LangGraph → 证据提取 → 图片精选     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  AquaBioMRAGWorkflow (src/aquabio_mrag/workflow.py)  │
│  LangGraph StateGraph — 12 个节点的 ReAct 工作流      │
│                                                      │
│  session_init → memory_load → followup_resolver      │
│  → router → rewrite → source_selection               │
│  → react_tool_plan ⇄ vision → retrieval_agent        │
│  → answer_agent → finalize → memory_save             │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌──────────────┐
│ Retrieval │  │   MCP     │  │  LLM/VLM     │
│ Agent     │  │  Client   │  │  Clients     │
│ (文本/图片│  │ (Chroma + │  │ (OpenRouter/ │
│ /PDF/图谱)│  │ RAG-Any)  │  │  Gemini/     │
└─────┬─────┘  └─────┬─────┘  │  Qwen)       │
      │              │         └──────────────┘
┌─────▼──────────────▼───────────────────────┐
│  Chroma (BGE-M3 向量) + LightRAG 图谱        │
│  + PDF 图片 Chroma 向量 + Wikimedia Commons │
└─────────────────────────────────────────────┘
```

---

## 3. 阶段一：离线 RAG 知识库构建（完整顺序）

### 3.1 构建总览

RAG 知识库的构建分为 **7 个顺序步骤**，由 `scripts/` 目录下的脚本按编号顺序执行：

```
Step 1: 爬取物种数据（Wikipedia + WoRMS）
   ↓
Step 2: 爬取图片数据（Wikimedia Commons + iNaturalist）
   ↓
Step 3: 构建多模态文档（文本 + 图片 + 图文配对 → rag_documents_combined.jsonl）
   ↓
Step 4: 下载并解析 PDF（FAO / NOAA / IUCN → MinerU 解析 → 文本 chunk + 图片提取）
   ↓
Step 5: 构建 Chroma + BGE-M3 向量索引
   ↓
Step 6: (隐式) Book Native 构建 → PDF 图片实体绑定 → PDF 图片 Chroma 索引
   ↓
Step 7: 系统预热（第一次启动时自动执行）
```

### 3.2 Step 1：爬取物种数据

**脚本**：`scripts/01_crawl_wikipedia_worms.py`

**输入**：`data/species_list.json`（20 类水下生物种子列表）

```json
{
  "species_id": "starfish",
  "english_name": "Starfish",
  "chinese_name": "海星",
  "scientific_name": "Asteroidea",
  "wiki_title": "Starfish",
  "worms_name": "Asteroidea",
  "commons_query": "starfish underwater"
}
```

**数据源**：
- **Wikipedia REST API** → 百科摘要、外观描述、栖息环境、生态行为
- **WoRMS REST API** → 海洋物种分类层级（门/纲/目/科/属 + AphiaID）

**输出**：
```
data/mrag/raw/wikipedia_records.jsonl
data/mrag/raw/worms_records.jsonl
```

**调用链**：
```
species_list.json
  → requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}")
  → requests.get("https://www.marinespecies.org/rest/AphiaRecordsByName/{worms_name}")
  → 写入 JSONL
```

### 3.3 Step 2：爬取图片数据

**脚本**：`scripts/02_crawl_commons_images.py` → `scripts/02b_fill_inaturalist_images.py`

**输入**：Step 1 输出的物种列表

**数据源**：
- **Wikimedia Commons API** → 公开授权图片（CC BY-SA / CC BY / Public Domain）
- **iNaturalist API**（补充）→ 观测记录图片

**处理流程**：
```
Wikimedia Commons 查询 "starfish underwater"
  → 获取图片 URL、作者、许可证、来源页面
  → 下载图片到 data/mrag/images/{species_id}/
  → 保存 raw_commons_images.jsonl
```

**输出**：
```
data/mrag/images/
  ├── starfish/
  │   ├── img_starfish_001.jpg
  │   ├── img_starfish_002.jpg
  │   └── ...
  ├── sea_urchin/
  └── ...（共 20 个物种子目录）

data/mrag/raw/commons_records.jsonl
```

### 3.4 Step 3：构建多模态文档

**脚本**：`scripts/03_build_multimodal_documents.py`

**这是核心构建步骤**，从原始数据生成结构化的 RAG 文档。包含 **5 个子步骤**：

#### 子步骤 3a：构建物种卡片 `species_cards.jsonl`

```
输入：wikipedia_records.jsonl + worms_records.jsonl
处理：
  1. 按 species_id 聚合 Wikipedia 文本和 WoRMS 分类
  2. 提取 overview / visual_summary / habitat_summary
  3. 补充 similar_species / identification_tips
输出：data/mrag/knowledge/species_cards.jsonl
```

每物种 1 条，字段含：`species_id`, `chinese_name`, `scientific_name`, `overview`, `visual_summary`, `habitat_summary`, `similar_species`, `identification_tips`

示例：
```json
{
  "id": "card_starfish",
  "source_type": "species_card",
  "species_id": "starfish",
  "english_name": "Starfish",
  "chinese_name": "海星",
  "scientific_name": "Asteroidea",
  "overview": "海星是一类常见的海洋棘皮动物，通常具有放射状身体结构...",
  "visual_summary": "通常呈星形或放射状，有五条或更多腕足...",
  "habitat_summary": "常见于潮间带、浅海海底、岩石、珊瑚礁和沙地附近..."
}
```

#### 子步骤 3b：构建文本知识 chunk `species_text_docs.jsonl`

```
输入：wikipedia_records.jsonl + species_cards.jsonl
处理：
  1. 将每物种文本按主题拆分为 5-8 个 chunk
  2. 标注 chunk_type：overview / taxonomy / visual_features /
     habitat / ecology_behavior / similar_species / image_recognition_tips
  3. 补充 keywords + source_url
输出：data/mrag/knowledge/species_text_docs.jsonl
```

每种 chunk_type 示例：
```json
{
  "id": "text_starfish_visual_features",
  "source_type": "species_text_chunk",
  "species_id": "starfish",
  "chunk_type": "visual_features",
  "content": "海星在水下图像中通常表现为星形或放射状结构...",
  "keywords": ["starfish", "sea star", "海星", "放射状"],
  "source_url": "https://en.wikipedia.org/wiki/Starfish"
}
```

#### 子步骤 3c：构建图片知识 `image_docs.jsonl`

```
输入：downloaded images + commons_records.jsonl
处理：
  1. 读取每张图片的本地路径
  2. 生成 caption（模板或 VLM）
  3. 提取 visual_keywords
  4. 构造 embedding_text（用于后续向量化）
输出：data/mrag/knowledge/image_docs.jsonl
```

```json
{
  "id": "img_starfish_001",
  "source_type": "image_doc",
  "species_id": "starfish",
  "image_path": "data/mrag/images/starfish/img_001.jpg",
  "caption": "图像中是一只位于海底岩石附近的海星，主体呈放射状结构...",
  "visual_keywords": ["star-shaped", "radial arms", "seabed"],
  "embedding_text": "starfish 海星 underwater radial arms star-shaped body",
  "license": "CC BY-SA 4.0"
}
```

#### 子步骤 3d：构建图文配对 `multimodal_pairs.jsonl`

```
输入：species_text_docs.jsonl + image_docs.jsonl + species_cards.jsonl
处理：
  1. 遍历每张图片，根据 species_id 找到对应物种文本
  2. 选取 overview + visual_features + habitat 的 chunk
  3. 组合成 rag_context
  4. 生成 embedding_text
输出：data/mrag/knowledge/multimodal_pairs.jsonl
```

```json
{
  "id": "pair_starfish_001",
  "source_type": "multimodal_pair",
  "species_id": "starfish",
  "image_id": "img_starfish_001",
  "text_ids": ["text_starfish_overview", "text_starfish_visual_features"],
  "rag_context": "海星通常具有星形或放射状身体结构...",
  "embedding_text": "starfish 海星 放射状 腕足 海底岩石"
}
```

#### 子步骤 3e：合并为统一 RAG 文档

```
输入：species_cards.jsonl + species_text_docs.jsonl + image_docs.jsonl + multimodal_pairs.jsonl
处理：
  1. 统一为规范化的文档格式
  2. 每条包含 id / source_type / species_id / modality / content / embedding_text / metadata
输出：data/mrag/knowledge/rag_documents_combined.jsonl
```

统一格式：
```json
{
  "id": "text_starfish_visual_features",
  "source_type": "species_text_chunk",
  "species_id": "starfish",
  "modality": "text",
  "content": "海星在水下图像中通常表现为星形或放射状结构...",
  "embedding_text": "Starfish 海星 visual features 放射状 腕足 水下识别",
  "metadata": {
    "chunk_type": "visual_features",
    "source_url": "https://en.wikipedia.org/wiki/Starfish"
  }
}
```

**modality 分类**：
- `text` — 物种文本知识（species_text_chunk）
- `image_caption` — 图片 caption（image_doc）
- `image_text_pair` — 图文配对（multimodal_pair）
- `species_card` — 物种总卡片

### 3.5 Step 4：下载并解析 PDF

**脚本**：`scripts/04_download_parse_pdfs.py`

**输入**：`configs/pdf_sources.json`（5 个 PDF 的 URL 和元数据）

```json
[
  {
    "doc_id": "fao_species_catalogue",
    "title": "FAO Species Catalogue",
    "url": "https://...",
    "type": "field_guide",
    "tags": ["taxonomy", "identification"]
  }
]
```

**处理流程**：

```
下载 PDF 到 data/mrag/pdfs/
  ↓
MinerU 解析引擎（/api/v1/parse）
  → 提取文本内容（按页分段）
  → 提取图片与表格
  → 生成 Markdown 格式输出
  ↓
输出到 data/mrag/raganything/parser_output/{book_id}/
```

**输出**：
```
data/mrag/pdfs/*.pdf                          # 原始 PDF
data/mrag/raganything/parser_output/          # MinerU 解析结果
  ├── sa_invertebrates/
  │   ├── *.md                                # Markdown 文本
  │   └── images/                             # 提取的图片
  └── ...
```

### 3.6 Step 5：构建 Chroma + BGE-M3 向量索引

**脚本**：`scripts/05_build_chroma_bge.py`

**这是向量化的核心步骤**。由 `src/aquabio_mrag/vector_db.py` 中的 `ChromaMRAGStore` 执行。

**Embedding 模型**：`BAAI/bge-m3`（支持中英混合，1024 维）

**处理流程**：

```
rag_documents_combined.jsonl（约 220-300 条）
  ↓
逐条读取 embedding_text
  ↓
BGE-M3 encode_documents() → 生成 1024 维向量
  ↓
分批写入 Chroma collection: aquabio_mrag
  ↓
每条保存：id / document(content) / embedding / metadata
```

**metadata 过滤字段**（留给检索阶段使用）：
```
source_type, species_id, modality, chunk_type, image_path, source_url
```

**Chroma Collection 清单**：

| Collection | 存储内容 | Embedding 模型 |
|---|---|---|
| `aquabio_mrag` | 全部 RAG 文档（文本/图片/图文配对/卡片） | BGE-M3 |
| `aquabio_pdf_chunks` | PDF 文本 chunk | BGE-M3 |
| `aquabio_pdf_images` | PDF 提取图片的 caption | BGE-M3 |

### 3.7 Step 6：PDF 图片实体绑定（Book Native 管道）

这一步构建 PDF 图片的完整索引（由 RAG-Anything 子系统完成）。

**模块**：`src/aquabio_raganything/book_native.py` → `image_rag.py`

**处理流程**：

```
┌─────────────────────────────────────────────────┐
│  1. Book Native 构建                              │
│  book_native.py                                 │
│  输入：PDF 文件 + MinerU 解析结果                   │
│  处理：                                           │
│    - 按章节/VII 级分类分割 PDF                      │
│    - 提取每个分类单元（taxon）的页面范围              │
│    - 聚合文本描述、分类信息、分布信息                  │
│  输出：                                           │
│    data/mrag/raganything/book_native/{book_id}/   │
│    ├── species_page_units.jsonl  ← 分类单元-页码绑定 │
│    └── rag_chunks.jsonl          ← 文本 chunk      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  2. PDF 图片资产构建                               │
│  image_rag.build_pdf_image_assets()              │
│  输入：species_page_units.jsonl                   │
│  处理：                                           │
│    - 用 PymuPDF (fitz) 从 PDF 逐页提取嵌入图片       │
│    - 根据 bbox 位置自动标注图片角色：                 │
│      · top < 250px  → distribution_map（分布图）   │
│      · left < 250px → specimen_overview（全标本图） │
│      · 其他         → specimen_detail（细节图）     │
│    - 绑定图片到所属分类单元（entity）                 │
│    - 生成上下文 caption（含学名、特征、分布等）       │
│    - 计算 SHA256 去重                             │
│  输出：                                           │
│    data/mrag/raganything/extracted_assets/        │
│      {book_id}/images/*.jpeg    ← 提取的图片       │
│      {book_id}/image_index/                       │
│        ├── pdf_image_captions.jsonl  ← 图片 caption │
│        ├── linked_pdf_images.jsonl   ← 图片-实体绑定 │
│        └── pdf_image_rag_docs.jsonl  ← RAG 文档格式 │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  3. PDF 图片向量索引构建                            │
│  PDFImageVectorStore.build()                     │
│  输入：pdf_image_rag_docs.jsonl                   │
│  处理：                                           │
│    - BGE-M3 encode_documents(embedding_text)      │
│    - 写入 Chroma collection: aquabio_pdf_images   │
│  metadata 含：entity_names, scientific_name,      │
│    common_name, image_role, page, printed_page    │
└─────────────────────────────────────────────────┘
```

**图片角色自动分类逻辑**（`image_rag.classify_image_role()`）：

```python
def classify_image_role(image):
    left, top, _, _ = image["bbox"]
    if top < 250:
        return "distribution_map"    # 分布图（页面上方）
    if left < 250:
        return "specimen_overview"   # 全标本图（页面左侧）
    return "specimen_detail"         # 细节图
```

**关系图谱构建**（存入 LightRAG）：
```
taxon:asteroidea → has_distribution_map → image:sa_taxon_xxx_img_01
image:sa_taxon_xxx_img_01 → maps_distribution_of → taxon:asteroidea
image:sa_taxon_xxx_img_01 → located_on_page → page:406
```

### 3.8 Step 7：系统预热

第一次启动时，`POST /api/system/warmup` 自动执行：

```
1. 加载 BGE-M3 模型到内存
2. 打开 Chroma 全部 collection
3. 执行一次样例查询（seahorse underwater）
4. 确认 LangGraph checkpoint 数据库可用
```

---

## 4. 阶段二：在线 LangGraph ReAct 问答流程

### 4.1 LangGraph StateGraph 总览

整个工作流是一个 **12 节点、7 条件分支** 的 StateGraph，存储在 `src/aquabio_mrag/workflow.py` 中：

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ session_init│  初始化会话 ID
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ memory_load │  加载会话历史（最近 8 轮）
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │ followup_resolver│ 解析追问指代（"它/这个生物"）
                    └──────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │ need_clarification?      │
              │ (追问无上一轮物种时)       │
              └────┬──────────────┬─────┘
                   │ Yes          │ No
              ┌────▼────┐   ┌─────▼──────┐
              │clarify  │   │   router    │ 判断任务类型
              └────┬────┘   └─────┬──────┘
                   │              │
                   │        ┌─────▼──────┐
                   │        │  rewrite    │ 改写检索查询
                   │        └─────┬──────┘
                   │              │
                   │        ┌─────▼─────────┐
                   │        │source_selection│ 选择检索源和工具
                   │        └─────┬─────────┘
                   │              │
                   │        ┌─────▼──────────┐
                   │        │ react_tool_plan │ 规划工具调用顺序
                   │        └─────┬──────────┘
                   │              │
                   │     ┌────────▼─────────┐
                   │     │ need_vlm?         │
                   │     └──┬──────────┬────┘
                   │        │ Yes      │ No
                   │   ┌────▼───┐     │
                   │   │ vision │     │
                   │   └────┬───┘     │
                   │        │         │
                   │   ┌────▼─────────▼────┐
                   │   │  retrieval_agent   │ 子图：检索 → 重排 → 构建上下文
                   │   │  (subgraph)        │
                   │   └────┬───────────────┘
                   │        │
                   │   ┌────▼──────────┐
                   │   │ 证据=0 且      │
                   │   │ react_step<4?  │
                   │   └──┬────────┬───┘
                   │      │ Yes    │ No
                   │      │ ┌──────▼──────┐
                   │      └►│ back to     │ 回到 react_tool_plan 重试
                   │        │ react_tool  │
                   │        └─────────────┘
                   │              │
                   │        ┌─────▼─────────┐
                   │        │ answer_agent   │ 子图：生成 → 守卫 → 评估
                   │        │ (subgraph)     │
                   │        └─────┬─────────┘
                   │              │
                   │     ┌────────▼─────────┐
                   │     │ evaluation        │
                   │     │ passed?           │
                   │     └──┬──────────┬────┘
                   │        │ No       │ Yes
                   │   ┌────▼───┐     │
                   │   │ retry  │     │ (最多 max_retry 次)
                   │   └───┬────┘     │
                   │       │          │
                   │       └──────────┘
                   │              │
                   │        ┌─────▼──────┐
                   └───────►│  finalize   │
                            └─────┬──────┘
                                  │
                            ┌─────▼──────┐
                            │ memory_save │ 保存本轮对话
                            └─────┬──────┘
                                  │
                            ┌─────▼──────┐
                            │    END     │
                            └────────────┘
```

### 4.2 ReAct 模式说明

本项目实现了 **受控 ReAct（Controlled ReAct）** 模式，区别于原生 LLM 驱动的 ReAct：

**确定性路由（Deterministic）**：
- 工具选择由路由规则 + 任务类型决定，不依赖 LLM 输出
- 这保证了可审计性和可预测性

**原生函数调用模式（Native Function Calling，可选）**：
- 通过 `MRAG_REACT_NATIVE=true` 环境变量启用
- LLM 可以自主选择工具调用顺序
- 但 `vlm_caption` 和 `pdf_image_retriever` 是强制工具，LLM 不可移除

```python
# react_tool_plan_node 的核心逻辑
if MRAG_REACT_NATIVE and llm.enabled:
    selected = react_planner.plan(query, plan, observations, step)
else:
    # 确定性路由：按 selected_tools 顺序执行
    plan = routed_tools

# 强制工具不可被 LLM 移除
for required in ("vlm_caption", "pdf_image_retriever"):
    if required in routed_tools and required not in plan:
        plan.append(required)
```

### 4.3 Router 节点：任务类型判断

路由规则（`workflow.py:287-346`）：

| 条件 | 任务类型 |
|---|---|
| 追问指代 + 有历史物种 | `followup_text_qa` |
| 有图片 + 有文本 | `multimodal_qa` |
| 有图片 + 无文本 | `image_qa` |
| 含"依据/来源/出处/source" | `source_trace` |
| 含"pdf/文档/手册/报告/fao/noaa" | `pdf_qa` |
| 含"区别/区分/比较/不同/versus" | `comparison_qa` |
| 其他纯文本 | `text_qa` |

路由输出决定后续的工具选择：

```python
RouteDecision(
    task_type="text_qa",
    need_vlm=False,                          # 无需视觉模型
    need_text_retrieval=True,                # 需要文本检索
    need_image_retrieval=False,              # 不需要图片检索
    need_multimodal_retrieval=False,         # 不需要多模态检索
    need_pdf_retrieval=True,                 # 需要 PDF 检索
)
```

### 4.4 Source Selection 节点：选择检索工具

根据 `RouteDecision` 和用户查询内容，选择可用的检索工具：

| 工具 | 触发条件 |
|---|---|
| `vlm_caption` | `need_vlm=True`（用户上传了图片） |
| `text_retriever` | `need_text_retrieval=True` |
| `image_retriever` | `need_image_retrieval=True` |
| `pdf_image_retriever` | `need_image_retrieval=True` **且** `asks_for_reference_images(query)=True` |
| `multimodal_retriever` | `need_multimodal_retrieval=True` |
| `pdf_retriever` | `need_pdf_retrieval=True` 且用户没有在要参考图 |

### 4.5 Retrieval 子图（3 节点）

```
┌──────────────┐     ┌──────────┐     ┌─────────────────┐
│   retrieve   │ →→→ │  rerank  │ →→→ │  build_context  │
│ (多路检索)    │     │ (加权排序) │     │ (组织最终上下文)  │
└──────────────┘     └──────────┘     └─────────────────┘
```

**retrieve 节点** 同时执行 5 路检索：

```
1. text_retriever     → Chroma 文本检索 (species_card + species_text_chunk)
2. image_retriever    → Chroma 图片检索 (image_doc + multimodal_pair)
3. pdf_image_retriever → PDF 图片实体检索（本地注册表 → Chroma 向量 → MCP）
4. multimodal_retriever → Chroma 多模态检索（caption + query 联合）
5. pdf_retriever      → PDF 混合检索（Chroma + BM25 + LightRAG 图谱 MCP）
```

**加权重排序公式**（`retrieval.py:164-171`）：

```python
final_score = (
    0.45 * semantic_similarity    # BGE-M3 余弦相似度
    + 0.20 * species_match        # 是否匹配候选物种
    + 0.15 * source_weight        # 知识源类型权重
    + 0.10 * chunk_weight         # chunk 类型权重
    + 0.10 * keyword_overlap      # 关键词重叠度
)
```

**知识源类型权重**（随任务类型变化）：

| 任务类型 | 最高权重源 |
|---|---|
| `text_qa` | `species_text_chunk: 1.0`, `species_card: 0.9` |
| `image_qa` | `multimodal_pair: 1.0`, `image_doc: 0.95` |
| `comparison_qa` | `species_text_chunk: 1.0`, `similar_species chunk: 1.0` |
| `pdf_qa` | `pdf_chunk: 1.0`, `pdf_figure: 0.85` |

### 4.6 Answer 子图（3 节点）

```
┌──────────────┐     ┌──────────┐     ┌─────────────┐
│   generate   │ →→→ │  guard   │ →→→ │  evaluate   │
│ (LLM 生成答案) │     │ (回答守卫) │     │ (质量评估)   │
└──────────────┘     └──────────┘     └─────────────┘
```

**generate 节点** 构造完整的 LLM prompt，包含：

```python
payload = {
    "task_type": "...",
    "original_query": "用户原始问题",
    "resolved_species_ids": ["已解析的物种"],
    "conversation_history": [...],           # 最近 8 轮对话
    "image_caption": "...",                  # VLM 生成的图片描述
    "image_analysis_available": True/False,
    "evidence": "[E1] ...\n[E2] ...",       # 检索证据（最多 24 条）
    "retrieved_images": [...],               # 检索到的图片信息
    "strict_instructions": [                 # 严格指令
        "只依据图片 caption、会话实体和检索证据回答",
        "普通回答的关键结论使用[E编号]引用",
        "使用中文回答",
        "如果是分布图请求，只有 image_role==distribution_map 才能称为分布图",
    ]
}
```

**evaluate 节点** 打分维度（`workflow.py:1329-1450`）：

```python
score = (
    0.20 * has_answer         # 是否非空
    + 0.25 * has_context      # 是否有检索上下文
    + 0.20 * has_citation     # 是否有 [E编号] 引用
    + 0.15 * image_consistency # 图片任务是否有 caption
    + 0.20 * complete_answer  # 是否完整（≥100 字符，括号闭合）
)
passed = score >= 0.8 and has_context and has_citation and ...
```

### 4.7 回退机制

```
evaluation_failed → retry_target:
  "retrieval" → 回到 retrieval 节点，扩大 top_k
  "rewrite"   → 回到 rewrite 节点，重新改写
  "vision"    → 回到 vision 节点，换 prompt
  "answer"    → 回到 answer 节点，重新生成
  "none"      → 直接 finalize（即使不合格也不重试）

retry_count >= max_retry → 强制 finalize
```

---

## 5. 用户提问 → 模块调用完整链路

以具体问题 **"海星和海胆怎么区分？"** 为例，展示从用户输入到返回答案的每一步。

### 5.1 前端层

```
用户输入："海星和海胆怎么区分？"
  │
  ▼
mrag_app.py:818-836  st.chat_input 捕获输入
  │
  ▼
mrag_app.py:838-855  打包 payload:
  {
    "session_id": "abc123",
    "query": "海星和海胆怎么区分？",
    "attachments": [],
    "options": { "image_search_enabled": true, "pdf_enabled": true, ... }
  }
  │
  ▼
mrag_app.py:859  POST /api/chat/tasks  →  返回 task_id
  │
  ▼  轮询 GET /api/chat/tasks/{task_id} 直到 completed
  │
  ▼
mrag_app.py:942-963  收到 result  →  st.markdown(result["answer"])
```

### 5.2 API 层

```
POST /api/chat/tasks
  │
  ▼
src/aquabio_web/api.py: submit_chat()
  │
  ▼
src/aquabio_web/service.py:782  ChatService.submit_chat()
  → 创建后台线程 → _run_chat_task()
  │
  ▼
src/aquabio_web/service.py:841  ChatService.chat(request)
  → 预处理附件 → 调用 LangGraph
```

### 5.3 LangGraph 工作流逐节点执行

#### 节点 1：session_init（`workflow.py:167-175`）

```
规范化 session_id
→ state["session_initialized"] = True
→ trace: "session_init:{session_id}"
```

#### 节点 2：memory_load（`workflow.py:177-206`）

```
从 data/mrag/sessions/{session_id}.json 加载最近 8 轮对话
→ state["conversation_history"] = [...]
→ state["memory_summary"] = { last_species_ids: [...], ... }
```

#### 节点 3：followup_resolver（`workflow.py:208-285`）

```
query = "海星和海胆怎么区分？"
→ 检查 FOLLOWUP_MARKERS ("刚才/上次/它/这个生物...")：无匹配
→ followup = False
→ 检查 COMPARISON_MARKERS ("区别/区分/比较...")：不在此处处理

→ state["resolved_species_ids"] = _normalize_species_ids(["海星", "海胆"])
  → species_aliases 中查找 "海星" → "starfish"
  → species_aliases 中查找 "海胆" → "sea_urchin"
→ state["resolved_query"] = "海星和海胆怎么区分？"
```

#### 节点 4：router（`workflow.py:287-346`）

```
query = "海星和海胆怎么区分？"
→ followup=False, has_image=False
→ 检查 COMPARISON_MARKERS: "区别"/"区分"/"比较" 命中
→ task = "comparison_qa"

RouteDecision:
  task_type = "comparison_qa"
  need_vlm = False
  need_text_retrieval = True
  need_image_retrieval = True
  need_multimodal_retrieval = False
  need_pdf_retrieval = True
```

#### 节点 5：rewrite（`workflow.py:348-375`）

```
原始 query: "海星和海胆怎么区分？"
→ task == "comparison_qa"
→ 追加: "。重点检索视觉特征、身体结构、相似物种、image_recognition_tips 和权威 PDF 描述。"

rewritten_query = "海星和海胆怎么区分？。重点检索视觉特征、身体结构、相似物种、image_recognition_tips 和权威 PDF 描述。"
```

#### 节点 6：source_selection（`workflow.py:377-404`）

```
route.need_text_retrieval=True  → tools += ["text_retriever"]
route.need_image_retrieval=True → tools += ["image_retriever"]
route.need_pdf_retrieval=True   → tools += ["pdf_retriever"]

selected_tools = ["text_retriever", "image_retriever", "pdf_retriever"]
```

#### 节点 7：react_tool_plan（`workflow.py:406-447`）

```
plan = ["text_retriever", "image_retriever", "pdf_retriever"]
→ 按顺序执行
```

#### 节点 8：vision（`workflow.py:485-599`）

```
需要 vlm? → route.need_vlm = False → 跳过
→ trace: "vision:skipped"
```

#### 节点 9：retrieval（子图，`workflow.py:601-976`）

##### 9a. retrieve（多路并行检索）

**路 1：text_retriever**
```
MultiSourceRetriever.search(
    RetrievalRequest(
        query="海星和海胆怎么区分？。重点检索视觉特征...",
        task_type="comparison_qa",
        top_k=12,
        species_ids=["starfish", "sea_urchin"],
        source_types=["species_card", "species_text_chunk"]
    )
)
  ↓
ChromaMRAGStore.query()
  → BGE-M3 encode_query(query)
  → Chroma collection: aquabio_mrag
  → where: { "$and": [
        {"species_id": {"$in": ["starfish", "sea_urchin"]}},
        {"source_type": {"$in": ["species_card", "species_text_chunk"]}}
    ]}
  → 返回 top_k*2=24 条候选
  ↓
加权重排序（comparison_qa 权重）:
  similar_species chunk: 1.0
  visual_features chunk: 0.95
  image_recognition_tips chunk: 0.9
  species_text_chunk source: 1.0
  species_card source: 0.85
  ↓
返回 top_k=12 条
```

**路 2：image_retriever**
```
MultiSourceRetriever.image_search(
    caption="海星和海胆怎么区分？",
    top_k=12,
    species_ids=["starfish", "sea_urchin"]
)
  → 检索 image_doc + multimodal_pair
  → 返回相关图片 caption 记录
```

**路 3：pdf_retriever**
```
RetrievalAgent.search_pdf(
    query="海星和海胆怎么区分？...",
    top_k=12,
    species_ids=["starfish", "sea_urchin"],
    graph_entities=["Asteroidea", "Echinoidea"],
    use_mcp=True
)
  ↓
三路并行：
  A. Chroma 本地 PDF chunk 检索
  B. BookNativeBM25 本地 BM25 检索
  C. MCP raganything: raganything_graph_neighbors("Asteroidea", depth=1)
     → LightRAG 图谱邻居节点
  ↓
weighted_rrf 融合（RRF 算法）:
  graph: 0.50
  chroma: 0.35
  bm25: 0.15
  ↓
返回 12 条 PDF 证据
```

##### 9b. rerank
```
对所有检索结果按 final_score 排序，截取 top_k 条
```

##### 9c. build_context（`workflow.py:1000-1040`）

```
将 5 路检索结果去重，组装为 [E编号] 格式：

[E1] id=text_starfish_visual_features species=starfish type=species_text_chunk score=0.92
source=Wikipedia, page
海星在水下图像中通常表现为星形或放射状结构...

[E2] id=text_sea_urchin_visual_features species=sea_urchin type=species_text_chunk score=0.89
source=Wikipedia
海胆通常呈圆形或半球形，体表覆盖棘刺...

[E3] id=pdf_starfish_406 species=starfish type=pdf_chunk score=0.87
source=Field-Guide-to-SA-Offshore-Marine-Invertebrates.pdf, page 406
...
```

#### 节点 10：answer（子图，`workflow.py:1042-1300`）

##### 10a. generate

```
构造 LLM prompt:
{
  "task_type": "comparison_qa",
  "original_query": "海星和海胆怎么区分？",
  "resolved_species_ids": ["starfish", "sea_urchin"],
  "evidence": "[E1] ...\n[E2] ...\n[E3] ...",
  "requirements": [
    "只依据检索证据回答",
    "关键结论使用[E编号]引用",
    "回答使用中文"
  ],
  "strict_instructions": [
    "Directly answer the requested comparison...",
    ...
  ]
}
  ↓
LLM (OpenRouter/Qwen/DeepSeek) 生成答案
max_tokens=1800
```

**如果 LLM 调用失败** → 本地降级：
```
构造结构化降级回答：
  "上一轮识别的生物是海星、海胆。"
  "- 生活习性：...(摘取 ecology_behavior chunk)"
  "- 栖息环境：...(摘取 habitat chunk)"
  "已从本地知识库找到 5 张图片，见下方图片画廊。"
```

##### 10b. guard
```
response_mode == "normal" → 不修改
```

##### 10c. evaluate

```
打分：
  has_answer = True (非空)
  has_context = True (有检索结果)
  has_citation = True (有 [E编号])
  image_consistency = True (不需要 VLM)
  complete_answer = True (>=100 字符，括号闭合)

score = 0.20*1 + 0.25*1 + 0.20*1 + 0.15*1 + 0.20*1 = 1.0
passed = True
```

#### 节点 11：finalize（`workflow.py:1458-1469`）

```
evaluation passed → final_answer = draft_answer
```

#### 节点 12：memory_save（`workflow.py:1471-1574`）

```
保存到 data/mrag/sessions/{session_id}.json:
{
  "turns": [..., {
    "user_query": "海星和海胆怎么区分？",
    "assistant_answer": "...",
    "species_ids": ["starfish", "sea_urchin"],
    "evidence_ids": [...]
  }],
  "summary": {
    "last_species_ids": ["starfish", "sea_urchin"],
    "last_image_caption": "",
    ...
  }
}
同时写入 LangGraph checkpoint (langgraph.sqlite)
```

### 5.4 返回前端

```
AquaBioMRAGWorkflow.invoke() 返回 state
  ↓
ChatService.chat() 提取:
  - final_answer → response["answer"]
  - evidence rows → response["evidence"]
  - image rows → response["images"] (有 image_path 的)
  - trace → response["trace"]
  - warnings → response["warnings"]
  - memory_summary → response["memory"]
  ↓
ChatService._single_best_image() 精选 1 张最佳图片
  (PDF 来源优先 > Wikimedia > 本地图库)
  ↓
JSON response 返回给前端
  ↓
Streamlit st.markdown(answer) + st.image(best_image)
```

---

## 6. 关键数据结构

### 6.1 AquaBioState（LangGraph 状态）

```python
class AquaBioState(TypedDict):
    # 用户输入
    session_id: str
    original_query: str
    image_path: str | None
    image_caption: str

    # 对话上下文
    conversation_history: list[dict]
    memory_summary: dict

    # 路由与计划
    route: dict                    # RouteDecision
    selected_tools: list[str]
    tool_plan: list[str]
    react_step: int

    # 检索结果
    text_context: list[dict]       # 文本检索结果
    image_context: list[dict]      # 图片检索结果
    multimodal_context: list[dict] # 多模态检索结果
    pdf_context: list[dict]        # PDF 检索结果
    web_context: list[dict]        # 网络搜索
    final_context: str             # 组装后的上下文

    # 答案
    draft_answer: str
    final_answer: str
    evaluation_result: dict
    retry_count: int

    # 运行时
    runtime_options: dict
    trace: list[str]
    warnings: list[str]
```

### 6.2 RouteDecision

```python
class RouteDecision(BaseModel):
    task_type: Literal[
        "text_qa", "followup_text_qa", "image_qa",
        "multimodal_qa", "comparison_qa",
        "source_trace", "pdf_qa"
    ]
    need_vlm: bool                 # 是否需要视觉模型
    need_text_retrieval: bool      # 是否需要文本检索
    need_image_retrieval: bool     # 是否需要图片检索
    need_multimodal_retrieval: bool # 是否需要多模态检索
    need_pdf_retrieval: bool       # 是否需要 PDF 检索
```

### 6.3 RetrievalRequest

```python
@dataclass
class RetrievalRequest:
    query: str
    task_type: RetrievalTask
    top_k: int = 12
    species_ids: list[str] | None = None
    source_types: list[str] | None = None
```

### 6.4 检索结果行

```python
{
    "id": "text_starfish_visual_features",
    "content": "海星在水下图像中通常表现为星形...",
    "semantic_similarity": 0.8765,
    "final_score": 0.9123,
    "metadata": {
        "source_type": "species_text_chunk",
        "species_id": "starfish",
        "chunk_type": "visual_features",
        "modality": "text",
        "retrieval_source": "chroma",       # chroma / bm25 / lightrag_graph / raganything_pdf_image
        "image_path": "",                    # 仅图片检索结果有值
        "image_role": "",                    # distribution_map / specimen_overview / specimen_detail
        "page": None,                        # 仅 PDF 结果有值
        "doc_title": "Wikipedia",
        ...
    }
}
```

---

## 7. 配置文件说明

### 7.1 `.env` 关键变量

| 类别 | 变量 | 说明 |
|---|---|---|
| LLM | `AQUABIO_LLM_PROVIDER` | openrouter / qwen / deepseek |
| LLM | `OPENROUTER_MODEL` | 模型名称 |
| Embedding | `MRAG_EMBEDDING_MODEL` | BAAI/bge-m3（默认） |
| 检索 | `MRAG_TOP_K` | 默认 12 |
| 检索 | `MRAG_RERANK_TOP_K` | 默认 6 |
| 重试 | `MRAG_MAX_RETRY` | 默认 2 |
| ReAct | `MRAG_REACT_NATIVE` | true/false（LLM 自主规划） |
| MCP | `MRAG_USE_GRAPH_MCP` | true/false（图谱 MCP） |
| RAG-Anything | `AQUABIO_RAG_MCP_TRANSPORT` | streamable-http |
| RAG-Anything | `AQUABIO_RAG_MCP_PORT` | 8765 |
| RAG-Anything | `RAGANYTHING_PARSER` | mineru（PDF 解析引擎） |
| 超时 | `AQUABIO_CHAT_UI_TIMEOUT` | 默认 360 秒 |

### 7.2 `configs/agent_config.json`

```json
{
  "max_retry": 2,
  "enable_query_rewrite": true,
  "enable_answer_evaluation": true,
  "enable_source_selection": true,
  "tools": {
    "use_mcp": true,
    "enable_vlm_caption": true,
    "enable_text_retriever": true,
    "enable_image_retriever": true,
    "enable_multimodal_retriever": true,
    "enable_pdf_retriever": true
  }
}
```

### 7.3 `configs/vector_db_config.json`

```json
{
  "type": "chroma",
  "persist_dir": "data/mrag/vector_db/chroma",
  "collection_name": "aquabio_mrag",
  "retrieval": {
    "top_k": 12,
    "rerank_top_k": 6
  }
}
```

---

## 8. 启动与运行命令

### 8.1 启动聊天助手（完整模式）

```cmd
start_chat_assistant.cmd
```

启动 3 个进程：

| 进程 | 端口 | 用途 |
|---|---|---|
| FastAPI (uvicorn) | 8000 | 后端 API |
| Streamlit | 8510 | 前端 UI |
| RAG-Anything MCP | 8765 | MCP 工具服务（图谱 + PDF 图片检索） |

### 8.2 停止聊天助手

```cmd
stop_chat_assistant.cmd
```

### 8.3 CLI 模式

```bash
# 文本问答
python mrag_cli.py ask "海星和海胆怎么区分？"

# 带图片
python mrag_cli.py ask --image path/to/image.jpg "这可能是什么生物？"

# 查看会话历史
python mrag_cli.py history --session-id default

# 恢复中断的 Human-in-the-Loop 任务
python mrag_cli.py resume --session-id abc123 --answer "是海星"
```

### 8.4 离线模式

```bash
# 不调用外部 LLM API，仅返回检索证据
python mrag_cli.py ask --offline "海星有什么特征？"
```

### 8.5 RAG-Anything CLI

```bash
# 构建 PDF 索引
python raganything_cli.py index --book-id sa_invertebrates

# 查询 PDF 图谱
python raganything_cli.py search "Asteroidea distribution"

# 构建 PDF 图片资产
python raganything_cli.py image-assets --book-id sa_invertebrates

# 构建 PDF 图片向量索引
python raganything_cli.py image-index --book-id sa_invertebrates

# 查询 PDF 图片
python raganything_cli.py image-search "starfish distribution map"
```

---

## 附录 A：完整文件索引

### 核心代码文件

| 文件 | 职责 |
|---|---|
| `mrag_app.py` | Streamlit 多会话聊天前端 |
| `api_app.py` | FastAPI 入口 |
| `src/aquabio_web/api.py` | 20+ REST API 端点 |
| `src/aquabio_web/service.py` | ChatService 业务逻辑 |
| `src/aquabio_mrag/workflow.py` | LangGraph 12 节点 StateGraph |
| `src/aquabio_mrag/retrieval.py` | 多源检索 + 加权重排序 |
| `src/aquabio_mrag/retrieval_agent.py` | 检索 Agent（PDF/图谱/图片） |
| `src/aquabio_mrag/react_agent.py` | ControlledReActPlanner |
| `src/aquabio_mrag/vector_db.py` | ChromaMRAGStore |
| `src/aquabio_mrag/config.py` | MRAGPaths + MRAGSettings |
| `src/aquabio_mrag/mcp_client.py` | MCP 客户端（双 Server） |
| `src/aquabio_raganything/book_native.py` | PDF 章节/分类单元结构化 |
| `src/aquabio_raganything/image_rag.py` | PDF 图片实体绑定 + 向量库 + 角色分类 |
| `src/aquabio_raganything/indexer.py` | PDF 分段索引 |
| `src/aquabio/config.py` | Settings + load_env |
| `src/aquabio/openrouter.py` | OpenRouter LLM 封装 |
| `src/aquabio/gemini.py` | Gemini Vision 封装 |

### 构建脚本

| 脚本 | 步骤 |
|---|---|
| `scripts/01_crawl_wikipedia_worms.py` | 爬取物种文本 + 分类 |
| `scripts/02_crawl_commons_images.py` | 爬取图片 |
| `scripts/02b_fill_inaturalist_images.py` | 补充 iNaturalist 图片 |
| `scripts/03_build_multimodal_documents.py` | 构建多模态文档 |
| `scripts/04_download_parse_pdfs.py` | 下载 + 解析 PDF |
| `scripts/05_build_chroma_bge.py` | 构建 Chroma 向量索引 |

### 配置文件

| 文件 | 用途 |
|---|---|
| `.env` | 环境变量（LLM/Embedding/MCP 配置） |
| `configs/agent_config.json` | Agent 行为参数 |
| `configs/model_config.json` | LLM/VLM/Embedding 模型选择 |
| `configs/pdf_sources.json` | PDF 来源列表 |
| `configs/vector_db_config.json` | 向量库参数 |

---

## 附录 B：Glossary

| 术语 | 说明 |
|---|---|
| **MRAG** | Multimodal RAG（多模态检索增强生成） |
| **ReAct** | Reasoning + Acting，LLM 交替推理和调用工具的范式 |
| **Controlled ReAct** | 受控 ReAct，工具选择由确定性规则而非 LLM 决定 |
| **RRF** | Reciprocal Rank Fusion，多路检索结果融合算法 |
| **StateGraph** | LangGraph 的有状态图，每个节点读写共享 State |
| **BGE-M3** | BAAI 的多语言 Embedding 模型，1024 维 |
| **Chroma** | 开源向量数据库 |
| **LightRAG** | 基于图谱的 RAG 框架 |
| **MCP** | Model Context Protocol，AI 应用与外部工具的标准协议 |
| **MinerU** | PDF 解析引擎 |
| **Book Native** | 将 PDF 按分类单元结构化的确定性管道 |
| **distribution_map** | 分布图角色标签（页面顶部图片） |
| **specimen_overview** | 全标本图角色标签（页面左侧图片） |
| **specimen_detail** | 细节图角色标签（其他位置图片） |
