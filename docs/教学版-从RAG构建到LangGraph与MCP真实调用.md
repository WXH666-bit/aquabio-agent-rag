# 教学版：从 RAG 构建到 LangGraph 与 MCP 真实调用

本文面向想复现 AquaBio-AgentRAG 架构的学生，目标不是只看懂这个项目，而是能照着模块顺序搭一个自己的多模态 Agentic RAG 系统。

项目当前主线可以概括为：

```text
PDF / 图片 / 文本数据
  -> 结构化抽取与清洗
  -> BGE-M3 向量化
  -> Chroma / LightRAG / PDF 图片索引持久化
  -> FastAPI + Streamlit 接收用户请求
  -> LangGraph 负责对话流程编排
  -> RetrievalAgent 通过 MCP 调用外部检索工具
  -> LLM 结合证据生成回答
  -> 会话记录、执行轨迹、证据、记忆持久化
```

---

## 1. 项目分层总览

### 1.1 目录职责

```text
AquaBio-AgentRAG/
  data/
    mrag/
      knowledge/                 # 20 类基础物种文本、图片 caption、图文对
      vector_db/chroma/           # Chroma 向量库
      raganything/
        book_native/              # PDF 原生结构化结果
        extracted_assets/         # PDF 抽取出的真实图片和图片索引
        working/                  # LightRAG 持久化图谱、KV、向量文件
        manifests/                # PDF / 图谱索引状态
        logs/                     # API、UI、MCP 日志
      sessions/                   # 多轮会话持久化 JSON

  src/
    aquabio_mrag/                 # LangGraph 主流程、Chroma 检索、MCP Client
    aquabio_raganything/          # PDF 结构化、图片实体绑定、LightRAG/MCP 工具
    aquabio_web/                  # FastAPI 服务逻辑和前端结果整理
    aquabio/                      # LLM/VLM API 配置与客户端

  raganything_cli.py              # PDF 建库、图片索引、图谱查询 CLI
  mrag_cli.py                     # 命令行问答 / MCP 检查
  mrag_app.py                     # Streamlit 前端
  start_chat_assistant.cmd        # 一键启动 API + UI + MCP
```

### 1.2 四个核心模块

| 模块 | 作用 | 关键文件 |
|---|---|---|
| RAG 建库 | 把文本、PDF、图片变成可检索知识 | `data_pipeline.py`、`book_native.py`、`image_rag.py`、`vector_db.py` |
| LangGraph | 管理一次问答的节点、状态、分支、重试、记忆 | `workflow.py` |
| MCP | 把检索能力变成标准工具，由主 Agent 调用 | `mcp_client.py`、`mcp_server.py` |
| 记忆持久化 | 保存多轮对话、上一轮物种、图片、证据 | `conversation.py`、`workflow.py` |

---

## 2. RAG 数据构建流程

### 2.1 基础 20 类知识库

基础知识库来自文本、图片 caption 和图文对，适合做一般物种问答和上传图片后的辅助解释。

关键代码：

| 功能 | 文件与行号 |
|---|---|
| 加载物种列表 | `src/aquabio_mrag/data_pipeline.py:62` |
| 抓取 Wikipedia / WoRMS 文本 | `src/aquabio_mrag/data_pipeline.py:71` |
| 抓取 Wikimedia Commons 图片 | `src/aquabio_mrag/data_pipeline.py:168` |
| 图片记录写入 `image_docs.jsonl` | `src/aquabio_mrag/data_pipeline.py:177`、`288-309` |
| 向量化与写入 Chroma | `src/aquabio_mrag/vector_db.py:61`、`77-134` |
| 查询 Chroma | `src/aquabio_mrag/vector_db.py:146-177` |

基础向量库流程：

```text
species_list.json
  -> data_pipeline.py 抓文本和图片
  -> 生成 RAGDocument / ImageDocument / MultimodalPair
  -> BGE-M3 embedding
  -> Chroma PersistentClient
  -> data/mrag/vector_db/chroma/
```

对应代码位置：

```text
src/aquabio_mrag/vector_db.py:15    BGEEmbedder
src/aquabio_mrag/vector_db.py:38    SentenceTransformer 加载 BGE-M3
src/aquabio_mrag/vector_db.py:61    ChromaMRAGStore
src/aquabio_mrag/vector_db.py:65    chromadb.PersistentClient
src/aquabio_mrag/vector_db.py:77    build()
src/aquabio_mrag/vector_db.py:146   query()
```

教学建议：学生先做一个最小版，只准备 `species_list.json + 5 张图片 + 5 段文本`，跑通 Chroma 查询后再加 PDF。

---

## 3. PDF RAG 构建流程

这个项目的 PDF 不只是切 chunk，而是尽量保留“书页 -> 物种 -> 图片 -> 页码 -> 分类关系”的结构。

### 3.1 PDF 原生结构化

关键文件：

```text
src/aquabio_raganything/book_native.py
```

关键行号：

| 功能 | 行号 |
|---|---|
| 解析南非图鉴分类目录 | `book_native.py:158` |
| 提取页面文字 span | `book_native.py:132` |
| 提取标题、分类字段、印刷页 | `book_native.py:254`、`268`、`308` |
| 提取页面图片对象 | `book_native.py:316` |
| 解析物种页单元 | `book_native.py:354` |
| 生成 RAG chunks | `book_native.py:551` |
| 生成实体关系 triples | `book_native.py:615` |
| 转成 RAG-Anything content list | `book_native.py:661` |
| 构建 book-native 数据 | `book_native.py:823` |

输出目录：

```text
data/mrag/raganything/book_native/sa_invertebrates/
  species_page_units.jsonl       # 每个物种页的完整结构
  rag_chunks.jsonl               # 可检索文本 chunk
  relation_triples.jsonl         # 实体关系
  book_taxa_catalog.jsonl        # 目录和物种页映射
  raganything_content_list.jsonl # 给 RAG-Anything/LightRAG 的输入
```

### 3.2 PDF 图片实体绑定

关键文件：

```text
src/aquabio_raganything/image_rag.py
```

关键行号：

| 功能 | 行号 |
|---|---|
| 判断图片是实体图还是分布图 | `image_rag.py:106` |
| 解析用户是否要分布图/实例图 | `image_rag.py:117` |
| 选择符合角色的图片 | `image_rag.py:162` |
| 解析用户指定页码 | `image_rag.py:184` |
| 构造图片 caption | `image_rag.py:252` |
| 构造图片 embedding 文本 | `image_rag.py:291` |
| 抽取 PDF 图片资产 | `image_rag.py:325` |
| PDF 图片 Chroma 向量库 | `image_rag.py:585` |
| 构建图片向量索引 | `image_rag.py:611` |
| 查询图片向量索引 | `image_rag.py:679` |
| 统一 PDF 图片查询入口 | `image_rag.py:794` |
| 按实体名直接查图片 | `image_rag.py:851` |

输出目录：

```text
data/mrag/raganything/extracted_assets/sa_invertebrates/
  images/
    sa_taxon_jaslal_p0149_img_03.jpeg
    sa_taxon_jaslal_p0149_img_04.jpeg
  image_index/
    pdf_image_captions.jsonl
    pdf_image_rag_docs.jsonl
    linked_pdf_images.jsonl
    image_relation_triples.jsonl
```

示例：`Jasus lalandii`

```text
PDF page: 149
printed page: 146
common name: West coast rock lobster
image 03: specimen_overview
image 04: distribution_map
relations:
  Jasus lalandii --is_a--> Palinuridae
  Jasus lalandii --is_a--> Decapoda
  Jasus lalandii --distributed_in--> Southern African endemic...
```

### 3.3 PDF 建库命令

CLI 入口：

```text
raganything_cli.py
```

关键命令定义：

| 命令 | 行号 |
|---|---|
| `book-native` | `raganything_cli.py:43` |
| `index` | `raganything_cli.py:59` |
| `index-book-native` | `raganything_cli.py:65` |
| `image-assets` | `raganything_cli.py:74` |
| `index-images` | `raganything_cli.py:87` |
| `image-query` | `raganything_cli.py:103` |
| `status` | `raganything_cli.py:111` |
| `query` | `raganything_cli.py:120` |
| `neighbors` | `raganything_cli.py:125` |

推荐教学顺序：

```cmd
cd /d F:\rag\AquaBio-AgentRAG

.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
.\.venv\Scripts\python.exe raganything_cli.py image-assets --book sa_invertebrates --overwrite
.\.venv\Scripts\python.exe raganything_cli.py index-images --book sa_invertebrates --reset
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 分布图" --entity "Jasus lalandii"
```

---

## 4. LightRAG / 图谱持久化

LightRAG 用于保存图结构、文档块、实体向量、关系向量和状态。

主要持久化目录：

```text
data/mrag/raganything/working/
  graph_chunk_entity_relation.graphml
  kv_store_doc_status.json
  kv_store_entity_chunks.json
  kv_store_full_docs.json
  kv_store_full_entities.json
  kv_store_full_relations.json
  kv_store_llm_response_cache.json
  kv_store_relation_chunks.json
  kv_store_text_chunks.json
  vdb_chunks.json
  vdb_entities.json
  vdb_relationships.json
```

查询适配器：

```text
src/aquabio_raganything/query_adapter.py
```

关键行号：

| 功能 | 行号 |
|---|---|
| 懒加载 LightRAG 查询对象 | `query_adapter.py:40` |
| 解析 LightRAG 返回的实体/关系上下文 | `query_adapter.py:81` |
| 构造 evidence | `query_adapter.py:93` |
| 查 GraphML 邻居 | `query_adapter.py:172` |
| 读取 book-native 来源详情 | `query_adapter.py:276` |
| 读取索引状态 | `query_adapter.py:356` |
| hybrid 检索 | `query_adapter.py:401` |

为什么还要 `source_detail()`？

LightRAG hybrid 有时慢，而且 GraphML 不一定覆盖全部 book-native 物种。为了回答“书中图片、页码、分类关系、PDF 证据”，项目增加了确定性查询：

```text
source_detail()
  -> 查 species_page_units.jsonl
  -> 查 rag_chunks.jsonl
  -> 查 pdf_image_captions.jsonl
  -> 查 relation_triples.jsonl
```

这能保证像 `Jasus lalandii` 这种书中明确存在的物种，不会因为图谱检索没命中而回答“没有”。

---

## 5. MCP 的真实调用设计

### 5.1 MCP Server

RAG-Anything MCP Server：

```text
src/aquabio_raganything/mcp_server.py
```

工具注册行号：

```text
mcp_server.py:32   raganything_index_status
mcp_server.py:38   raganything_graph_neighbors
mcp_server.py:44   raganything_hybrid_search
mcp_server.py:52   raganything_image_search
mcp_server.py:60   search_pdf_images
mcp_server.py:68   raganything_entity_images
mcp_server.py:75   raganything_source_detail
mcp_server.py:83   get_source_detail
```

教学重点：  
MCP tool 是“可被 Agent 调用的外部能力”，不是 LangGraph 节点。比如：

```text
search_pdf_images        # 查 PDF 真实图片
get_source_detail        # 查页码、source、chunks、images、relations
raganything_graph_neighbors
raganything_hybrid_search
```

### 5.2 MCP Client

主流程调用 MCP 的客户端：

```text
src/aquabio_mrag/mcp_client.py
```

关键行号：

| 功能 | 行号 |
|---|---|
| MCP server 配置结构 | `mcp_client.py:16` |
| MCP 客户端类 | `mcp_client.py:26` |
| 建立 stdio / streamable-http 会话 | `mcp_client.py:46` |
| list_tools | `mcp_client.py:108` |
| call_tool | `mcp_client.py:114` |
| 同步封装 | `mcp_client.py:136`、`139` |
| 项目默认 MCP 配置 | `mcp_client.py:145` |

当前项目的两个 MCP server：

```text
chroma:
  python = .venv\Scripts\python.exe
  module = aquabio_mrag.mcp_server

raganything:
  python = .venv-raganything\Scripts\python.exe
  url = http://127.0.0.1:8765/mcp
  module = aquabio_raganything.mcp_server
```

### 5.3 RetrievalAgent 如何真实调用 MCP

关键文件：

```text
src/aquabio_mrag/retrieval_agent.py
```

关键行号：

| 功能 | 行号 |
|---|---|
| RetrievalAgent 类 | `retrieval_agent.py:94` |
| PDF 文本 + 图谱融合检索 | `retrieval_agent.py:289` |
| 记录 tool_calls | `retrieval_agent.py:298`、`470` |
| 调 `raganything_graph_neighbors` | `retrieval_agent.py:324` |
| 调 `get_source_detail` | `retrieval_agent.py:360` |
| 调 `raganything_hybrid_search` | `retrieval_agent.py:401` |
| PDF 图片 MCP 检索 | `retrieval_agent.py:473` |
| 调 `search_pdf_images` | `retrieval_agent.py:488` |

真实调用链：

```text
LangGraph retrieval_node
  -> RetrievalAgent.search_pdf_images()
      -> MCPClient.call_tool_sync("raganything", "search_pdf_images", ...)

LangGraph retrieval_node
  -> RetrievalAgent.search_pdf()
      -> MCPClient.call_tool_sync("raganything", "raganything_graph_neighbors", ...)
      -> MCPClient.call_tool_sync("raganything", "get_source_detail", ...)
      -> MCPClient.call_tool_sync("raganything", "raganything_hybrid_search", ...)
```

每个 MCP 调用都会写入：

```json
{
  "tool_name": "search_pdf_images",
  "tool_source": "mcp",
  "server": "raganything",
  "status": "success",
  "latency_ms": 217,
  "result_count": 2
}
```

---

## 6. LangGraph 主流程

LangGraph 是整个项目的“流程编排层”。它决定：

```text
这是不是追问？
要不要用视觉模型？
要不要检索文本？
要不要查 PDF 图片？
要不要查 MCP 图谱？
回答失败后要不要重试？
本轮结束后保存什么记忆？
```

关键文件：

```text
src/aquabio_mrag/workflow.py
```

### 6.1 StateGraph 节点

关键行号：

```text
workflow.py:120   AquaBioMRAGWorkflow
workflow.py:182   session_init_node
workflow.py:192   memory_load_node
workflow.py:223   followup_resolver_node
workflow.py:302   router_node
workflow.py:377   rewrite_node
workflow.py:406   source_selection_node
workflow.py:438   react_tool_plan_node
workflow.py:517   vision_node
workflow.py:633   retrieval_node
workflow.py:1076  rerank_node
workflow.py:1098  context_node
workflow.py:1140  answer_node
workflow.py:1400  response_guard_node
workflow.py:1427  evaluation_node
workflow.py:1556  finalize_node
workflow.py:1569  memory_save_node
```

完整节点流：

```text
session_init
  -> memory_load
  -> followup_resolver
  -> router
  -> rewrite
  -> source_selection
  -> react_tool_plan
  -> vision
  -> retrieval_agent
  -> answer_agent
  -> finalize
  -> memory_save
```

### 6.2 图的构建代码

```text
workflow.py:1721  _build_retrieval_subgraph()
workflow.py:1732  _build_answer_subgraph()
workflow.py:1743  _build()
workflow.py:1744  graph = StateGraph(AquaBioState)
workflow.py:1745-1764  graph.add_node(...)
workflow.py:1766-1818  graph.add_edge / graph.add_conditional_edges
workflow.py:1826  SqliteSaver checkpointer
workflow.py:1829  invoke()
```

教学解释：

```text
Node = 一个流程步骤
Edge = 下一个步骤
Conditional Edge = 根据 state 决定下一个步骤
State = 所有中间结果的大字典
Checkpointer = 把 LangGraph 状态保存到 SQLite，支持恢复
```

### 6.3 路由如何决定工具

`router_node()` 在 `workflow.py:302`。

它根据用户输入判断：

```text
text_qa
image_qa
multimodal_qa
comparison_qa
source_trace
pdf_qa
followup_text_qa
```

`source_selection_node()` 在 `workflow.py:406`。

它把任务类型转换成工具：

```text
vlm_caption
text_retriever
image_retriever
pdf_image_retriever
multimodal_retriever
pdf_retriever
```

例如：

```text
用户问：
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。

router_node:
  task_type = pdf_qa

source_selection_node:
  selected_tools =
    image_retriever
    pdf_image_retriever
    pdf_retriever
```

### 6.4 retrieval_node 如何调用 MCP

`retrieval_node()` 在 `workflow.py:633`。

关键逻辑：

```text
workflow.py:758   调 search_pdf_images()
workflow.py:1006  汇总 tool_calls
workflow.py:1019  trace 写入 mcp_tool_call
workflow.py:1049  tool_calls 写回 state
```

trace 示例：

```text
retrieval:text=0,web=0,image=2,pair=0,pdf=12,mcp_graph=12,mcp_pdf_images=2
mcp_tool_call:search_pdf_images status=success latency=217ms results=2
mcp_tool_call:raganything_graph_neighbors status=success latency=544ms results=12
mcp_tool_call:get_source_detail status=success latency=462ms results=12
mcp_tool_call:raganything_hybrid_search status=success latency=32265ms results=0
```

这就是判断“是否真的使用 MCP”的依据。

---

## 7. 多轮对话与记忆持久化

### 7.1 会话 JSON 持久化

关键文件：

```text
src/aquabio_mrag/conversation.py
```

关键行号：

```text
conversation.py:14   ConversationStore
conversation.py:22   normalize_session_id()
conversation.py:28   path_for()
conversation.py:31   load()
conversation.py:56   save()
conversation.py:69   append_turn()
conversation.py:91   list_sessions()
```

会话保存目录：

```text
data/mrag/sessions/
  sess_xxx.json
```

每轮保存内容包括：

```text
user_query
resolved_query
assistant_answer
species_ids
image_path
image_caption
evidence
trace
warnings
```

### 7.2 LangGraph 中的记忆加载

```text
workflow.py:192  memory_load_node()
```

它会读取最近 8 轮历史：

```text
turn_index
user_query
resolved_query
assistant_answer
species_ids
image_path
```

然后写入 state：

```text
conversation_history
memory_summary
```

### 7.3 追问解析

```text
workflow.py:223  followup_resolver_node()
```

它识别：

```text
刚才
上次
前面
这个生物
那个生物
它
其
```

如果上一轮保存了物种，就把追问改写成明确问题。

示例：

```text
第一轮：
这是什么生物？

系统保存：
last_species_ids = ["starfish"]

第二轮：
刚才那个生物的分布图给我

followup_resolver:
追问指代的上一轮物种是海星。用户追问：刚才那个生物的分布图给我
```

### 7.4 记忆保存

```text
workflow.py:1569  memory_save_node()
```

这一节点在回答结束后保存：

```text
本轮问题
最终回答
解析出的物种
图片 caption
上传图片路径
检索证据
执行 trace
```

同时，LangGraph 自己也有 checkpoint：

```text
workflow.py:12    SqliteSaver
workflow.py:1826  checkpointer=SqliteSaver(...)
```

区别：

| 类型 | 作用 |
|---|---|
| ConversationStore JSON | 给用户侧多轮记忆、会话列表、追问解析使用 |
| LangGraph SQLite checkpoint | 给 LangGraph 状态恢复、HITL 中断续跑使用 |

---

## 8. 前后端请求流

### 8.1 FastAPI / ChatService

前端不直接碰 LangGraph，而是调用 FastAPI。

关键文件：

```text
src/aquabio_web/service.py
src/aquabio_web/api.py
mrag_app.py
```

关键行号：

```text
service.py:39    ChatService
service.py:58    workflow()
service.py:391   warmup()
service.py:441   save_upload()
service.py:722   chat()
service.py:758   state = self.workflow().invoke(...)
service.py:1115  submit_chat()
service.py:1147  _run_chat_task()
service.py:1214  chat_task()
service.py:1251  status()
service.py:1316  mcp_tools()

mrag_app.py:225  api()
mrag_app.py:290  upload_file()
mrag_app.py:370  render_chat()
mrag_app.py:380  render_react_flow()
```

在线流程：

```text
Streamlit UI
  -> 上传图片 / PDF
  -> FastAPI save_upload
  -> FastAPI submit_chat
  -> ChatService._run_chat_task
  -> ChatService.chat
  -> AquaBioMRAGWorkflow.invoke
  -> LangGraph 节点流
  -> 返回 answer / evidence / trace / tool_calls
  -> UI 展示回答、证据、MCP、ReAct 流程
```

---

## 9. 学生复现建议：最小可运行版本

### 9.1 第一步：只做文本 RAG

必须实现：

```text
documents.jsonl
embedding.py
vector_store.py
retriever.py
```

流程：

```text
文本 -> chunk -> embedding -> Chroma -> query -> top_k chunks
```

### 9.2 第二步：加 PDF 原生结构

必须实现：

```text
pdf_parser.py
book_units.jsonl
rag_chunks.jsonl
relation_triples.jsonl
```

不要一开始就上复杂 OCR。优先解析数字文本 PDF。

### 9.3 第三步：加图片实体绑定

必须实现：

```text
image_objects.jsonl
pdf_image_captions.jsonl
image_relation_triples.jsonl
```

每张图片必须保存：

```text
image_id
image_path
scientific_name
common_name
pdf_page
printed_page
image_role
caption
```

### 9.4 第四步：加 LangGraph

最小节点：

```text
session_init
memory_load
router
source_selection
retrieval
answer
memory_save
```

不要一开始就做 12 个节点。先跑通状态流。

### 9.5 第五步：加 MCP

最小 MCP tools：

```text
search_text
search_pdf_images
get_source_detail
graph_neighbors
```

判断 MCP 是否真实调用：

```text
trace 里必须出现：
mcp_tool_call:search_pdf_images
mcp_tool_call:get_source_detail
```

只看到 `pdf_retriever` 不算真的证明 MCP。

### 9.6 第六步：加多轮记忆

最小字段：

```text
session_id
turns[]
last_species_ids
last_image_caption
last_image_path
```

追问解析必须支持：

```text
刚才那个
这个生物
它的分布图
它是什么颜色
```

---

## 10. 课堂讲解用完整例子

用户问：

```text
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。
```

执行流程：

```text
session_init_node
  -> 创建 / 规范化 session_id

memory_load_node
  -> 读取历史 turns 和 summary

router_node
  -> 判断为 pdf_qa

query_rewrite_node
  -> 增加“优先使用 PDF、保留页码”的检索提示

source_selection_node
  -> 选择 image_retriever + pdf_image_retriever + pdf_retriever

react_tool_plan_node
  -> 确认工具计划

retrieval_node
  -> 调 MCP:
       search_pdf_images
       raganything_graph_neighbors
       get_source_detail
       raganything_hybrid_search

context_node
  -> 把图片、分类关系、PDF chunk 组装成 [E1] [E2] 证据

answer_node
  -> 调 Qwen 生成中文回答

memory_save_node
  -> 保存本轮问答和证据
```

实际证据：

```text
图片：
  sa_taxon_jaslal_p0149_img_03.jpeg
  role = specimen_overview
  PDF page = 149
  printed page = 146

  sa_taxon_jaslal_p0149_img_04.jpeg
  role = distribution_map
  PDF page = 149
  printed page = 146

分类：
  Jasus lalandii -> Jasus
  Jasus lalandii -> Palinuridae
  Jasus lalandii -> Decapoda
  Jasus lalandii -> Malacostraca
  Jasus lalandii -> Arthropoda

普通名：
  West coast rock lobster

分布：
  Southern African endemic. Restricted to southern Africa from Northern Namibia to Algoa Bay.
```

---

## 11. 常见错误与排查

### 11.1 “有 MCP Server，但没有真的调用 MCP”

错误表现：

```text
trace 只有 pdf_retriever
没有 mcp_tool_call
```

正确表现：

```text
mcp_tool_call:search_pdf_images status=success
mcp_tool_call:get_source_detail status=success
```

### 11.2 “问 PDF 图片却返回普通网络图”

原因通常是：

```text
router 没选 pdf_image_retriever
或者本地 PDF 图片索引没有命中
或者 image_role 没识别 distribution_map / specimen
```

排查：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 分布图" --entity "Jasus lalandii"
```

### 11.3 “MCP 卡住很久”

原因可能是 Hugging Face 试图联网补配置。

必须设置：

```text
HF_HOME=F:\huggingface
HUGGINGFACE_HUB_CACHE=F:\huggingface\hub
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
RAGANYTHING_MCP_TIMEOUT=35
RAGANYTHING_QUERY_TIMEOUT=25
```

### 11.4 “GraphML 没找到实体”

当前项目做了 fallback：

```text
graph_neighbors()
  -> 先查 LightRAG GraphML
  -> 查不到则用 book_native/relation_triples.jsonl
```

所以它仍然通过 MCP 返回图关系，只是来源从 GraphML 降级为 book-native triples。

---

## 12. 启动和验证命令

启动：

```cmd
cd /d F:\rag\AquaBio-AgentRAG
start_chat_assistant.cmd
```

停止：

```cmd
stop_chat_assistant.cmd
```

查看前端：

```text
http://127.0.0.1:8510
```

查看 API：

```text
http://127.0.0.1:8000/docs
```

验证 MCP 工具：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server raganything
```

验证 PDF 图片：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 分布图" --entity "Jasus lalandii"
```

验证完整问答：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session teach_jasus --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。"
```

---

## 13. 给学生的设计原则

1. 先把数据结构设计清楚，再谈 Agent。
2. 每个 chunk 必须有 `doc_id/page/source_file`。
3. 每张图片必须有 `image_id/image_path/entity/page/image_role`。
4. RAG 不是只做向量库，复杂 PDF 要保留实体和关系。
5. LangGraph 负责流程，不负责具体检索。
6. MCP tool 负责具体外部能力，必须在 trace 里可观察。
7. 会话记忆要持久化，不要只存在内存里。
8. 多轮追问必须先解析“它/刚才那个”指代。
9. 本地库能回答时优先本地库，网络图片只能作为 fallback。
10. 每次工具调用都要记录 `tool_name/status/latency/result_count`。

---

## 14. 一句话总结

这个项目的教学价值在于它把一个多模态 RAG 系统拆成了清晰的四层：

```text
数据层：PDF / 图片 / 文本结构化
检索层：Chroma / LightRAG / PDF 图片索引
工具层：MCP Server + MCP Client
智能体层：LangGraph StateGraph + 会话记忆 + LLM 回答
```

学生按照这四层逐步搭建，就能从一个普通 RAG 项目，扩展到真正有多工具调用、图文证据、PDF 页码来源和多轮记忆的 Agentic RAG 系统。
