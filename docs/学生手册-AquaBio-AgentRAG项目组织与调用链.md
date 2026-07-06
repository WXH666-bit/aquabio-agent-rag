# 学生手册：AquaBio-AgentRAG 项目组织、RAG 构建、LangGraph 与 MCP 调用链

这份手册给学生使用，目标是把 `F:\rag\AquaBio-AgentRAG` 这个项目按“能看懂、能运行、能复现、能继续开发”的顺序讲清楚。

核心问题先回答清楚：

```text
1. RAG-Anything MCP Server 和 Chroma MCP Server 的 tools 会不会根据问题自动调用？
   会，但不是 MCP 自己主动调用。
   是 LangGraph 根据用户问题选择工具，然后 retrieval_node / RetrievalAgent 通过 MCPClient 调用对应 MCP tool。

2. 期望 trace 是什么？
   期望 trace 是运行后的验收清单。
   它告诉你：如果这个场景真的跑通，state["trace"] 或 state["tool_calls"] 里应该出现哪些节点、哪些 MCP tool、什么 status 和 result_count。

3. 这些功能真实实现了吗？
   已实现核心链路：
   FastAPI/Streamlit -> ChatService -> LangGraph workflow -> RetrievalAgent -> MCPClient -> Chroma MCP / RAG-Anything MCP -> Qwen 生成回答 -> 会话持久化。
   但不同场景是否一定调用某个 tool，要看 router/source_selection 的判断、用户问题里是否明确要求 PDF 图片/图谱/页码，以及是否开启 --mcp-retrieval。
```

---

## 1. 项目总览

项目根目录：

```text
F:\rag\AquaBio-AgentRAG
```

项目做的事情：

```text
水下生物多模态 AgentRAG
  支持文本问答
  支持上传图片识别
  支持图文联合问答
  支持 PDF 图书证据检索
  支持 PDF 中抽取出来的生物图片、分布图、页码
  支持 LightRAG/NetworkX 图谱关系查询
  支持 LangGraph 多节点流程
  支持 MCP tools 调用
  支持多轮会话记忆持久化
```

一句话架构：

```text
前端 UI / CLI
  -> 后端服务 ChatService
  -> LangGraph 工作流
  -> 路由判断问题类型
  -> 自动选择检索工具
  -> RetrievalAgent 调 MCP tools / 本地向量库
  -> 汇总证据
  -> Qwen 生成回答
  -> 保存会话记忆和证据
```

---

## 2. 项目目录解释

### 2.1 根目录重要文件

| 路径 | 作用 |
|---|---|
| `start_chat_assistant.cmd` | 启动本地聊天助手，通常会启动 FastAPI 后端和前端 UI |
| `stop_chat_assistant.cmd` | 停止本地聊天助手相关进程 |
| `mrag_cli.py` | 命令行问答入口，可以直接跑 LangGraph 工作流 |
| `raganything_cli.py` | PDF / RAG-Anything / LightRAG 图谱与图片索引的命令行入口 |
| `api_app.py` | FastAPI 应用入口包装 |
| `mrag_app.py` | Streamlit/前端应用入口之一 |
| `.env` | API key、模型、路径等运行配置 |
| `requirements.txt` | Python 依赖 |
| `docs/` | 项目说明、教学文档、技术设计文档 |
| `data/` | 数据集、PDF、图片、向量库、RAG-Anything 持久化结果 |
| `src/` | 主要源码 |

### 2.2 `src/aquabio_mrag`

这是当前 AgentRAG 主流程模块。

| 文件 | 作用 |
|---|---|
| `workflow.py` | LangGraph 核心工作流，包含所有 node |
| `retrieval_agent.py` | 检索 Agent，负责调 Chroma、本地 PDF、RAG-Anything MCP |
| `mcp_client.py` | MCP stdio client，负责启动/连接 MCP Server 并调用 tool |
| `mcp_server.py` | Chroma MCP Server，暴露文本、图片 caption、多模态、PDF chunk 等工具 |
| `conversation.py` | 会话持久化，多轮对话记录保存 |
| `vector_db.py` | Chroma 向量库封装 |
| `data_pipeline.py` | 基础数据处理与入库 |
| `pdf_pipeline.py` | 普通 PDF chunk 处理 |
| `react_agent.py` | ReAct / 工具计划辅助逻辑 |
| `models.py` | State、Route、Evidence 等数据结构 |
| `config.py` | 路径和配置读取 |

### 2.3 `src/aquabio_raganything`

这是 PDF 图谱、PDF 图片、RAG-Anything/LightRAG 风格索引模块。

| 文件 | 作用 |
|---|---|
| `book_native.py` | 把 PDF 图书转成确定性的章节、物种页、page unit、chunk |
| `image_rag.py` | PDF 图片检索、角色识别，如 specimen、distribution_map |
| `indexer.py` | RAG-Anything/LightRAG 索引相关流程 |
| `inventory.py` | PDF 页扫描、候选页盘点 |
| `manifest.py` | 索引清单、断点和状态记录 |
| `query_adapter.py` | 查询适配层，提供 hybrid_search、graph_neighbors、source_detail 等 |
| `mcp_server.py` | RAG-Anything MCP Server，暴露 PDF 图谱和 PDF 图片工具 |
| `storage_audit.py` | 持久化存储审计 |
| `audit.py` | RAG-Anything 处理结果审计 |

### 2.4 `src/aquabio_web`

这是本地聊天助手后端和前端交互层。

| 文件 | 作用 |
|---|---|
| `api.py` | FastAPI 路由 |
| `service.py` | ChatService，连接前端请求和 LangGraph workflow |
| `schemas.py` | API 请求/响应结构 |
| `presentation.py` | 把 workflow state 转成前端展示格式 |
| `store.py` | 前端会话/附件存储 |
| `network_images.py` | 网络图片 fallback |
| `web_knowledge.py` | 网络知识补充，如 Wikipedia |

### 2.5 `src/aquabio`

这是早期基础 RAG/模型调用模块，部分仍被复用。

| 文件 | 作用 |
|---|---|
| `openrouter.py` | OpenRouter/OpenAI-compatible 调用 |
| `gemini.py` | Gemini 视觉/文本调用 |
| `pdf_ingest.py` | 早期 PDF 解析 |
| `retriever.py` | 早期检索 |
| `vector_store.py` | 早期向量库 |
| `agent.py` | 早期 Agent 封装 |
| `image_tools.py` | 图片工具 |

---

## 3. 数据和持久化位置

### 3.1 原始 PDF

```text
data/mrag/pdfs/
```

重点 PDF：

```text
data/mrag/pdfs/Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf
data/mrag/pdfs/FIELD IDENTIFICATION GUIDE TO THE LIVING.pdf
```

### 3.2 基础图片数据

```text
data/mrag/images/
```

这里是基础图片样例，例如 starfish、seahorse、manta ray 等。

### 3.3 Chroma 向量库

```text
data/mrag/vector_db/
```

职责：

```text
基础物种文本
基础图片 caption
图文对
普通 PDF chunks
```

### 3.4 RAG-Anything / LightRAG 风格持久化

```text
data/mrag/raganything/
```

典型子目录：

```text
book_native/
  sa_invertebrates/
    taxa.jsonl
    page_units.jsonl
    rag_chunks.jsonl
    graph_edges.jsonl
    image_assets.jsonl

working/
  图谱、向量、状态、缓存等持久化文件

extracted_assets/
  从 PDF 中提取出的图片资源

manifests/
  索引进度和处理状态
```

### 3.5 会话记忆

通常在：

```text
data/mrag/sessions/
```

职责：

```text
保存 session_id
保存每一轮用户问题
保存助手回答
保存候选物种、图片信息、证据摘要
用于下一轮理解“刚才那个生物”
```

---

## 4. RAG 构建流程

项目里有两套互补索引。

### 4.1 Chroma 基础向量索引

职责：

```text
快速语义检索
查常见水下生物文本
查图片 caption
查图文对
查普通 PDF chunk
```

流程：

```text
原始文本 / 图片 caption / PDF chunk
  -> 清洗
  -> 切分
  -> BGE-M3 embedding
  -> Chroma collection
  -> mrag_cli / workflow 检索
```

相关文件：

```text
src/aquabio_mrag/data_pipeline.py
src/aquabio_mrag/pdf_pipeline.py
src/aquabio_mrag/vector_db.py
src/aquabio_mrag/retrieval_agent.py
```

### 4.2 RAG-Anything / PDF 图谱索引

职责：

```text
处理 Field Guide 类 PDF
定位物种页
抽取物种页文本
抽取 PDF 图片
识别图片角色：物种例图、细节图、分布图
构建实体与关系：species、taxon、image、page、document
提供图谱邻居、source detail、PDF 图片检索
```

流程：

```text
PDF
  -> book-native 页面/物种页解析
  -> taxa.jsonl
  -> page_units.jsonl
  -> rag_chunks.jsonl
  -> image_assets.jsonl
  -> graph_edges.jsonl
  -> RAG-Anything MCP / query_adapter 查询
```

相关文件：

```text
raganything_cli.py
src/aquabio_raganything/book_native.py
src/aquabio_raganything/image_rag.py
src/aquabio_raganything/query_adapter.py
src/aquabio_raganything/mcp_server.py
```

常用构建命令：

```cmd
cd /d F:\rag\AquaBio-AgentRAG

.\.venv\Scripts\python.exe raganything_cli.py book-native
.\.venv\Scripts\python.exe raganything_cli.py image-assets
.\.venv\Scripts\python.exe raganything_cli.py index-images
.\.venv\Scripts\python.exe raganything_cli.py status
```

查询命令：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 分布图" --entity "Jasus lalandii"

.\.venv\Scripts\python.exe raganything_cli.py neighbors --entity "Jasus lalandii"

.\.venv\Scripts\python.exe raganything_cli.py query --mode hybrid --query "Jasus lalandii taxonomy image page evidence"
```

---

## 5. 用户询问时的总调用链

### 5.1 前端调用链

```text
浏览器 UI
  -> FastAPI
  -> src/aquabio_web/api.py
  -> src/aquabio_web/service.py / ChatService.chat()
  -> AquaBioMRAGWorkflow.invoke()
  -> LangGraph
  -> 返回 answer、evidence、trace、tool_calls
  -> presentation.py 转成前端消息
```

### 5.2 CLI 调用链

```text
mrag_cli.py ask
  -> MRAGPaths / MRAGSettings
  -> AquaBioMRAGWorkflow.invoke()
  -> LangGraph
  -> 输出 final_answer
  -> 如果 --json，输出完整 state
```

常用命令：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo --query "海星有哪些外观特征？"

.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_json --query "海星有哪些外观特征？" --json

.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_mcp --query "海星有哪些外观特征？" --mcp-retrieval --json

.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_image --query "这是什么生物？" --image "data\mrag\images\starfish\img_starfish_001.jpg" --json
```

---

## 6. LangGraph 节点名称、位置和功能

核心文件：

```text
src/aquabio_mrag/workflow.py
```

关键节点：

| 节点 | 代码位置 | 功能 |
|---|---:|---|
| `session_init_node` | `workflow.py:203` | 初始化 session_id、trace、运行状态 |
| `memory_load_node` | `workflow.py:213` | 读取历史会话，用于多轮追问 |
| `followup_resolver_node` | `workflow.py:244` | 解析“刚才那个”“它”等指代 |
| `router_node` | `workflow.py:323` | 判断问题类型，如 text_qa、image_qa、pdf_qa |
| `rewrite_node` | `workflow.py:398` | 改写查询，补充英文名、学名、检索词 |
| `source_selection_node` | `workflow.py:427` | 根据任务类型选择抽象工具 |
| `react_tool_plan_node` | `workflow.py:463` | 生成/确认工具执行计划 |
| `clarification_node` | `workflow.py:506` | 信息不足时触发人工补充 |
| `vision_node` | `workflow.py:542` | 调视觉模型分析上传图片，生成 caption 和候选物种 |
| `retrieval_node` | `workflow.py:658` | 执行检索，调用 RetrievalAgent 和 MCP tools |
| `rerank_node` | `workflow.py:1223` | 证据重排 |
| `context_node` | `workflow.py:1245` | 构建给大模型的证据上下文 |
| `answer_node` | `workflow.py:1287` | 调 Qwen 生成最终回答 |
| `response_guard_node` | `workflow.py:1547` | 回答安全和格式守卫 |
| `evaluation_node` | `workflow.py:1574` | 自动评估回答是否满足证据要求 |
| `finalize_node` | `workflow.py:1703` | 汇总最终状态 |
| `memory_save_node` | `workflow.py:1716` | 保存本轮对话和证据到会话记忆 |
| `invoke` | `workflow.py:1976` | 外部调用入口 |

### 6.1 LangGraph 主流程

```text
START
  -> session_init
  -> memory_load
  -> followup_resolver
  -> router
  -> rewrite
  -> source_selection
  -> react_tool_plan
  -> vision，可选
  -> retrieval_agent
  -> answer_agent
  -> retry，可选
  -> finalize
  -> memory_save
  -> END
```

### 6.2 retrieval 子图

```text
retrieve
  -> rerank
  -> build_context
```

对应：

```text
retrieval_node
rerank_node
context_node
```

### 6.3 answer 子图

```text
generate
  -> guard
  -> evaluate
```

对应：

```text
answer_node
response_guard_node
evaluation_node
```

---

## 7. MCP Server 与 tools

项目有两个 MCP Server。

### 7.1 Chroma MCP Server

文件：

```text
src/aquabio_mrag/mcp_server.py
```

作用：

```text
把 Chroma 向量库检索能力暴露成 MCP tools。
```

工具：

| Tool | 代码位置 | 功能 |
|---|---:|---|
| `search_species_text` | `mcp_server.py:34` | 查物种文本、生活习性、外观特征 |
| `search_image_captions` | `mcp_server.py:42` | 查图片 caption 和相似图片文本 |
| `search_pdf_entity_images` | `mcp_server.py:50` | 查 PDF 实体图片，旧工具名 |
| `get_entity_sample_images` | `mcp_server.py:60` | 按实体读取样例图，旧工具名 |
| `search_multimodal` | `mcp_server.py:67` | 图文联合检索 |
| `search_pdf` | `mcp_server.py:87` | 普通 PDF chunk 检索 |
| `generate_image_caption` | `mcp_server.py:101` | 调视觉模型生成图片描述 |
| `get_source_detail` | `mcp_server.py:111` | 按 doc_id 查 Chroma 文档详情 |

### 7.2 RAG-Anything MCP Server

文件：

```text
src/aquabio_raganything/mcp_server.py
```

作用：

```text
把 PDF 图谱、PDF 图片、页码、来源详情暴露成 MCP tools。
```

工具：

| Tool | 代码位置 | 功能 |
|---|---:|---|
| `raganything_index_status` | `mcp_server.py:33` | 查看图谱/索引状态 |
| `raganything_graph_neighbors` | `mcp_server.py:39` | 查实体邻居和分类关系 |
| `raganything_hybrid_search` | `mcp_server.py:45` | LightRAG 风格 hybrid 检索 |
| `raganything_image_search` | `mcp_server.py:53` | 查 PDF 图片，旧/兼容名 |
| `search_pdf_images` | `mcp_server.py:61` | 查 PDF 图片，主流程推荐工具 |
| `raganything_entity_images` | `mcp_server.py:69` | 按实体直接取 PDF 图片 |
| `raganything_source_detail` | `mcp_server.py:76` | 查来源详情，旧/兼容名 |
| `get_source_detail` | `mcp_server.py:84` | 查 PDF 页、图片、关系、来源详情 |

---

## 8. tools 会不会根据问题自动调用

答案：会，但调用者是 LangGraph，不是 MCP Server 自己。

### 8.1 正确理解

```text
用户问题
  -> router_node 判断任务类型
  -> source_selection_node 选择抽象工具
  -> react_tool_plan_node 确认执行计划
  -> retrieval_node 执行
  -> RetrievalAgent 判断调用本地检索还是 MCP
  -> MCPClient.call_tool_sync()
  -> 具体 MCP tool
```

### 8.2 举例：用户问 PDF 书中图片和页码

用户问题：

```text
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。
```

自动流程：

```text
router_node
  -> pdf_qa

source_selection_node
  -> pdf_retriever
  -> pdf_image_retriever
  -> image_retriever

retrieval_node
  -> search_pdf_images
  -> raganything_graph_neighbors
  -> get_source_detail
  -> raganything_hybrid_search
```

真实 tool_calls 应该看到：

```text
mcp_tool_call:search_pdf_images status=success
mcp_tool_call:raganything_graph_neighbors status=success
mcp_tool_call:get_source_detail status=success
mcp_tool_call:raganything_hybrid_search status=success
```

### 8.3 举例：用户只问普通知识

用户问题：

```text
海星有哪些外观特征？
```

默认可能是：

```text
router_node
  -> text_qa

source_selection_node
  -> text_retriever

retrieval_node
  -> 本地 Chroma 检索
```

如果加 `--mcp-retrieval`：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "海星有哪些外观特征？" --mcp-retrieval --json
```

则 Chroma 文本检索也会通过 MCP：

```text
mcp_tool_call:search_species_text status=success
```

---

## 9. 期望 trace 是什么意思

`trace` 是系统运行轨迹。

它保存在 workflow state 里：

```text
state["trace"]
```

如果用命令：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "..." --json
```

就能看到完整 JSON，其中包含：

```text
trace
tool_calls
route
selected_tools
evidence
final_answer
```

### 9.1 期望 trace 不是装饰

它是验收标准。

例如某文档写：

```text
期望 trace:
router:pdf_qa
mcp_tool_call:search_pdf_images status=success
```

意思是：

```text
如果这个场景真的正确运行，
trace 里应该出现 router:pdf_qa，
tool_calls 或 trace 里应该出现 search_pdf_images，
并且 status 应该是 success。
```

### 9.2 为什么 trace 可能不完全一样

因为系统有：

```text
规则路由
ReAct 工具计划
LLM/native function calling 工具重排
是否开启 --mcp-retrieval
是否上传图片
是否要求 PDF 证据
是否命中本地 PDF 图片
```

所以不同运行中顺序可能不同。

验收看核心字段：

```text
router 是否正确
selected_tools 是否包含需要的抽象工具
tool_calls 是否出现真实 MCP tool
status 是否 success 或明确 failed
result_count 是否合理
answer 是否引用了对应证据
```

---

## 10. 不同问题对应的 LangGraph 和 tools

### 10.1 普通文本问答

用户：

```text
海星有哪些外观特征？
```

LangGraph：

```text
session_init
memory_load
followup_resolver
router:text_qa
rewrite
source_selection:text_retriever
react_tool_plan
retrieval_agent
answer_agent
memory_save
```

tools：

```text
默认：
  本地 Chroma text retrieval

--mcp-retrieval：
  Chroma MCP search_species_text
```

### 10.2 上传图片识别

用户：

```text
这是什么生物？
```

输入：

```text
image_path != null
```

LangGraph：

```text
router:image_qa 或 multimodal_qa
source_selection:vlm_caption,text_retriever,image_retriever,multimodal_retriever
vision_node
retrieval_agent
answer_agent
memory_save
```

tools：

```text
视觉模型：
  Qwen/Gemini/OpenRouter VLM

基础检索：
  search_species_text
  search_image_captions
  search_multimodal
```

注意：

```text
默认模式下基础检索可能走本地函数；
--mcp-retrieval 下会走 Chroma MCP tools。
```

### 10.3 上传图片并要求书中证据

用户：

```text
识别这张图片里的生物，并给出 Field Guide 中的页码、分类关系和书中图片。
```

LangGraph：

```text
vision_node
retrieval_agent
pdf_image_retriever
pdf_retriever
answer_agent
```

tools：

```text
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

### 10.4 指定学名查 PDF 图书图片

用户：

```text
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。
```

LangGraph：

```text
router:pdf_qa
source_selection:pdf_image_retriever,pdf_retriever,image_retriever
retrieval_agent
answer_agent
```

tools：

```text
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

### 10.5 多轮追问

第一轮：

```text
这是什么生物？
```

第二轮：

```text
刚才那个生物的分布图给我。
```

LangGraph：

```text
memory_load_node
followup_resolver_node
router_node
source_selection_node
retrieval_node
memory_save_node
```

tools：

```text
如果追问分布图：
  search_pdf_images

如果追问颜色/生活习性：
  search_species_text
```

---

## 11. 前端/后端/CLI 运行命令

### 11.1 启动项目

建议用 cmd：

```cmd
cd /d F:\rag\AquaBio-AgentRAG
stop_chat_assistant.cmd
start_chat_assistant.cmd
```

访问：

```text
UI:  http://127.0.0.1:8510
API: http://127.0.0.1:8000/docs
```

### 11.2 检查服务状态

```cmd
curl http://127.0.0.1:8000/api/system/status
```

PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/system/status
```

### 11.3 查看 MCP tools

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

### 11.4 命令行普通问答

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_text --query "海星有哪些外观特征？"
```

### 11.5 命令行查看完整 trace

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_text_json --query "海星有哪些外观特征？" --json
```

### 11.6 强制 Chroma 检索也走 MCP

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_text_mcp --query "海星有哪些外观特征？" --mcp-retrieval --json
```

### 11.7 上传图片识别

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_image --query "这是什么生物？有什么识别依据？" --image "data\mrag\images\starfish\img_starfish_001.jpg" --json
```

### 11.8 查 PDF 图书图片和图谱

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_jasus --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。" --json
```

### 11.9 直接查 RAG-Anything 图片

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 分布图" --entity "Jasus lalandii"
```

### 11.10 直接查图谱邻居

```cmd
.\.venv\Scripts\python.exe raganything_cli.py neighbors --entity "Jasus lalandii"
```

---

## 12. 教学顺序建议

### 第 1 课：项目是什么

讲：

```text
水下生物多模态 RAG
文本 + 图片 + PDF + 图谱 + 多轮对话
```

看：

```text
README.md
docs/完整流程与向量库.md
```

### 第 2 课：数据和向量库

讲：

```text
什么是 chunk
什么是 embedding
什么是 Chroma
什么是图片 caption
```

看：

```text
data/mrag/
src/aquabio_mrag/vector_db.py
src/aquabio_mrag/data_pipeline.py
```

### 第 3 课：PDF 如何变成 RAG

讲：

```text
PDF -> 页面 -> 物种页 -> 文本 chunk -> 图片资产 -> 图谱边 -> 可检索证据
```

看：

```text
raganything_cli.py
src/aquabio_raganything/book_native.py
src/aquabio_raganything/image_rag.py
docs/项目从PDF建库到图文问答逐步教学.md
```

### 第 4 课：LangGraph 工作流

讲：

```text
state
node
edge
router
source_selection
retrieval
answer
memory
```

看：

```text
src/aquabio_mrag/workflow.py
```

### 第 5 课：MCP tools

讲：

```text
MCP Server 是工具服务
MCP Client 是工具调用者
LangGraph 决定调哪个工具
```

看：

```text
src/aquabio_mrag/mcp_client.py
src/aquabio_mrag/mcp_server.py
src/aquabio_raganything/mcp_server.py
```

跑：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

### 第 6 课：多轮记忆

讲：

```text
session_id
history turns
followup_resolver
memory_save
```

看：

```text
src/aquabio_mrag/conversation.py
src/aquabio_mrag/workflow.py:213
src/aquabio_mrag/workflow.py:244
src/aquabio_mrag/workflow.py:1716
```

跑：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_memory --query "这是什么生物？" --image "data\mrag\images\starfish\img_starfish_001.jpg"
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_memory --query "刚才那个生物的分布图给我。" --json
```

### 第 7 课：如何判断系统是不是真的工作

讲：

```text
不要只看最终回答
要看 route
要看 selected_tools
要看 trace
要看 tool_calls
要看 evidence
```

跑：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session class_jasus_check --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。" --json
```

检查：

```text
route.task_type 是否是 pdf_qa
selected_tools 是否包含 pdf_image_retriever / pdf_retriever
tool_calls 是否包含 search_pdf_images / get_source_detail / raganything_graph_neighbors
trace 是否出现 mcp_tool_call:...
final_answer 是否有页码、分类、图片说明
```

---

## 13. 学生开发时最容易混淆的点

### 13.1 MCP tools 不会自己决定什么时候调用

错误理解：

```text
MCP Server 会自动读用户问题，然后自己调用工具。
```

正确理解：

```text
LangGraph 读用户问题；
router/source_selection 决定要什么工具；
RetrievalAgent 通过 MCPClient 调 tool；
MCP Server 只负责执行被调用的 tool。
```

### 13.2 `期望 trace` 不是固定日志模板

错误理解：

```text
trace 必须和文档一字不差。
```

正确理解：

```text
trace 是验收清单。
只要核心节点、核心 tool、status、result_count 正确，就算符合。
```

### 13.3 默认模式和 `--mcp-retrieval` 不一样

默认模式：

```text
基础 Chroma 检索可能走本地函数；
PDF 图谱和 PDF 图片优先走 RAG-Anything MCP。
```

强制 MCP 模式：

```text
Chroma 文本、图片 caption、多模态检索也走 MCP。
```

### 13.4 图片识别不是直接回答

正确流程：

```text
上传图片
  -> vision_node 生成 caption / 候选物种
  -> 用 caption 查 RAG
  -> 查 PDF 图片/图谱，可选
  -> Qwen 综合证据回答
```

---

## 14. 一个完整例子：Jasus lalandii

用户问：

```text
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。
```

期望 LangGraph：

```text
session_init
memory_load
followup_resolver
router:pdf_qa
rewrite
source_selection:pdf_image_retriever,pdf_retriever,image_retriever
react_tool_plan
retrieval_agent
answer_agent
finalize
memory_save
```

期望 MCP tools：

```text
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

期望回答：

```text
物种：Jasus lalandii
常见名：West coast rock lobster
书中图片：物种例图 + 分布图
PDF page：149
printed page：146
分类关系：Arthropoda / Crustacea / Malacostraca / Decapoda / Palinuridae / Jasus / lalandii
证据引用：来自 Field Guide PDF
```

验收命令：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session verify_jasus_doc --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。" --json
```

验收 trace：

```text
router:pdf_qa
source_selection:...;flow=pdf_image_graph_evidence
mcp_tool_call:search_pdf_images status=success
mcp_tool_call:raganything_graph_neighbors status=success
mcp_tool_call:get_source_detail status=success
answer_generation:qwen:qwen3.7-plus
```

---

## 15. 最后总结

这个项目应该按四层理解：

```text
第一层：数据层
  PDF、图片、文本、caption、chunk、图谱边、会话历史

第二层：检索层
  Chroma 向量库 + RAG-Anything/LightRAG 图谱和 PDF 图片索引

第三层：工具层
  Chroma MCP tools + RAG-Anything MCP tools

第四层：Agent 层
  LangGraph 判断问题、选择工具、调用检索、组织证据、生成回答、保存记忆
```

给学生最重要的一句话：

```text
Agentic RAG 不是“把资料丢进向量库然后问模型”，
而是让系统先判断任务，再选择合适工具，再拿证据，再让大模型回答，并且每一步都能在 trace 里被检查。
```
