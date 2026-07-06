# 下一步：LangGraph 场景路由与 MCP Tools 调用设计

本文对应文件 `下一步mcp 和langgraph` 中提出的目标：让 AquaBio-AgentRAG 不只是“有 LangGraph、有 MCP”，而是能根据不同用户问题，走不同 LangGraph 分支，调用不同 MCP tools，并能在 trace / tool_calls 中证明真实执行。

---

## 1. 总原则

一句话：

```text
LangGraph 负责判断“这是什么问题、下一步做什么”；
MCP tools 负责真正执行“查文本、查图片、查 PDF、查图谱、查来源”。
```

当前项目中的真实链路：

```text
用户问题 / 图片 / PDF
  -> FastAPI / Streamlit
  -> ChatService.chat()
  -> AquaBioMRAGWorkflow.invoke()
  -> LangGraph StateGraph
  -> router_node 判断问题类型
  -> source_selection_node 选择工具
  -> react_tool_plan_node 确认工具计划
  -> retrieval_node 调 RetrievalAgent
  -> RetrievalAgent 通过 MCPClient 调 MCP tools
  -> context_node 组装证据
  -> answer_node 调 Qwen 生成回答
  -> memory_save_node 保存会话记忆
```

关键代码：

```text
src/aquabio_mrag/workflow.py
src/aquabio_mrag/retrieval_agent.py
src/aquabio_mrag/mcp_client.py
src/aquabio_mrag/mcp_server.py
src/aquabio_raganything/mcp_server.py
src/aquabio_raganything/query_adapter.py
```

---

## 1.1 运行命令总表

### A. 启动完整聊天助手

建议在 **cmd** 里运行 `.cmd`，因为项目启动脚本本身会再调用 PowerShell；当然 PowerShell 也能运行，但教学时 cmd 更直观。

```cmd
cd /d F:\rag\AquaBio-AgentRAG
stop_chat_assistant.cmd
start_chat_assistant.cmd
```

启动后打开：

```text
前端 UI: http://127.0.0.1:8510
后端 API: http://127.0.0.1:8000/docs
```

### B. 检查后端是否活着

cmd：

```cmd
curl http://127.0.0.1:8000/api/system/status
```

PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/system/status
```

### C. 查看 MCP 工具是否能列出来

```cmd
cd /d F:\rag\AquaBio-AgentRAG
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

如果能看到 `search_species_text`、`search_pdf_images`、`raganything_graph_neighbors`、`get_source_detail`，说明 Chroma MCP 和 RAG-Anything MCP 的工具注册是正常的。

### D. 命令行问答：普通模式

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_text --query "海星有哪些外观特征？"
```

### E. 命令行问答：输出完整 JSON 和 trace

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_text_json --query "海星有哪些外观特征？" --json
```

### F. 命令行问答：强制 Chroma 检索也走 MCP

默认情况下，PDF 图谱和 PDF 图片会走 RAG-Anything MCP；基础文本、图片 caption、多模态 Chroma 检索为了速度可以走本地函数。

如果教学时想明确看到 `search_species_text`、`search_image_captions`、`search_multimodal` 这些 Chroma MCP 工具被调用，使用：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_text_mcp --query "海星有哪些外观特征？" --mcp-retrieval --json
```

新增参数位置：

```text
mrag_cli.py
  --mcp-retrieval
    -> AquaBioMRAGWorkflow.invoke(..., options={"mcp_retrieval_enabled": True})
    -> workflow.retrieval_node()
    -> RetrievalAgent.search_text_mcp() / search_image_mcp() / search_multimodal_mcp()
    -> MCPClient.call_tool()
```

### G. 上传图片识别命令

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_image --query "这是什么生物？这张图里是什么水下生物，有什么识别依据？" --image "data\mrag\images\starfish\img_starfish_001.jpg" --json
```

如果要强制图片识别后的 Chroma 检索也走 MCP：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_image_mcp --query "这是什么生物？这张图里是什么水下生物，有什么识别依据？" --image "data\mrag\images\starfish\img_starfish_001.jpg" --mcp-retrieval --json
```

### H. PDF 图谱与书中图片查询

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_jasus --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。" --json
```

这个命令应该触发：

```text
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

### I. 只查 RAG-Anything 图片索引

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 分布图" --entity "Jasus lalandii"
```

### J. 查看历史会话

```cmd
.\.venv\Scripts\python.exe mrag_cli.py history --session demo_jasus --json
```

清空某个会话：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py clear-session --session demo_jasus
```

---

## 1.2 三个教学字段分别是什么意思

文档里每个场景都会写三块：`LangGraph 流程`、`工具流`、`期望 trace`。它们不是同一个东西。

### LangGraph 流程

它表示 **状态图节点怎样流转**，回答“这个问题在 Agent 大脑里先判断什么、再做什么”。

例如上传图片识别：

```text
router: image_qa 或 multimodal_qa
source_selection
vision_node
retrieval_agent
answer_agent
memory_save
```

这说明系统不是上传图后直接回答，而是：

```text
先路由 -> 再调用视觉模型 -> 再把视觉 caption 写入 state -> 再检索 RAG -> 最后生成回答并保存记忆
```

真实代码位置：

```text
src/aquabio_mrag/workflow.py
  build_graph()
  router_node()
  source_selection_node()
  vision_node()
  retrieval_node()
  answer_node()
  memory_save_node()
```

### 工具流

它表示 **某个抽象工具最终对应哪个真实工具或模型调用**。

例如：

```text
pdf_image_retriever
  -> search_pdf_images
```

意思是 LangGraph state 里选择的是 `pdf_image_retriever` 这个抽象能力，真正执行时会通过 MCP 调用 RAG-Anything 的 `search_pdf_images`。

真实代码位置：

```text
src/aquabio_mrag/workflow.py
  retrieval_node()

src/aquabio_mrag/retrieval_agent.py
  search_pdf_images()
  retrieve()

src/aquabio_mrag/mcp_client.py
  call_tool_sync()
```

### 期望 trace

它表示 **运行后应该在 state["trace"] 或 state["tool_calls"] 里看到的证据**。

它不是伪代码，而是验收标准。比如：

```text
mcp_tool_call:search_pdf_images status=success results=2
```

看到这行，才说明系统真的调用了 MCP 工具 `search_pdf_images`，不是只在文档里写了这个名字。

注意：`source_selection` 的工具顺序可能和 `react_tool_plan` 的工具顺序不同，这是正常的。`source_selection` 是规则路由先给出的工具集合，`react_tool_plan` 会根据用户问题再重排或补充执行顺序。验收时重点看：

```text
tool_calls 里是否出现了正确 tool_name
status 是否 success
result_count 是否合理
最终证据是否来自对应 source
```

---

## 2. 已实现的 MCP Tools

### 2.1 Chroma MCP Server

文件：

```text
src/aquabio_mrag/mcp_server.py
```

工具：

| MCP Tool | 作用 | 适合问题 |
|---|---|---|
| `search_species_text` | 查基础物种卡片和长文本 chunk | 普通知识问答、生活习性、外观特征 |
| `search_image_captions` | 查基础图片 caption 和图片文本对 | 上传图后找相似图、普通图像解释 |
| `search_multimodal` | 查图文联合证据 | 图文问题、图片识别后补充文本 |
| `search_pdf` | 查普通 PDF chunk | 需要 PDF 文本证据但不强求图谱 |
| `generate_image_caption` | 调 VLM 分析上传图片 | 图片识别 |
| `search_pdf_entity_images` | 查 PDF 图片 | 旧工具名，推荐新流程使用 RAG-Anything MCP 的 `search_pdf_images` |
| `get_entity_sample_images` | 按实体取 PDF 图片 | 旧工具名，推荐新流程用 `search_pdf_images` |
| `get_source_detail` | 查 Chroma doc 原始详情 | Chroma 文档追踪 |

### 2.2 RAG-Anything MCP Server

文件：

```text
src/aquabio_raganything/mcp_server.py
```

工具：

| MCP Tool | 作用 | 适合问题 |
|---|---|---|
| `search_pdf_images` | 查 PDF 中抽取并绑定实体的真实图片 | “给我书中图片/分布图/实例图” |
| `raganything_graph_neighbors` | 查实体周围图关系 | “分类关系/关系路径/属于哪个类群” |
| `get_source_detail` | 查物种页、chunk、图片、关系、页码 | “页码/PDF证据/来源详情” |
| `raganything_hybrid_search` | LightRAG hybrid 检索 | “从图谱和语义一起查证据” |
| `raganything_index_status` | 查图谱和持久化状态 | 系统诊断 |
| `raganything_entity_images` | 按实体直接取图片 | 实体图片补充 |

---

## 3. 当前代码中场景路由位置

### 3.1 Router

文件：

```text
src/aquabio_mrag/workflow.py
```

核心函数：

```text
router_node()
```

它负责把问题分成：

```text
text_qa
followup_text_qa
image_qa
multimodal_qa
comparison_qa
source_trace
pdf_qa
```

### 3.2 Source Selection

核心函数：

```text
source_selection_node()
```

它负责把任务类型转换成工具集合：

```text
vlm_caption
text_retriever
image_retriever
pdf_image_retriever
multimodal_retriever
pdf_retriever
```

现在 trace 中会额外写入场景流：

```text
source_selection:text_retriever,pdf_retriever;flow=comparison_text_pdf_graph
```

场景流标签：

```text
text_basic_rag
image_caption_lookup
multimodal_identification
pdf_image_lookup
pdf_graph_evidence
pdf_image_graph_evidence
comparison_text_pdf_graph
multimodal_pdf_image_graph
no_retrieval
```

---

## 4. 不同用户问题应该调用哪些工具

### 场景 1：普通文本知识问答

用户问题：

```text
海星有哪些外观特征？
海马有什么生活习性？
鲨鱼一般生活在哪里？
```

LangGraph 流程：

```text
session_init
memory_load
followup_resolver
router: text_qa
query_rewrite
source_selection: text_retriever
react_tool_plan
retrieval_agent
answer_agent
memory_save
```

工具流：

```text
text_retriever
  -> 默认本地 Chroma
  -> 如果 mcp_retrieval_enabled=true，则调 Chroma MCP:
       search_species_text
```

期望 trace：

```text
router:text_qa
source_selection:text_retriever;flow=text_basic_rag
mcp_tool_call:search_species_text status=success ...   # 仅 MCP 检索模式出现
```

适合教学说明：

```text
这个场景主要证明 LangGraph 基础流程和文本 RAG。
不一定必须走 RAG-Anything MCP。
```

---

### 场景 2：对比类问题

用户问题：

```text
海星和海胆怎么区分？
海参和海胆有什么区别？请给 PDF 证据。
鲨鱼和鳐鱼从外观上怎么区分？
```

LangGraph 流程：

```text
router: comparison_qa
source_selection:
  text_retriever
  pdf_retriever
```

工具流：

```text
text_retriever
  -> search_species_text

pdf_retriever
  -> Chroma PDF / BM25
  -> raganything_graph_neighbors
  -> get_source_detail
  -> raganything_hybrid_search
```

如果问题里还要求图片：

```text
再加：
pdf_image_retriever
  -> search_pdf_images
```

期望 trace：

```text
router:comparison_qa
source_selection:text_retriever,pdf_retriever;flow=comparison_text_pdf_graph
mcp_tool_call:raganything_graph_neighbors status=success
mcp_tool_call:get_source_detail status=success
mcp_tool_call:raganything_hybrid_search status=success/failed
```

回答内容应该包括：

```text
外观差异
分类差异
PDF 页码证据
可选图关系
```

---

### 场景 3：上传图片识别

用户问题：

```text
这是什么生物？
这张图里是什么水下生物，有什么识别依据？
```

输入：

```text
image_path != null
```

LangGraph 流程：

```text
router: image_qa 或 multimodal_qa
source_selection:
  vlm_caption
  text_retriever
  image_retriever
  multimodal_retriever
  pdf_retriever
vision_node
retrieval_agent
answer_agent
memory_save
```

工具流：

```text
vlm_caption
  -> Qwen/Gemini/OpenRouter 视觉模型

text_retriever
  -> search_species_text

image_retriever
  -> search_image_captions

multimodal_retriever
  -> search_multimodal

pdf_retriever
  -> raganything_graph_neighbors / get_source_detail / hybrid_search
```

期望 trace：

```text
router:multimodal_qa
source_selection:vlm_caption,text_retriever,image_retriever,multimodal_retriever,pdf_retriever
vision:completed
mcp_tool_call:search_species_text ...       # 使用 --mcp-retrieval 时出现
mcp_tool_call:search_image_captions ...     # 使用 --mcp-retrieval 时出现
mcp_tool_call:search_multimodal ...         # 使用 --mcp-retrieval 时出现
memory_save:...
```

教学重点：

```text
图片识别不是直接回答；
先由 vision_node 生成 caption，再把 caption 写入 state，
后续检索节点再根据 caption 和用户问题查 RAG。
```

重要区别：

```text
默认前端 / 默认 CLI:
  vision_node 会真实调用视觉模型；
  基础 Chroma 文本、图片 caption、多模态检索可能走本地函数；
  PDF 图谱、PDF 图片仍优先走 RAG-Anything MCP。

CLI 加 --mcp-retrieval:
  search_species_text / search_image_captions / search_multimodal 也会走 Chroma MCP；
  这时 trace 和 tool_calls 更适合教学验收。
```

---

### 场景 4：图片 + 要书中图片 / PDF 证据

用户问题：

```text
请识别这张图片里的生物，并从 Field Guide 中找出它的分类关系、PDF 页码证据和书中相关图片。
```

这是最适合展示系统复杂度的场景。

LangGraph 流程：

```text
router: multimodal_qa
source_selection:
  vlm_caption
  text_retriever
  image_retriever
  pdf_image_retriever
  multimodal_retriever
  pdf_retriever
```

MCP tools：

```text
search_species_text
search_image_captions
search_multimodal
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

期望 trace：

```text
source_selection:...;flow=multimodal_pdf_image_graph
vision:completed
mcp_tool_call:search_pdf_images status=success results>0
mcp_tool_call:raganything_graph_neighbors status=success
mcp_tool_call:get_source_detail status=success
```

回答应该包含：

```text
上传图片识别结论
视觉判断依据
书中对应物种图片
PDF 页码
分类关系
证据引用
```

---

### 场景 5：指定拉丁学名查书中图片

用户问题：

```text
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。
给我 Anseropoda grandis 的分布图。
AnsGra 的中文名是什么，分布图给我。
```

LangGraph 流程：

```text
router: pdf_qa
source_selection:
  image_retriever
  pdf_image_retriever
  pdf_retriever
```

MCP tools：

```text
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

Jasus lalandii 预期结果：

```text
search_pdf_images:
  imgdoc_sa_taxon_jaslal_p0149_img_03
  role = specimen_overview
  PDF page = 149
  printed page = 146

  imgdoc_sa_taxon_jaslal_p0149_img_04
  role = distribution_map
  PDF page = 149
  printed page = 146

graph_neighbors:
  Jasus lalandii --is_a--> Jasus
  Jasus lalandii --is_a--> Palinuridae
  Jasus lalandii --is_a--> Decapoda
  Jasus lalandii --is_a--> Malacostraca
  Jasus lalandii --is_a--> Arthropoda
```

期望 trace：

```text
router:pdf_qa
source_selection:image_retriever,pdf_image_retriever,pdf_retriever;flow=pdf_image_graph_evidence
mcp_tool_call:search_pdf_images status=success results=2
mcp_tool_call:raganything_graph_neighbors status=success results=12
mcp_tool_call:get_source_detail status=success results=12
```

---

### 场景 6：只要分布图 / 实例图

用户问题：

```text
给我海星的分布图。
给我章鱼的书中实例图。
给我 AnsGra 的分布图和生物例图。
```

LangGraph 流程：

```text
router: pdf_qa 或 text_qa + image intent
source_selection:
  image_retriever
  pdf_image_retriever
```

MCP tools：

```text
search_pdf_images
```

图片角色识别：

```text
分布图 -> distribution_map
实例图 / 生物例图 / 参考图 -> specimen / specimen_overview / specimen_detail
```

关键代码：

```text
src/aquabio_raganything/image_rag.py
  requested_image_roles()
  role_matches()
  select_requested_roles()
```

期望 trace：

```text
source_selection:image_retriever,pdf_image_retriever;flow=pdf_image_lookup
mcp_tool_call:search_pdf_images status=success
```

如果本地 PDF 图片没有命中：

```text
系统才允许走网络图片 fallback。
```

---

### 场景 7：来源追踪 / 页码追踪

用户问题：

```text
这个结论来自哪一页？
从 PDF 图谱中查找外观特征并给出来源页码。
Jasus lalandii 的分类和描述在哪个 PDF 页？
```

LangGraph 流程：

```text
router: source_trace 或 pdf_qa
source_selection:
  pdf_retriever
```

MCP tools：

```text
get_source_detail
raganything_graph_neighbors
raganything_hybrid_search
```

期望 trace：

```text
router:source_trace 或 router:pdf_qa
mcp_tool_call:get_source_detail status=success
```

回答必须包含：

```text
PDF 文件名
PDF page
printed page，如果有
doc_id / unit_id
证据片段
```

---

### 场景 8：多轮追问

第一轮：

```text
这是什么生物？
```

上传图片。

第二轮：

```text
刚才那个生物的分布图给我。
刚才那个常见颜色是什么？只回答颜色。
```

LangGraph 流程：

```text
memory_load
followup_resolver
router
source_selection
retrieval_agent
memory_save
```

工具选择：

```text
如果只是颜色：
  search_species_text

如果要分布图：
  search_pdf_images
  get_source_detail，可选
```

期望 trace：

```text
memory_load:history>0
followup_resolver:True:starfish
source_selection:...
memory_save:...
```

教学重点：

```text
MCP tool 不负责理解“刚才那个”。
“刚才那个”由 LangGraph 的 memory_load_node + followup_resolver_node 解析。
```

---

## 5. 代码改动说明

### 5.1 统一 RAG-Anything MCP 工具名

文件：

```text
src/aquabio_mrag/retrieval_agent.py
```

现在统一使用：

```text
search_pdf_images
get_source_detail
raganything_graph_neighbors
raganything_hybrid_search
```

不再在主流程里记录旧名：

```text
raganything_image_search
raganything_source_detail
```

这样前端和文档看到的工具名，与用户理解的工具名一致。

### 5.2 Chroma MCP 检索也返回 tool_calls

文件：

```text
src/aquabio_mrag/retrieval_agent.py
```

这些方法现在返回：

```text
(rows, meta)
```

而不是只返回 rows：

```text
search_text_mcp()
search_image_mcp()
search_multimodal_mcp()
search_pdf_chunks_mcp()
```

meta 包含：

```json
{
  "counts": {"mcp_chroma_text": 12},
  "warnings": [],
  "tool_calls": [
    {
      "tool_name": "search_species_text",
      "tool_source": "mcp",
      "server": "chroma",
      "status": "success",
      "latency_ms": 300,
      "result_count": 12
    }
  ]
}
```

### 5.3 workflow.py 汇总所有 MCP 调用

文件：

```text
src/aquabio_mrag/workflow.py
```

`retrieval_node()` 现在会汇总：

```text
text_retrieval_meta.tool_calls
chroma_image_retrieval_meta.tool_calls
multimodal_retrieval_meta.tool_calls
retrieval_meta.tool_calls
image_retrieval_meta.tool_calls
```

写入：

```text
state["tool_calls"]
state["trace"]
```

trace 中能看到：

```text
mcp_tool_call:search_species_text status=success latency=... results=...
mcp_tool_call:search_pdf_images status=success latency=... results=...
mcp_tool_call:get_source_detail status=success latency=... results=...
```

### 5.4 source_selection 增加 flow 标签

文件：

```text
src/aquabio_mrag/workflow.py
```

新增：

```text
_tool_flow_label()
```

source_selection trace 示例：

```text
source_selection:image_retriever,pdf_image_retriever,pdf_retriever;flow=pdf_image_graph_evidence
```

这样学生看 trace 时，能直接知道当前问题属于哪个工具流。

---

## 6. 推荐测试问题

### 6.1 文本基础 RAG

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session flow_text --query "海星有哪些外观特征？"
```

预期：

```text
router:text_qa
flow=text_basic_rag
```

### 6.2 PDF 图片 + 图谱

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session flow_jasus --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。"
```

预期：

```text
router:pdf_qa
flow=pdf_image_graph_evidence
mcp_tool_call:search_pdf_images
mcp_tool_call:raganything_graph_neighbors
mcp_tool_call:get_source_detail
```

### 6.3 对比 + PDF 证据

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session flow_compare --query "海星和海胆怎么区分？请给出文本证据、PDF证据和图谱关系。"
```

预期：

```text
router:comparison_qa
flow=comparison_text_pdf_graph
mcp_tool_call:raganything_graph_neighbors
mcp_tool_call:raganything_hybrid_search
```

### 6.4 上传图片 + 书中证据

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session flow_image_pdf --query "请识别这张图片里的生物，并从 Field Guide 中找出分类关系、PDF 页码证据和书中相关图片。" --image "data\mrag\images\starfish\img_starfish_001.jpg"
```

预期：

```text
router:multimodal_qa
vision:completed
flow=multimodal_pdf_image_graph
mcp_tool_call:search_pdf_images
```

### 6.5 多轮追问

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session flow_memory --query "这是什么生物？" --image "data\mrag\images\starfish\img_starfish_001.jpg"
.\.venv\Scripts\python.exe mrag_cli.py ask --session flow_memory --query "刚才那个生物的分布图给我。"
```

预期第二轮：

```text
memory_load:1
followup_resolver:True:starfish
mcp_tool_call:search_pdf_images
memory_save
```

---

## 7. 前端展示建议

前端右侧或调试面板建议展示三块：

### 7.1 LangGraph Flow

```text
session_init
memory_load
followup_resolver
router:pdf_qa
source_selection:flow=pdf_image_graph_evidence
react_tool_plan
retrieval_agent
answer_agent
memory_save
```

### 7.2 MCP Tool Calls

```text
MCP raganything.search_pdf_images
  status=success
  result_count=2
  latency=217ms

MCP raganything.get_source_detail
  status=success
  result_count=12
  latency=462ms
```

### 7.3 Evidence Summary

```text
text evidence: 8
pdf evidence: 12
graph relations: 12
pdf images: 2
network fallback images: 0
```

---

## 8. 判断是否真的按场景调用 tools

不能只看最终答案，要看：

```text
trace
tool_calls
graph_trace
evidence.source_system
```

必须满足：

```text
1. router 显示正确任务类型
2. source_selection 显示正确 selected_tools 和 flow
3. tool_calls 中出现具体 MCP tool
4. tool_calls.status 是 success 或明确 failed
5. result_count 合理
6. evidence 中出现对应来源
```

例如 Jasus 问题的合格 trace：

```text
router:pdf_qa
source_selection:image_retriever,pdf_image_retriever,pdf_retriever;flow=pdf_image_graph_evidence
mcp_tool_call:search_pdf_images status=success results=2
mcp_tool_call:raganything_graph_neighbors status=success results=12
mcp_tool_call:get_source_detail status=success results=12
```

如果只有：

```text
pdf_retriever
```

这不能证明 MCP 真实调用。

---

## 9. 后续可继续增强

### 9.1 ToolExecutionNode 独立化

当前项目是：

```text
retrieval_node 内部调用 RetrievalAgent
```

更教学化的下一步可以拆成：

```text
tool_planning_node
  -> tool_execution_node
  -> evidence_normalize_node
```

这样学生更容易看到：

```text
计划工具
执行工具
整理证据
生成回答
```

### 9.2 LLM Function Calling Planner

当前是：

```text
规则路由 + 可选 react_planner
```

后续可让 Qwen 根据工具 schema 选择工具：

```text
用户问题 + available_tools
  -> Qwen function calling
  -> tool_plan
  -> MCPClient.call_tool
```

注意：即使用 LLM 选择工具，也要保留规则兜底，避免模型漏选 `search_pdf_images` 这种硬需求工具。

### 9.3 MCP Memory Tools

当前会话记忆主要用本地 `ConversationStore`。后续可加：

```text
load_session_memory
save_turn_memory
search_session_memory
```

这样记忆也能通过 MCP tools 暴露，适合多 Agent 或外部客户端复用。

---

## 10. 一句话总结

下一步的设计重点是：

```text
不是让所有问题都调用所有 MCP tools，
而是让不同问题触发不同 LangGraph flow，
每个 flow 只调用必要 tools，
并且把 tool_calls 清楚写回 state 和 trace。
```

这样系统才像真正的 Agentic RAG：

```text
会判断问题
会选择工具
会调用 MCP
会融合证据
会保存记忆
还能证明自己每一步做了什么
```

---

## 11. 本轮已实测通过的内容

### 11.1 CLI 已支持强制 MCP 检索

代码位置：

```text
mrag_cli.py:56
  新增 --mcp-retrieval

mrag_cli.py:149
  options={"mcp_retrieval_enabled": args.mcp_retrieval}
```

验证命令：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --help
```

实际能看到：

```text
--mcp-retrieval
  Route Chroma text/image/multimodal/PDF chunk retrieval through MCP tools...
```

这说明教学时可以明确切换两种模式：

```text
默认模式：
  基础 Chroma 检索可走本地函数；
  RAG-Anything PDF 图谱、PDF 图片仍走 MCP。

--mcp-retrieval 模式：
  search_species_text
  search_image_captions
  search_multimodal
  search_pdf
  这些 Chroma 检索也会通过 MCPClient 调用。
```

### 11.2 MCP 工具注册已实测可列出

验证命令：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

实际已经能列出：

```text
chroma:
  search_species_text
  search_image_captions
  search_multimodal
  search_pdf
  generate_image_caption
  get_source_detail

raganything:
  search_pdf_images
  raganything_graph_neighbors
  raganything_hybrid_search
  get_source_detail
  raganything_index_status
```

这说明 MCP 不是只写在设计文档里，工具注册层已经可访问。

### 11.3 Jasus lalandii 场景已实测符合要求

验证命令：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session verify_jasus_doc --query "给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。" --json
```

实际路由：

```json
{
  "task_type": "pdf_qa",
  "need_vlm": false,
  "need_text_retrieval": false,
  "need_image_retrieval": true,
  "need_multimodal_retrieval": false,
  "need_pdf_retrieval": true
}
```

实际选中的工具：

```json
[
  "pdf_retriever",
  "pdf_image_retriever",
  "image_retriever"
]
```

实际 MCP tool_calls：

```json
[
  {
    "tool_name": "raganything_graph_neighbors",
    "tool_source": "mcp",
    "server": "raganything",
    "status": "success",
    "result_count": 12
  },
  {
    "tool_name": "get_source_detail",
    "tool_source": "mcp",
    "server": "raganything",
    "status": "success",
    "result_count": 12
  },
  {
    "tool_name": "raganything_hybrid_search",
    "tool_source": "mcp",
    "server": "raganything",
    "status": "success",
    "result_count": 4
  },
  {
    "tool_name": "search_pdf_images",
    "tool_source": "mcp",
    "server": "raganything",
    "status": "success",
    "result_count": 2
  }
]
```

实际 trace 关键行：

```text
router:pdf_qa
source_selection:image_retriever,pdf_image_retriever,pdf_retriever;flow=pdf_image_graph_evidence
react_tool_plan:step=1;mode=native_function_calling;tools=pdf_retriever,pdf_image_retriever,image_retriever
retrieval:text=0,web=0,image=2,pair=0,pdf=12,mcp_graph=28,mcp_pdf_images=2
mcp_tool_call:raganything_graph_neighbors status=success
mcp_tool_call:get_source_detail status=success
mcp_tool_call:raganything_hybrid_search status=success
mcp_tool_call:search_pdf_images status=success
answer_generation:qwen:qwen3.7-plus
memory_save:0
```

解释：

```text
source_selection 的顺序是规则路由的工具集合；
react_tool_plan 的顺序是执行计划，可能重排；
真正判断是否调用 MCP，要看 tool_calls 和 mcp_tool_call trace。
```

这个例子已经满足用户原始要求：

```text
给我 Jasus lalandii 的书中图片、页码、分类关系和 PDF 证据。
```

因为它真实调用了：

```text
search_pdf_images
raganything_graph_neighbors
get_source_detail
raganything_hybrid_search
```

---

## 12. 场景 3：上传图片识别到底哪些是真调用

以这个场景为例：

```text
用户上传图片，并问：
这是什么生物？
这张图里是什么水下生物，有什么识别依据？
```

文档里的三块含义如下。

### 12.1 LangGraph 流程是真实节点路径

```text
router: image_qa 或 multimodal_qa
source_selection
vision_node
retrieval_agent
answer_agent
memory_save
```

这些对应真实代码：

```text
src/aquabio_mrag/workflow.py
  router_node()
  source_selection_node()
  vision_node()
  retrieval_node()
  answer_node()
  memory_save_node()
```

运行后应该在 trace 里看到类似：

```text
router:multimodal_qa
source_selection:vlm_caption,text_retriever,image_retriever,multimodal_retriever,pdf_retriever;flow=multimodal_identification
vision:completed
retrieval:...
answer_generation:qwen:qwen3.7-plus
memory_save:...
```

### 12.2 工具流是真实工具映射，但是否走 MCP 取决于模式

默认模式：

```text
vlm_caption
  -> 真实调用视觉模型

text_retriever / image_retriever / multimodal_retriever
  -> 可能走本地 Chroma 函数

pdf_retriever / pdf_image_retriever
  -> 优先走 RAG-Anything MCP
```

强制 MCP 模式：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo_image_mcp --query "这是什么生物？这张图里是什么水下生物，有什么识别依据？" --image "data\mrag\images\starfish\img_starfish_001.jpg" --mcp-retrieval --json
```

这时 Chroma 检索也应该出现 MCP tool_calls：

```text
search_species_text
search_image_captions
search_multimodal
```

### 12.3 期望 trace 是验收清单，不是保证每次字面完全一样

例如：

```text
mcp_tool_call:search_species_text status=success
mcp_tool_call:search_image_captions status=success
mcp_tool_call:search_multimodal status=success
```

只有在 `--mcp-retrieval` 模式下才应该强制要求这些 Chroma MCP trace。

如果不用 `--mcp-retrieval`，教学验收可以改看：

```text
vision:completed
retrieval:text>0 或 image>0 或 pair>0
answer_generation:qwen:qwen3.7-plus
```

如果问题要求“从 PDF 书中找图、页码、图谱关系”，还必须看到：

```text
mcp_tool_call:search_pdf_images status=success
mcp_tool_call:raganything_graph_neighbors status=success
mcp_tool_call:get_source_detail status=success
```

这才说明它不是单纯看图回答，而是完成了：

```text
上传图片 -> 视觉识别 -> RAG 检索 -> PDF 图谱/图片证据 -> 大模型组织回答
```
