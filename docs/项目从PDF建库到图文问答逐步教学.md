# AquaBio-AgentRAG：从 PDF 建库到图文问答的逐步教学

> 项目目录：`F:\rag\AquaBio-AgentRAG`  
> 目标 PDF：`data/mrag/pdfs/Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf`

本文不只给出运行命令，而是按照代码真正执行的先后顺序，解释：

1. 这本 501 页 PDF 被哪些 Python 文件处理。
2. 文本、表格、图片、分类实体和关系怎样生成。
3. 数据怎样写入 Chroma、LightRAG 和 NetworkX。
4. 用户传入纯文本、纯图片或图文问题后，代码怎样执行。
5. LangGraph、ReAct、MCP Client、Retrieval Agent 分别负责什么。
6. 推荐按什么顺序阅读和学习整个项目。

---

## 1. 先建立整体认识

项目有两个不同阶段。

```text
阶段 A：离线建库

PDF / 网络文本 / 数据集图片
  -> 解析和清洗
  -> 结构化记录
  -> chunk
  -> embedding
  -> Chroma / LightRAG / NetworkX


阶段 B：在线问答

用户文本 + 可选图片
  -> LangGraph
  -> 视觉理解
  -> ReAct 工具规划
  -> Chroma + BM25 + LightRAG 检索
  -> RRF 融合
  -> LLM 生成带证据引用的答案
```

离线建库不会因为用户问一次问题就重新解析 PDF。PDF 应先处理并持久化，在线问答只读取已经生成的索引。

---

## 2. 项目的主要入口文件

项目当前有两个 CLI 入口。

| 入口 | 用途 |
|---|---|
| `raganything_cli.py` | PDF 盘点、结构解析、RAG-Anything 入库、LightRAG 查询 |
| `mrag_cli.py` | 用户文本/图片问答、会话、HITL、MCP 工具检查 |

可以先把它们理解为：

```text
raganything_cli.py = 建 PDF 知识库和单独检查图索引
mrag_cli.py        = 用户真正提问时使用的 Agent 入口
```

---

# 第一部分：目标 PDF 是如何被处理的

## 3. PDF 处理有两条路线

对于南非海洋无脊椎动物指南，项目保留了两条 PDF 处理路线。

### 3.1 路线一：书籍原生结构精确解析

入口命令：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
```

调用顺序：

```text
raganything_cli.py
  -> RAGAnythingPaths.from_root()
  -> RAGAnythingSettings.from_env()
  -> build_book_native()
       -> parse_sa_taxa_catalog()
       -> parse_sa_species_units()
       -> _enrich_units_from_catalog()
       -> _join_catalog_and_units()
       -> _species_chunks()
       -> _relation_triples()
       -> _raganything_content_items()
       -> 写入 JSONL
```

核心文件：

```text
src/aquabio_raganything/book_native.py
```

这条路线利用该书固定版式精确恢复目录表和物种鉴定页，不调用 LLM，因此学名、页码和字段边界更稳定。

### 3.2 路线二：MinerU 通用多模态解析

入口命令：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py inventory
.\.venv-raganything\Scripts\python.exe raganything_cli.py index --scope relevant --limit 1
```

调用顺序：

```text
raganything_cli.py
  -> inventory.py
       -> build_inventory()
       -> extract_segment_pdf()
  -> indexer.py
       -> index_segments()
       -> rag.parse_document()
       -> MinerU
       -> prepare_content_list()
       -> rag.insert_content_list()
       -> LightRAG 持久化
```

这条路线适合处理任意 PDF 的文本、图片、表格和公式，但对这本固定模板图鉴而言，字段精度不如专用的 `book_native.py`。

当前推荐组合是：

```text
book_native.py 保证分类、字段和页码准确
MinerU/RAG-Anything 补充表格、图片和跨模态描述
```

---

## 4. 第一步：配置 PDF 路径

文件：

```text
src/aquabio_raganything/config.py
```

`BOOKS` 记录逻辑书籍 ID 与实际文件名：

```python
BOOKS = {
    "living_guide": "FIELD IDENTIFICATION GUIDE TO THE LIVING.pdf",
    "sa_invertebrates": (
        "Field-Guide-to-SA-Offshore-Marine-Invertebrates_"
        "web-full-version_compressed.pdf"
    ),
}
```

`RAGAnythingPaths.from_root()` 统一构造目录：

```text
data/mrag/pdfs/                       原始 PDF
data/mrag/raganything/inventory/      页面盘点
data/mrag/raganything/parser_output/  MinerU 输出
data/mrag/raganything/book_native/    精确结构化结果
data/mrag/raganything/manifests/      断点和状态
data/mrag/raganything/working/        LightRAG 持久化文件
```

这样业务代码不需要到处拼接绝对路径。

---

## 5. 第二步：PyMuPDF 打开 PDF

主函数：

```python
build_book_native(paths, book_id="sa_invertebrates")
```

内部执行：

```python
document = fitz.open(source)
```

这里使用 `PyMuPDF`，导入名是 `fitz`。

它负责读取：

- PDF 总页数。
- 每页文本。
- 文本 span 的坐标。
- 每页图片对象。
- 图片 xref、宽高和页面位置。

该 PDF 实测有 501 个物理页。

---

## 6. 第三步：恢复 Table of Taxa 分类目录

函数：

```python
parse_sa_taxa_catalog(document, source_file)
```

该书 PDF 第 27 至 39 个物理页是分类目录表。代码不是把整页文本按空格强行切割，而是先读取每个文字 span：

```python
page.get_text("dict", sort=True)
```

`_text_spans()` 保存：

```text
text
x0, y0
x1, y1
中心纵坐标 cy
字体和字号
```

目录表的列坐标定义在 `TABLE_COLUMNS`：

```text
Class
Order
Family
Genus
Species
Common name
Authority
FB Code
Page
```

具体恢复过程：

1. 在页面最右侧找到书内页码。
2. 书内物种页从 41 开始，因此小于 41 的页脚数字被排除。
3. 使用相邻页码的纵坐标中点确定每一行的上下边界。
4. 根据横坐标把同一行文本放入对应列。
5. 合并单元格内的换行文本。
6. 生成 `TaxaCatalogRow`。

每行保存：

```text
catalog_id
table_pdf_page
printed_page
expected_pdf_page
class/order/family/genus/species
scientific_name
common_name
authority
fb_code
taxon_level
```

这里有两个页码：

```text
printed_page      页面上印刷的书内页码
expected_pdf_page PDF 文件中的物理页码
```

该书通常满足：

```text
expected_pdf_page = printed_page + 3
```

目前恢复出 409 条分类目录记录。

---

## 7. 第四步：逐页解析物种鉴定页

函数：

```python
parse_sa_species_units()
```

代码从 PDF 第 40 个物理页之后逐页扫描。

标准鉴定页必须包含：

```text
distinguishing features
Phylum:
```

源书中的 `xxx` 空模板页会被排除，避免制造假物种。

每个有效页面生成一个 `SpeciesPageUnit`：

```python
class SpeciesPageUnit:
    unit_id
    doc_id
    source_file
    pdf_page
    printed_page
    title
    fb_code
    scientific_name
    common_name
    taxon_level
    taxonomy
    distinguishing_features
    colour
    size
    distribution
    similar_species
    references
    image_records
    raw_text
```

这相当于把“一张物种鉴定页”转换成一个完整、可追踪的知识单元。

---

## 8. 第五步：按标题切分正文信息

函数：

```python
_extract_labeled_sections(text)
```

代码识别这些标题：

```text
distinguishing features
Colour / Color
Size
distribution
Similar species
reference / references
```

例如 PDF 第 401 物理页的海星页面会得到：

```text
scientific_name: Coronaster volsellatus
common_name: False brisingid/Spiny pom-pom starfish
distinguishing_features: 小中央盘、细长腕、腕部有刺等
colour: 橙白、鲑红到红色
size: 半径及直径描述
distribution: 南非西海岸和深度范围
similar_species: Stegnobrisinga splendens
references: 原始参考文献
```

字段切分的价值是：用户问“颜色”时只检索颜色 chunk，不必让整页参考文献干扰相似度。

---

## 9. 第六步：解析分类层级

函数：

```python
_extract_taxonomy(text)
```

可识别：

```text
Phylum
Subphylum
Class
Subclass
Order
Suborder
Infraorder
Superfamily
Family
Subfamily
Genus
Species
Common name
```

结果保存在：

```python
unit.taxonomy
```

示例：

```json
{
  "phylum": "Echinodermata",
  "class": "Asteroidea",
  "order": "Forcipulatida",
  "family": "Asteriidae",
  "genus": "Coronaster",
  "species": "volsellatus"
}
```

---

## 10. 第七步：处理特殊版式

并非所有页面都完全相同。

书内第 382 页同时描述：

```text
Ornithoteuthis antillarum
Ornithoteuthis volatilis
```

`parse_sa_species_units()` 对 PDF 第 385 物理页做了单独处理，保存：

- 两个学名。
- 两个俗名。
- 共同分类。
- 各自分布。
- 雄性腕结构的区别。

这比把页面错误地识别成一个物种可靠。

---

## 11. 第八步：目录和正文互相校正

函数：

```python
_enrich_units_from_catalog()
```

匹配依据：

1. 优先使用书内页码 `printed_page`。
2. 再使用 `FB Code`。

校正规则：

- 正文有详细描述时，保留正文。
- 正文缺少标题或分类字段时，用目录补齐。
- 目录短文本不能覆盖正文详细字段。

之后 `_join_catalog_and_units()` 输出目录和正文的匹配审计。

当前结果：

```text
目录记录        409
页面单元        409
目录/正文匹配   409
```

---

## 12. 第九步：处理 PDF 图片

函数：

```python
_page_images()
```

调用：

```python
page.get_image_info(xrefs=True)
```

每张图片记录：

```text
image_id
xref
width
height
bbox
digest
extracted_path
```

默认命令：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
```

只记录图片元数据，不写出图片二进制，因此磁盘占用较小。

确实需要导出图片时：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates --extract-images
```

此时才调用：

```python
document.extract_image(xref)
```

当前识别到 1091 个有效图片对象。

需要注意：这份 PDF 中很多有效照片只有约 `191×191` 或 `156×141`，因此项目使用 128 像素阈值，而不是 256 像素。

---

## 13. 第十步：把页面拆成 RAG chunk

函数：

```python
_species_chunks()
```

一个页面会生成多个 chunk：

```text
taxonomy
distinguishing_features
colour
size
distribution
similar_species
references
pdf_image
```

每个 chunk 都带来源：

```json
{
  "doc_id": "doc_sa_invertebrates_p0401",
  "unit_id": "sa_taxon_corvol_p0401",
  "source_file": "...pdf",
  "page": 401,
  "printed_page": 398,
  "scientific_name": "Coronaster volsellatus",
  "chunk_type": "distinguishing_features",
  "modality": "text",
  "content": "..."
}
```

当前生成 3871 个 chunk。

输出文件：

```text
data/mrag/raganything/book_native/sa_invertebrates/rag_chunks.jsonl
```

---

## 14. 第十一步：生成基础关系三元组

函数：

```python
_relation_triples()
```

规则关系包括：

```text
taxon --is_a--> genus
taxon --is_a--> family
taxon --is_a--> order
taxon --is_a--> class
taxon --is_a--> phylum

taxon --has_common_name--> common name
taxon --has_feature--> distinguishing features
taxon --distributed_in--> distribution
taxon --similar_to--> similar species
taxon --described_in--> PDF page

image --illustrates--> taxon
image --belongs_to--> page unit
```

每条关系都保存：

```text
doc_id
page
unit_id
evidence
```

当前生成 6241 条确定性基础关系：

```text
data/mrag/raganything/book_native/sa_invertebrates/relation_triples.jsonl
```

这些关系是依据页面字段生成的，不是 LLM 自由猜测。

---

## 15. 第十二步：转换为 RAG-Anything content_list

函数：

```python
_raganything_content_items()
```

每个文本 chunk 会加入来源头：

```text
[DOC_ID=...]
[SOURCE=...]
[PAGE=...]
[UNIT_ID=...]
[CHUNK_TYPE=...]
Taxon: ...
正文内容
```

输出：

```text
raganything_content_list.jsonl
```

来源标记非常重要。后续 LightRAG 返回上下文时，`query_adapter.py` 可以重新解析出原始 `doc_id/source/page`。

---

## 16. 第十三步：把结构化内容写入 LightRAG

入口命令：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --limit-units 1
```

先用一个单元测试。

确认成功后：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --resume
```

调用顺序：

```text
raganything_cli.py
  -> indexer.py
  -> index_book_native_units()
       -> 读取 raganything_content_list.jsonl
       -> 按 unit_id 分组
       -> create_rag()
       -> ensure_initialized()
       -> rag.insert_content_list()
       -> rag.lightrag._insert_done()
       -> get_document_processing_status()
       -> 写 book_native_status.jsonl
```

`--resume` 会跳过已经 `fully_processed` 的页面单元。

这个阶段会调用文本模型抽取实体和关系，也会运行 BGE-M3 embedding，因此与前面的无模型结构解析不同。

---

## 17. RAG-Anything 和 LightRAG 如何初始化

文件：

```text
src/aquabio_raganything/runtime.py
```

核心函数：

```python
create_rag(paths, settings)
```

### 17.1 Embedding

类：

```python
LocalBGEEmbedding
```

模型：

```text
BAAI/bge-m3
embedding_dim = 1024
normalize_embeddings = True
```

### 17.2 文本模型

`llm_model_func()` 使用 OpenAI-compatible 调用方式，可接：

- DeepSeek。
- OpenRouter。
- 其他兼容 `/chat/completions` 的 API。

它主要用于：

- 实体抽取。
- 关系抽取。
- LightRAG 关键词解析。
- 图检索阶段的模型操作。

### 17.3 视觉模型

`vision_model_func()` 优先使用 Gemini 处理图片；未配置 Gemini 时回退文本模型函数。

### 17.4 持久化组件

```python
"kv_storage": "JsonKVStorage"
"vector_storage": "NanoVectorDBStorage"
"graph_storage": "NetworkXStorage"
"doc_status_storage": "JsonDocStatusStorage"
```

所以当前没有使用 Neo4j。

数据写入：

```text
data/mrag/raganything/working/
  graph_chunk_entity_relation.graphml
  kv_store_doc_status.json
  kv_store_full_docs.json
  kv_store_full_entities.json
  kv_store_full_relations.json
  kv_store_text_chunks.json
  vdb_chunks.json
  vdb_entities.json
  vdb_relationships.json
  kv_store_llm_response_cache.json
```

### 17.5 实体类型

在 `config.py` 中定义：

```text
species
taxon
anatomical_feature
habitat
behavior
distribution
conservation_status
image
table
equation
document
section
```

它们通过：

```python
addon_params["entity_types"]
addon_params["entity_types_guidance"]
```

传给 LightRAG。

---

# 第二部分：用户传入图文问题后的代码流程

## 18. 三类输入命令

### 18.1 纯文本

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "海星有哪些外观特征？" --session demo
```

### 18.2 纯图片

`--query` 可以留空：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --image "data\mrag\images\starfish\img_starfish_001.jpg" --session demo
```

### 18.3 图文同时输入

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "这是什么生物？请说明依据。" --image "data\mrag\images\starfish\img_starfish_001.jpg" --session demo
```

入口：

```text
mrag_cli.py
```

它创建：

```python
AquaBioMRAGWorkflow(paths, settings)
```

然后调用：

```python
workflow.invoke(query, image_path, session_id, hitl)
```

---

## 19. LangGraph State 是什么

文件：

```text
src/aquabio_mrag/models.py
```

核心类型：

```python
AquaBioState
```

它是整条 Agent 流程共享的数据容器，主要字段包括：

```text
original_query
resolved_query
image_path
image_caption
candidate_species
detected_species_ids
route
selected_tools
tool_plan
tool_observations
text_context
image_context
multimodal_context
pdf_context
graph_context
fused_context
final_context
draft_answer
final_answer
trace
warnings
```

每个 LangGraph node 接收当前 state，只返回自己修改或新增的字段。

---

## 20. LangGraph 主图建立顺序

文件：

```text
src/aquabio_mrag/workflow.py
```

函数：

```python
_build()
```

主图顺序：

```text
START
  -> session_init
  -> memory_load
  -> followup_resolver
       -> clarification/HITL
       -> router
  -> rewrite
  -> source_selection
  -> react_tool_plan
       -> vision
       -> retrieval_agent
  -> answer_agent
  -> evaluation
       -> retry
       -> finalize
  -> memory_save
  -> END
```

其中 `retrieval_agent` 和 `answer_agent` 又是两个子图。

---

## 21. 节点一：初始化会话

函数：

```python
session_init_node()
```

作用：

- 规范化 `session_id`。
- 初始化 trace。

接着：

```python
memory_load_node()
```

从：

```text
data/mrag/sessions/{session_id}.json
```

加载最近 8 轮对话及摘要。

---

## 22. 节点二：解析追问

函数：

```python
followup_resolver_node()
```

它判断用户是否在说：

```text
刚才那个
这个生物
它的颜色
上一个物种
```

如果当前 session 有上一轮物种，就把代词转换为明确物种。

如果没有上一轮信息并启用了 `--hitl`，进入真实人工中断。

---

## 23. HITL 如何工作

函数：

```python
clarification_node()
```

调用：

```python
interrupt({...})
```

在线状态保存到：

```text
data/mrag/sessions/langgraph.sqlite
```

查询等待状态：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py pending --session demo
```

继续：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py resume --session demo --answer "我指的是海星"
```

恢复函数：

```python
workflow.resume()
  -> Command(resume={"answer": answer})
```

这不是重新运行，而是从中断节点继续执行。

---

## 24. 节点三：Router 判断任务类型

函数：

```python
router_node()
```

任务类型：

```text
text_qa
followup_text_qa
image_qa
multimodal_qa
comparison_qa
source_trace
pdf_qa
```

示例：

```text
只有文字              -> text_qa
只有图片              -> image_qa
文字 + 图片           -> multimodal_qa
“海星和海胆区别”      -> comparison_qa
“来源在哪一页”        -> source_trace
明确提到 PDF/手册     -> pdf_qa
```

Router 同时决定是否需要：

```text
VLM
文本检索
图片检索
图文对检索
PDF 检索
```

---

## 25. 节点四：查询改写

函数：

```python
rewrite_node()
```

它不会替用户回答，而是为检索补充意图。

例如比较问题会补充：

```text
视觉特征
身体结构
相似物种
识别提示
权威 PDF 描述
```

改写后的内容写入：

```text
state["rewritten_query"]
```

---

## 26. 节点五：选择允许使用的工具

函数：

```python
source_selection_node()
```

工具集合：

```text
vlm_caption
text_retriever
image_retriever
multimodal_retriever
pdf_retriever
```

这一步先由 Router 确定工具白名单，避免模型调用不相关或危险工具。

---

## 27. 节点六：受控 ReAct 工具规划

文件：

```text
src/aquabio_mrag/react_agent.py
```

类：

```python
ControlledReActPlanner
```

当环境变量开启：

```cmd
set MRAG_REACT_NATIVE=true
```

`react_tool_plan_node()` 会调用：

```python
OpenRouterClient.select_tools()
```

使用原生 function calling，从 Router 允许的工具中选择本轮需要调用的工具。

限制：

- 只能选择白名单工具。
- 最多 4 步。
- 模型工具规划失败时回退到 Router 的确定性工具列表。
- 检索完全无结果时才重新规划，不会无条件循环。

---

## 28. 节点七：图片如何进入视觉模型

函数：

```python
vision_node()
```

视觉客户端：

```text
src/aquabio/openrouter.py
src/aquabio/gemini.py
```

选择逻辑：

```text
AQUABIO_VISION_PROVIDER=qwen  -> OpenRouter/OpenAI-compatible
Gemini 已配置                 -> GeminiVisionClient
否则                          -> OpenRouterClient
```

`OpenRouterClient.analyze_image()` 执行：

1. 检查图片是否存在。
2. 读取图片二进制。
3. Base64 编码。
4. 生成 `data:image/...;base64,...`。
5. 将文本 prompt 和 image_url 一起发给多模态模型。
6. 尝试解析 JSON。
7. JSON 不规范时把普通文本作为 caption。

期望输出：

```json
{
  "description": "...",
  "possible_species": [],
  "visible_features": [],
  "degradation": [],
  "confidence": 0.0,
  "limitations": []
}
```

然后 `vision_node()` 将候选名称映射为项目内部 `species_id`。

如果视觉 API 失败，但图片已存在于本地知识库，系统使用登记的 caption 和物种标签降级。

需要注意：当前 `vision_node()` 的 prompt 仍优先考虑原有 20 类本地图像体系；PDF 全书知识已经不限制 20 类，但上传图片的候选识别策略仍可继续扩展。

---

## 29. Retrieval Agent 的总体结构

文件：

```text
src/aquabio_mrag/retrieval_agent.py
```

类：

```python
RetrievalAgent
```

PDF 检索并不是只查一个向量库，而是：

```text
Chroma 语义检索
  +
book_native BM25 关键词检索
  +
LightRAG Hybrid 图检索
  -> Weighted RRF
```

---

## 30. Chroma 检索

文件调用：

```text
retrieval.py
  -> MultiSourceRetriever
  -> vector_db.py
  -> ChromaMRAGStore
  -> BGEEmbedder
```

`MultiSourceRetriever.search()` 对 Chroma 原始相似度再次加权：

```text
语义相似度
候选物种匹配
来源类型权重
chunk 类型权重
关键词重叠
```

可检索来源：

```text
species_card
species_text_chunk
image_doc
multimodal_pair
pdf_chunk
pdf_figure
```

---

## 31. BM25 检索

类：

```python
BookNativeBM25
```

它直接读取：

```text
data/mrag/raganything/book_native/*/rag_chunks.jsonl
```

BM25 对这些情况很有价值：

- 精确学名。
- FB Code。
- 英文形态术语。
- 页面中罕见但明确的词。
- 向量模型不容易区分的相似名称。

当前 BM25 是项目内轻量实现，没有引入额外搜索服务。

---

## 32. LightRAG 图检索

Retrieval Agent 不直接 import 图服务，而是通过 MCP 调用：

```text
raganything_hybrid_search
```

调用链：

```text
retrieval_agent.py
  -> mcp_client.py
  -> .venv-raganything Python 子进程
  -> aquabio_raganything.mcp_server
  -> query_adapter.hybrid_search()
  -> rag.aquery(mode="hybrid")
```

`hybrid_search()` 固定使用：

```python
rag.aquery(
    query,
    mode="hybrid",
    only_need_context=True,
    top_k=top_k,
    chunk_top_k=top_k,
    vlm_enhanced=False,
)
```

LightRAG 内部结合：

```text
实体关键词
关系图
实体/关系向量
相关文本 chunk
```

---

## 33. MCP Client 是如何真正工作的

文件：

```text
src/aquabio_mrag/mcp_client.py
```

核心对象：

```python
MCPStdioClient
```

真实调用步骤：

```text
StdioServerParameters
  -> stdio_client()
  -> ClientSession()
  -> session.initialize()
  -> session.list_tools() / session.call_tool()
```

项目配置两个 MCP Server：

### Chroma MCP

```text
Python: .venv\Scripts\python.exe
模块:   aquabio_mrag.mcp_server
```

工具：

```text
search_species_text
search_image_captions
search_multimodal
search_pdf
generate_image_caption
get_source_detail
```

### RAG-Anything MCP

```text
Python: .venv-raganything\Scripts\python.exe
模块:   aquabio_raganything.mcp_server
```

工具：

```text
raganything_index_status
raganything_graph_neighbors
raganything_hybrid_search
raganything_source_detail
```

查看工具：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

---

## 34. 三路证据如何融合

文件：

```text
src/aquabio_mrag/evidence.py
```

统一结构：

```python
EvidenceItem
```

字段包括：

```text
id
source
doc_id
page
source_file
modality
content
rank
score
entity_names
relation_path
metadata
```

融合函数：

```python
weighted_rrf()
```

当前权重：

```text
LightRAG graph  0.50
Chroma          0.35
BM25            0.15
```

公式：

```text
score += weight / (60 + rank)
```

去重键：

```text
doc_id + page + content_hash
```

因此同一证据同时被 Chroma 和图检索命中时，不会重复展示两次。

如果 RAG-Anything MCP 不可用：

```text
保留 Chroma + BM25
写入 warning
继续回答
```

---

## 35. Retrieval 子图

`workflow.py` 中：

```python
_build_retrieval_subgraph()
```

顺序：

```text
retrieve
  -> rerank
  -> build_context
```

### retrieve

函数：

```python
retrieval_node()
```

根据 `selected_tools` 执行文本、图片、图文和 PDF 检索。

### rerank

函数：

```python
rerank_node()
```

按 `final_score` 排序并截断。

### build_context

函数：

```python
context_node()
```

将证据转换为：

```text
[E1] ...
[E2] ...
[E3] ...
```

并保存来源名称和页码。

---

## 36. Answer 子图

`workflow.py` 中：

```python
_build_answer_subgraph()
```

顺序：

```text
generate
  -> guard
  -> evaluate
```

### generate

`answer_node()` 将以下内容发给文本模型：

```text
任务类型
原问题
解析后的问题
对话历史
图片 caption
视觉特征
候选物种
输出约束
最终证据
```

模型必须用 `[E数字]` 引用证据。

### guard

`response_guard_node()` 处理“只回答颜色”“简短回答”等硬约束。

### evaluate

`evaluation_node()` 检查：

- 是否有答案。
- 是否有上下文。
- 是否包含有效引用。
- 答案是否疑似被截断。
- 图片任务是否有视觉描述。

失败时可以重试：

```text
retrieval
rewrite
vision
answer
```

重试次数由：

```text
MRAG_MAX_RETRY
```

控制。

---

## 37. 最终答案和会话保存

函数：

```python
finalize_node()
```

生成最终答案。

随后：

```python
memory_save_node()
```

保存：

```text
用户问题
解析后问题
图片路径
答案
物种 ID
图片 caption
证据 ID
执行轨迹
警告
```

文件：

```text
data/mrag/sessions/{session_id}.json
```

查看：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py history --session demo
```

---

# 第三部分：从零理解整个项目的推荐顺序

## 38. 第一层：先读配置和数据模型

建议先读：

```text
src/aquabio_mrag/config.py
src/aquabio_raganything/config.py
src/aquabio_mrag/models.py
```

要理解：

- 数据放在哪里。
- 使用哪些环境变量。
- State 有哪些字段。
- 文档、图片和 PDF 注册记录长什么样。

---

## 39. 第二层：理解普通 Chroma 知识库

按顺序读：

```text
src/aquabio_mrag/data_pipeline.py
src/aquabio_mrag/pdf_pipeline.py
src/aquabio_mrag/vector_db.py
src/aquabio_mrag/retrieval.py
```

重点理解：

```text
原始数据
  -> RAGDocument
  -> embedding_text
  -> BGE-M3
  -> Chroma
  -> MultiSourceRetriever
```

这是项目最基础、最稳定的一条 RAG 路线。

---

## 40. 第三层：理解这本 PDF 的精确解析

只读：

```text
src/aquabio_raganything/book_native.py
```

按函数顺序：

```text
TaxaCatalogRow / SpeciesPageUnit
_text_spans
parse_sa_taxa_catalog
_extract_labeled_sections
_extract_taxonomy
_page_images
parse_sa_species_units
_enrich_units_from_catalog
_species_chunks
_relation_triples
_raganything_content_items
build_book_native
```

执行一次：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
```

然后亲自打开这些文件：

```text
book_taxa_catalog.jsonl
species_page_units.jsonl
rag_chunks.jsonl
relation_triples.jsonl
extraction_report.json
```

---

## 41. 第四层：理解 RAG-Anything 入库

按顺序读：

```text
src/aquabio_raganything/runtime.py
src/aquabio_raganything/indexer.py
src/aquabio_raganything/manifest.py
src/aquabio_raganything/storage_audit.py
```

重点追踪：

```text
create_rag
ensure_initialized
index_book_native_units
insert_content_list
_insert_done
get_document_processing_status
finalize_storages
```

---

## 42. 第五层：理解 LightRAG 查询

按顺序读：

```text
src/aquabio_raganything/query_adapter.py
src/aquabio_raganything/mcp_server.py
```

重点函数：

```text
hybrid_search
graph_neighbors
index_status
_parse_graph_context
_build_evidence
```

单独测试：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py query --query "海星有哪些识别特征？"
.\.venv-raganything\Scripts\python.exe raganything_cli.py neighbors --entity "Echinodermata"
```

---

## 43. 第六层：理解 MCP

按顺序读：

```text
src/aquabio_mrag/mcp_server.py
src/aquabio_raganything/mcp_server.py
src/aquabio_mrag/mcp_client.py
```

先理解 Server 暴露什么，再理解 Client 如何启动和调用 Server。

---

## 44. 第七层：理解 Retrieval Agent

按顺序读：

```text
src/aquabio_mrag/evidence.py
src/aquabio_mrag/retrieval_agent.py
```

重点是：

```text
统一 EvidenceItem
Chroma 标准化
BM25
MCP 图检索
Weighted RRF
降级策略
```

---

## 45. 第八层：最后读 LangGraph

最后读：

```text
src/aquabio_mrag/react_agent.py
src/aquabio_mrag/workflow.py
mrag_cli.py
```

如果一开始直接读 `workflow.py`，会同时看到会话、视觉、检索、生成、评价和重试，比较容易乱。先理解前七层，再回来读主图会清楚很多。

---

# 第四部分：建议的完整运行顺序

## 46. 检查基础环境

```cmd
cd /d F:\rag\AquaBio-AgentRAG
.\.venv\Scripts\python.exe --version
.\.venv-raganything\Scripts\python.exe --version
```

检查 Chroma：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py db-info
```

检查 MCP：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

---

## 47. 生成 PDF 结构化数据

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
```

检查报告：

```text
data/mrag/raganything/book_native/sa_invertebrates/extraction_report.json
```

---

## 48. 单页试运行 LightRAG 入库

该步骤会调用 API：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --limit-units 1
```

检查：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py status
```

---

## 49. 断点执行全书入库

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --resume
```

状态文件：

```text
data/mrag/raganything/manifests/book_native_status.jsonl
```

全书需要多次模型调用，建议低并发、保留缓存并使用 `--resume`。

---

## 50. 测试三类问答

纯文本：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "海星有哪些外观特征？" --session lesson
```

纯图片：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --image "data\mrag\images\starfish\img_starfish_001.jpg" --session lesson
```

图文：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "这是什么生物？请结合图片和PDF证据说明。" --image "data\mrag\images\starfish\img_starfish_001.jpg" --session lesson
```

---

## 51. 开启 ReAct

CMD：

```cmd
set MRAG_REACT_NATIVE=true
```

PowerShell：

```powershell
$env:MRAG_REACT_NATIVE="true"
```

然后正常执行 `mrag_cli.py ask`。

---

## 52. 开启或关闭图检索

开启：

```cmd
set MRAG_USE_GRAPH_MCP=true
```

临时关闭，只使用 Chroma + BM25：

```cmd
set MRAG_USE_GRAPH_MCP=false
```

关闭图检索适合：

- 调试普通问答。
- 图环境暂时不可用。
- 不希望等待 RAG-Anything 模型查询。

---

## 53. 运行测试

项目使用 `unittest`：

```cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖：

- 会话记忆。
- 路由。
- Chroma 数据完整性。
- 图上下文解析。
- PDF 表格恢复。
- RRF 去重。
- MCP 配置。
- 视觉失败降级。

---

# 第五部分：必须牢记的实现边界

## 54. 当前已经实现

```text
PyMuPDF 全书结构解析
409 条目录与 409 个鉴定页面匹配
字段级 chunk
PDF 图片对象元数据
确定性关系三元组
RAG-Anything content_list
LightRAG/NetworkX 入库入口
Chroma 检索
BM25 检索
MCP Client
LightRAG Hybrid MCP 查询
加权 RRF
LangGraph 主图和子图
受控 ReAct
SQLite HITL interrupt/resume
多轮会话 JSON
图文问答
API 失败降级
```

## 55. 当前仍需要执行的工作

“代码入口已经实现”和“全部数据已经完成模型入库”不是一回事。

当前 `book-native` 已经完成全书确定性结构化，但 409 个页面单元仍需要通过：

```cmd
index-book-native --resume
```

逐步送入 LightRAG，才能让 NetworkX 图覆盖整本书。

当前 `working/` 中已有图索引主要来自之前完成的真实 PDF 小段，而不是已经完整跑完 501 页。

因此查看状态时要同时关注：

```text
book_native/extraction_report.json
manifests/book_native_status.jsonl
raganything_cli.py status
```

## 56. 一句话总结整个调用链

```text
PDF 建库：
raganything_cli.py
-> book_native.py
-> JSONL
-> indexer.py
-> runtime.py
-> RAG-Anything
-> LightRAG
-> NanoVectorDB + NetworkX + JSON Storage

图文问答：
mrag_cli.py
-> workflow.py
-> Vision
-> ReAct
-> RetrievalAgent
-> Chroma + BM25 + MCP LightRAG
-> RRF
-> Answer Agent
-> Evaluation
-> Session Storage
```

理解这两条主线后，整个 `F:\rag\AquaBio-AgentRAG` 就不会再像一堆互相独立的 Python 文件，而是一个“离线知识构建 + 在线 Agent 问答”的完整系统。
