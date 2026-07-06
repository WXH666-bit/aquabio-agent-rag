# 南非海洋无脊椎动物 PDF 逐步处理流程

本文只解释以下文件在当前项目中如何被处理：

```text
data/mrag/pdfs/
Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf
```

## 1. 已确认的原始结构

- PDF 共 501 个物理页。
- PDF 第 4 个物理页是目录，书内页码从正文开始编号。
- 书内页码与 PDF 物理页通常相差 3 页。例如书内第 41 页是 PDF 第 44 页。
- PDF 第 26 个物理页是 `Table of Taxa` 标题页。
- PDF 第 27 至 39 个物理页是分类单元总表。
- 分类总表恢复出 409 行目录记录。
- 后续正文中有 408 个标准或特殊鉴定页面单元。
- 标准鉴定页通常一页一个分类单元，固定包含形态、颜色、尺寸、分布、相似种、参考文献和分类字段。
- 书内第 382 页是特殊双物种页，同时描述两种 `Ornithoteuthis`。

这里必须区分：

```text
PDF physical page = 文件中的第几个页面
printed page      = 页面上印刷的书内页码
```

所有输出同时保存二者，检索结果引用 `pdf_page`，并保留 `printed_page` 供人工核对。

## 2. 当前可运行命令

在 `F:\rag\AquaBio-AgentRAG` 下执行：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
```

默认只记录图片对象元数据，不把 1000 多张图片再次写入磁盘。

确实需要导出图片时：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates --extract-images
```

输出目录：

```text
data/mrag/raganything/book_native/sa_invertebrates/
  book_sections.jsonl
  book_taxa_catalog.jsonl
  species_page_units.jsonl
  rag_chunks.jsonl
  relation_triples.jsonl
  extraction_report.json
  raganything_content_list.jsonl
  images/                         # 仅使用 --extract-images 时产生
```

## 3. 一步一步的文件调用流程

```text
raganything_cli.py
  -> aquabio_raganything.config.RAGAnythingPaths
  -> aquabio_raganything.book_native.build_book_native()
       -> PyMuPDF 打开 501 页 PDF
       -> parse_sa_taxa_catalog()
       -> parse_sa_species_units()
       -> _enrich_units_from_catalog()
       -> _join_catalog_and_units()
       -> _species_chunks()
       -> _relation_triples()
       -> 写入 JSONL 和 extraction_report.json
```

这一阶段是确定性版面解析，不调用 LLM，也不生成 embedding。这样做能避免模型把学名、页码和字段边界猜错。

## 4. 分类总表如何恢复

分类表不是简单的纯文本行。代码读取每个文字 span 的坐标：

```text
Class | Order | Family | Genus | Species | Common name | Authority | FB Code | Page
```

处理步骤：

1. 在表格最右列寻找 41 至 493 的页码锚点。
2. 页脚数字小于 41，因此被排除。
3. 用相邻两个页码锚点的纵坐标中点确定一行的上下边界。
4. 用固定横坐标区间恢复九列内容。
5. 合并同一单元格内因换行产生的多个 span。
6. 根据 `sp.`、`spp.` 和缺失属名判断记录是 species、group 还是 higher taxon。
7. 计算预计 PDF 页：`expected_pdf_page = printed_page + 3`。

该步骤当前恢复 409 条目录记录。目录表是正文定位索引，不直接代替正文证据。

将结构化页面单元送入 RAG-Anything/LightRAG：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --limit-units 1
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --resume
```

第一条用于单页验收，第二条按 `book_native_status.jsonl` 断点续跑全书。该过程会调用配置的文本模型抽取实体关系，因此会产生 API 调用，不能与不调用模型的 `book-native` 结构解析混为一谈。

## 5. 每个鉴定页如何解析

代码首先检查页面是否包含：

```text
distinguishing features
Phylum:
```

随后按字段标题切分：

```text
distinguishing_features
colour
size
distribution
similar_species
references
```

页面后半部分再解析：

```text
Phylum
Subphylum
Class
Subclass
Order
Suborder
Infraorder
Family
Genus
Species
Common name
FB Code
```

目录表和正文页会按照书内页码、FB Code 双重匹配。正文缺少标题或分类层级时，使用目录表补齐；正文中的详细形态与分布始终优先，不用目录表短文本覆盖。

源书中的占位模板页含有 `xxx`，代码明确排除，不把它当成真实物种。

双鸟鱿页面不是标准模板。代码将它保存为共享页面单元，明确记录：

```text
Ornithoteuthis antillarum
Ornithoteuthis volatilis
Atlantic bird squid
Shiny bird squid
同页分布信息
雄性 hectocotylus 的列数和凹点数量差异
```

## 6. PDF 图片究竟做了什么

PyMuPDF 使用 `page.get_image_info(xrefs=True)` 读取：

- xref
- 原始宽高
- 页面 bbox
- 图像 digest
- 所属鉴定页和分类单元

这份压缩 PDF 的主体照片常见尺寸约为 191×191 或 156×141，因此不能机械使用 256 px 阈值，否则会把真实物种照片全部删掉。当前元数据阈值为 128 px。

默认不导出二进制图片，只生成 1089 条有效图片对象记录。加 `--extract-images` 后才调用 `document.extract_image(xref)` 写入磁盘。

图片与物种生成：

```text
image_id --illustrates--> taxon
image_id --belongs_to--> species_page_unit
```

当前没有对这些图片做 CLIP embedding。后续多模态描述由 RAG-Anything 的 modal processor 或视觉模型生成。

## 7. 如何成为 RAG chunk

每个鉴定页不是只生成一个大 chunk，而是按字段拆分：

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

每个 chunk 都带：

```json
{
  "doc_id": "doc_sa_invertebrates_p0401",
  "unit_id": "sa_taxon_corvol_p0401",
  "source_file": "...pdf",
  "page": 401,
  "printed_page": 398,
  "scientific_name": "Coronaster volsellatus",
  "fb_code": "CorVol",
  "chunk_type": "distinguishing_features",
  "modality": "text"
}
```

这样用户问“颜色”“尺寸”“如何区分”时，可以命中对应字段，而不是被整页参考文献稀释。

## 8. 实体关系如何生成

确定性阶段先生成有来源的基础关系：

```text
taxon --is_a--> genus/family/order/class/phylum
taxon --has_common_name--> common name
taxon --has_feature--> distinguishing features
taxon --distributed_in--> distribution
taxon --similar_to--> similar species
taxon --described_in--> PDF page
image --illustrates--> taxon
image --belongs_to--> page unit
```

每条关系保存 `doc_id/page/unit_id/evidence`。之后将 `rag_chunks.jsonl` 或 RAG-Anything `content_list` 送入 `insert_content_list()`，LightRAG 会进一步抽取实体和关系并写入：

```text
文档块存储
实体向量存储
关系向量存储
NetworkX GraphML
文档状态存储
模型响应缓存
```

当前图存储是 NetworkX，不是 Neo4j。原项目设计也没有实际使用 Neo4j；Neo4j 只是未来数据量和并发查询明显增加时的可选迁移目标。

## 9. 查询时的完整链路

```text
用户问题
  -> LangGraph Router
  -> Controlled ReAct Planner
  -> Retrieval Agent
       -> Chroma BGE-M3 语义检索
       -> book_native BM25 字段检索
       -> MCP 调用 RAG-Anything LightRAG hybrid
            -> 图实体/关系检索
            -> 图相关 chunk 向量检索
  -> Weighted RRF
       graph 0.50
       chroma 0.35
       bm25 0.15
  -> doc_id + page + content_hash 去重
  -> 前 12 条证据
  -> Answer Agent 使用 [E1] 引用
```

MCP 不可用时不会破坏查询，Retrieval Agent 会记录警告并退化为 Chroma + BM25。

## 10. 当前实测结果

```text
PDF pages                  501
Table of Taxa rows         409
Identification page units 409
Catalog/page matches       409
RAG chunks                 约 3900
Image metadata records     约 1090
Relation triples           约 6200
```

最终数字以每次生成的 `extraction_report.json` 为准。

## 11. LangGraph、MCP、ReAct 和 HITL

- LangGraph：已真实使用 `StateGraph`、条件边、检索子图、回答子图和 checkpoint。
- ReAct：支持 OpenAI-compatible 原生 function calling；设置 `MRAG_REACT_NATIVE=true` 后由模型在允许工具集合内选择工具。无证据时最多重新规划 4 步。
- MCP Client：真实使用 MCP SDK 的 `stdio_client + ClientSession + initialize + list_tools + call_tool`。
- HITL：使用 `langgraph.types.interrupt` 和 `Command(resume=...)`，在线模式的状态写入 `data/mrag/sessions/langgraph.sqlite`。
- 离线测试使用 MemorySaver，避免测试临时目录被 SQLite 文件句柄占用。

验证 MCP 工具：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py mcp-tools --server all
```

启用人工中断：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --session demo --hitl --query "这个生物是什么？"
.\.venv\Scripts\python.exe mrag_cli.py pending --session demo
.\.venv\Scripts\python.exe mrag_cli.py resume --session demo --answer "我指的是海星"
```
