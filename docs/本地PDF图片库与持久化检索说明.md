# 本地 PDF 图片库与持久化检索说明

## 1. start_chat_assistant.cmd 会不会重新建库？

不会。`start_chat_assistant.cmd` 的职责是启动服务，不负责重新解析 PDF、重新切 chunk 或重新抽取图片。

启动时主要做这些事：

1. 启动 FastAPI：`api_app:app`，端口 `8000`。
2. 启动 Streamlit 前端：`mrag_app.py`，端口 `8510`。
3. 可选启动 RAG-Anything MCP。
4. 加载已经持久化在磁盘上的 Chroma、PDF chunk、图片索引和会话数据库。

所以，如果 PDF 已经处理过，重启服务后不会丢失；如果你想重新生成 PDF 图片库或 chunk，需要手动运行 `raganything_cli.py` 的建库命令。

## 2. chunk 和索引是否已经保存在磁盘？

已经保存。当前项目不是内存临时库，主要持久化文件如下：

```text
data/mrag/raganything/book_native/sa_invertebrates/
  rag_chunks.jsonl
  species_page_units.jsonl
  book_taxa_catalog.jsonl

data/mrag/raganything/extracted_assets/sa_invertebrates/image_index/
  pdf_image_captions.jsonl
  linked_pdf_images.jsonl

data/mrag/raganything/extracted_assets/sa_invertebrates/images/
  sa_taxon_*.jpeg

data/mrag/raganything/working/
  graph_chunk_entity_relation.graphml
  kv_store_*.json
  vdb_*.json

data/mrag/vector_db/chroma/
  Chroma 向量库
```

其中：

- `rag_chunks.jsonl`：PDF 页面的文本 chunk。
- `species_page_units.jsonl`：按物种页组织的结构化单元。
- `book_taxa_catalog.jsonl`：PDF 中物种目录、页码、分类信息。
- `pdf_image_captions.jsonl`：PDF 图片与物种、页码、图片角色的绑定索引。
- `images/sa_taxon_*.jpeg`：从 PDF 中真实保存下来的图片文件。
- `working/`：LightRAG/图索引相关持久化文件。

## 3. PDF 里面的图片保存在哪里？

以这份 PDF 为例：

```text
data/mrag/pdfs/Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf
```

抽取后的图片保存在：

```text
data/mrag/raganything/extracted_assets/sa_invertebrates/images/
```

图片索引保存在：

```text
data/mrag/raganything/extracted_assets/sa_invertebrates/image_index/pdf_image_captions.jsonl
```

例如 `Anseropoda grandis (AnsGra)` 的本地 PDF 图片：

```text
实体图：
data/mrag/raganything/extracted_assets/sa_invertebrates/images/sa_taxon_ansgra_p0406_img_01.jpeg

分布图：
data/mrag/raganything/extracted_assets/sa_invertebrates/images/sa_taxon_ansgra_p0406_img_02.jpeg
```

对应 PDF 页码是 `406`，印刷页码是 `403`。英文俗名是 `Pancake/Goosefoot star`。

## 4. 用户端如何检索出 PDF 图片？

前端提问后，调用链是：

```text
mrag_app.py
  -> POST /api/chat/tasks
  -> src/aquabio_web/service.py
  -> AquaBioMRAGWorkflow.invoke()
  -> src/aquabio_mrag/workflow.py
  -> RetrievalAgent.search_pdf_images()
  -> src/aquabio_raganything/image_rag.py
  -> pdf_image_captions.jsonl / Chroma 图片向量索引
  -> 返回 image_url 给前端显示
```

当前修正后的图片检索优先级是：

```text
1. 本地 PDF 图片实体索引 pdf_image_captions.jsonl
2. 本地 PDF 图片 caption 向量库
3. RAG-Anything MCP 图/图片工具
4. 本地仍然没有对应角色图片时，才尝试网络图片
```

也就是说，用户问“本地 PDF 图片库里的分布图”时，系统现在会优先返回 `raganything_pdf_image` 来源的图片，不会先去网上找。

## 5. 为什么之前明明 PDF 有分布图却没有找到？

之前主要有三个问题：

1. 前端默认开启 MCP 时，图片检索先走 MCP；MCP 超时、旧进程或旧索引返回空时，本地 JSONL 精确图片索引没有被强制优先使用。
2. 服务层发现“缺少分布图”后会触发网络兜底，因此可能返回 Wikimedia 图片，而不是本地 PDF 图片。
3. 图片返回对象没有透传 `printed_page`，导致即使命中本地 PDF 图，也不容易看出它对应 PDF 的印刷页。

现在已经改为：

```text
本地 PDF 图片实体索引优先
  -> 命中 distribution_map 后禁止网络图冒充
  -> 前端返回 image_role/source/page/printed_page
```

## 6. 海星这种泛称如何回答？

如果用户问：

```text
给我海星分布图
```

“海星”不是一个具体物种，而是上位类群。系统会从本地 PDF 图片库中选择一个具体海星物种的分布图，并在回答中说明具体物种，例如：

```text
Coronaster volsellatus
False brisingid/Spiny pom-pom starfish
PDF page 401, printed page 398
```

这张图只代表该具体物种，不代表所有海星的全球分布。

如果用户问：

```text
Anseropoda grandis (AnsGra) 的中文名是什么，而且分布图给我要库里保存的
```

系统会返回本地 PDF 分布图：

```text
sa_taxon_ansgra_p0406_img_02.jpeg
```

并说明：

- 本地 PDF 记录的英文俗名是 `Pancake/Goosefoot star`。
- 当前本地证据没有权威中文名。
- 可以给出非正式直译，如“薄饼/鹅足海星”或“煎饼/鹅掌海星”，但必须标明不是权威中文名。

## 7. 常用维护命令

重新生成物种页文本 chunk：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py book-native --book sa_invertebrates
```

重新抽取 PDF 图片并绑定物种：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py image-assets --book sa_invertebrates --overwrite
```

重建 PDF 图片 caption 向量库：

```cmd
.\.venv-raganything\Scripts\python.exe raganything_cli.py index-images --book sa_invertebrates --reset
```

直接命令行检索本地 PDF 图片：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Anseropoda grandis AnsGra 分布图"
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "海星 分布图"
```

启动前后端：

```cmd
start_chat_assistant.cmd
```

停止前后端：

```cmd
stop_chat_assistant.cmd
```
