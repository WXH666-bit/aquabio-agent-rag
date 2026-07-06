# PDF 图片实体绑定与向量检索实现

## 1. 当前已经实现的结果

目标 PDF：

```text
data/mrag/pdfs/
Field-Guide-to-SA-Offshore-Marine-Invertebrates_web-full-version_compressed.pdf
```

当前实测结果：

```text
物种页面单元：409
真实导出图片：1091
唯一图片二进制：1090
图片与实体绑定：1091
图片关系三元组：3273
图片向量：1091
导出失败：0
```

这条流程不要求 vLLM，也不依赖 CLIP。图片文件保存在文件系统，BGE-M3
只对图片关联的实体、分类、页面文字和识别特征进行向量化。

## 2. 完整调用顺序

```text
raganything_cli.py image-assets
  -> aquabio_raganything.image_rag.build_pdf_image_assets()
  -> 读取 species_page_units.jsonl
  -> PyMuPDF document.extract_image(xref)
  -> 保存真实 JPEG/PNG
  -> 图片绑定 taxon/entity/page/unit
  -> 生成上下文 caption
  -> 生成图片关系三元组

raganything_cli.py index-images
  -> PDFImageVectorStore.build()
  -> BAAI/bge-m3
  -> Chroma collection: aquabio_pdf_images

mrag_cli.py ask
  -> LangGraph Router
  -> SourceSelectionNode
  -> pdf_image_retriever
  -> Chroma MCP: search_pdf_entity_images
  -> image_context
  -> ContextBuilder
  -> AnswerNode
  -> CLI 输出图片绝对路径
```

## 3. 图片保存位置

```text
data/mrag/raganything/extracted_assets/sa_invertebrates/
  images/
    sa_taxon_jaslal_p0149_img_03.jpeg
    sa_taxon_jaslal_p0149_img_04.jpeg
    ...
  image_index/
    image_objects.jsonl
    linked_pdf_images.jsonl
    pdf_image_captions.jsonl
    pdf_image_rag_docs.jsonl
    image_relation_triples.jsonl
    image_extraction_failures.jsonl
    image_pipeline_report.json
    image_vector_manifest.json
```

图片二进制不存入 Chroma。Chroma 保存：

```text
caption embedding
image_path
image_id
entity_id
scientific_name
common_name
doc_id
PDF page
taxonomy
```

## 4. 图片如何绑定实体

每个 `species_page_unit` 已经具有确定的物种实体和 PDF 页码。图片通过
同页关系绑定：

```text
PDF page 149
  -> species_page_unit: sa_taxon_jaslal_p0149
  -> taxon: Jasus lalandii
  -> image: sa_taxon_jaslal_p0149_img_03
```

生成三类关系：

```text
taxon:Jasus_lalandii
  --depicted_by-->
image:sa_taxon_jaslal_p0149_img_03

image:sa_taxon_jaslal_p0149_img_03
  --depicts-->
taxon:Jasus_lalandii

image:sa_taxon_jaslal_p0149_img_03
  --located_on_page-->
page:149
```

图片绑定记录保存在 `linked_pdf_images.jsonl`，不会依靠 LLM 猜测物种。

## 5. 为什么暂时不调用 VLM

当前 caption 使用该图片所属物种页的确定性上下文生成：

```text
科学名 + 常见名 + 分类信息 + distinguishing features
+ colour + PDF 页码 + 图片尺寸
```

优点：

```text
不产生 API 费用
不会因免费 VLM 限流而中断建库
不会把图片绑定到错误物种
可以直接用 BGE-M3 支持中英文文本查图
```

它不是像素级视觉描述。以后可以增加 VLM caption，但必须作为补充字段，
不能覆盖当前的确定性实体绑定。

## 6. 运行命令

第一次导出图片：

```cmd
cd /d F:\rag\AquaBio-AgentRAG
.\.venv\Scripts\python.exe raganything_cli.py image-assets --book sa_invertebrates --min-image-dimension 128
```

建立或断点续建图片向量库：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py index-images --book sa_invertebrates --batch-size 32
```

默认会跳过已经存在的向量。只有需要清空重建时才使用：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py index-images --book sa_invertebrates --batch-size 32 --reset
```

直接查询图片：

```cmd
.\.venv\Scripts\python.exe raganything_cli.py image-query --query "Jasus lalandii 的图片" --top-k 5

.\.venv\Scripts\python.exe raganything_cli.py image-query --query "海星的样例图片" --top-k 5
```

通过完整 LangGraph 问答：

```cmd
.\.venv\Scripts\python.exe mrag_cli.py ask --query "给我 Jasus lalandii 的样例图片，并说明识别特征" --session image-demo
```

运行可直接显示图片的页面：

```cmd
.\.venv\Scripts\python.exe -m streamlit run mrag_app.py
```

## 7. 中英文实体对齐

图片文档保存科学名、常见名、属、科、目、纲和门作为实体别名。

中文大类查询还会扩展为分类实体：

```text
海星 -> starfish, sea star, Asteroidea
海胆 -> sea urchin, Echinoidea
海参 -> sea cucumber, Holothuroidea
龙虾 -> lobster, Decapoda, Palinuridae
海绵 -> sponge, Porifera
```

如果查询中存在明确实体，例如 `Jasus lalandii`，最终结果会优先限制到该
实体，不再把近缘物种混入样例图片列表。

## 8. MCP 职责

图片向量查询属于 Chroma，因此工具部署在主环境：

```text
Chroma MCP:
  search_pdf_entity_images
  get_entity_sample_images
```

RAG-Anything MCP 仍提供：

```text
raganything_graph_neighbors
raganything_hybrid_search
raganything_entity_images
```

`raganything_entity_images` 是不加载向量模型的实体登记表查询；
`search_pdf_entity_images` 是 BGE-M3 + Chroma 语义图片查询。

## 9. 用户提问时的执行过程

问题：

```text
给我 Jasus lalandii 的样例图片，并说明识别特征
```

执行：

```text
Router -> text_qa + need_image_retrieval
SourceSelection -> pdf_image_retriever
ReAct -> 必须保留 pdf_image_retriever
Chroma MCP -> aquabio_pdf_images
实体精确对齐 -> Jasus lalandii
返回 image_context
AnswerNode -> 使用图片 caption 与 PDF 特征回答
CLI -> 输出真实绝对路径
Streamlit -> st.image() 显示图片
```

已验证的图片：

```text
PDF page 149
sa_taxon_jaslal_p0149_img_03.jpeg
sa_taxon_jaslal_p0149_img_04.jpeg
```
