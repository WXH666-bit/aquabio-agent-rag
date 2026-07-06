# AquaBio-MRAG Long Text Database

这是一个面向“水下生物多模态 RAG”的长文本种子知识库。

## 数据规模

- 物种数量：20 类
- 物种卡片：20 条
- 文本知识 chunk：160 条，每类 8 条
- 统一 RAG 文档：180 条

## 文件说明

```text
data/species_list.json
data/knowledge/species_cards.jsonl
data/knowledge/species_text_docs.jsonl
data/knowledge/rag_documents_combined_text_only.jsonl
data/knowledge/stats.json
```

## 重要说明

当前文本是 assistant_curated_long_seed_text，即“整理生成的长文本种子库”，不是对 Wikipedia/WoRMS 的逐字爬取内容。每条记录保留了 wikipedia 和 WoRMS 的 source_urls，方便后续运行脚本进行真实来源补充、验证或替换。

这样设计的原因是：
1. 先保证 RAG 项目可以直接跑通；
2. 避免直接复制大段外部网页内容；
3. 保持字段结构与后续爬虫结果一致；
4. 后续可以用 scripts/crawl_wikipedia_worms_extend.py 扩展真实摘要和分类学字段。

## 推荐使用方式

```bash
pip install -r requirements.txt
python scripts/build_chroma_index.py
python scripts/query_demo.py "海星有什么视觉特征？"
```

## 后续图片库

本压缩包只包含文本长库。图片库建议后续使用 Wikimedia Commons API 获取图片、作者、许可证和 source_page，然后用 VLM 生成 caption，再构建 image_docs.jsonl 和 multimodal_pairs.jsonl。
