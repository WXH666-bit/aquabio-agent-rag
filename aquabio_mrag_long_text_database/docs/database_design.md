# 文本知识库设计说明

## 1. 为什么不只保存 content？

多模态 RAG 需要在回答时知道一条知识来自哪个物种、属于什么 chunk 类型、可以解决什么问题，所以每条记录都保存：

- id
- source_type
- species_id
- english_name
- chinese_name
- scientific_name
- chunk_type
- title
- content
- keywords
- source_urls
- provenance

## 2. species_cards.jsonl

每个物种 1 条，适合回答：
- 这个物种是什么？
- 它的整体特征是什么？
- 它属于什么分类？
- 它与哪些生物容易混淆？

## 3. species_text_docs.jsonl

每个物种 8 条 chunk：
- overview
- taxonomy
- visual_features
- habitat
- ecology_behavior
- similar_species
- image_recognition_tips
- rag_usage

## 4. rag_documents_combined_text_only.jsonl

这是直接写入向量库的统一文档格式。每条记录包含 content 和 embedding_text。

构建 Chroma / FAISS 时，用 embedding_text 生成向量，用 metadata 过滤 source_type、species_id、chunk_type。
