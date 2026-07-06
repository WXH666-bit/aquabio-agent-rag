# Embedding 与向量库建议

## 1. 文本 RAG

如果当前阶段只做文本问答或 caption-based 图文 RAG，优先使用文本 embedding：

- 中文为主：BAAI/bge-small-zh-v1.5
- 中英混合：BAAI/bge-m3
- 轻量测试：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

## 2. 图文 RAG

如果要支持“以图搜文 / 以文搜图 / 以图搜图”，可以使用 CLIP 或 OpenCLIP。

CLIP 与普通文本 embedding 的区别：
- 普通文本 embedding：只能把文本变成向量，适合文本查文本；
- CLIP：有图像编码器和文本编码器，把图片和文本映射到同一语义空间，适合图文互检；
- BERT + ViT：通常只是两个独立编码器，除非经过图文对比学习或跨模态对齐，否则文本向量和图片向量不能直接比较；
- CLIP 常见图像端可以是 ResNet 或 ViT，文本端是 Transformer，不等于“文本用 BERT、图片用 ViT”这么简单。

## 3. 推荐路线

阶段一：
文本 + 图片 caption 全部转成文本，用 bge-m3 或 bge-small-zh 建 Chroma。

阶段二：
加入 CLIP / OpenCLIP，为图片建立 image embedding，实现以图搜图、以图搜文。

阶段三：
文本检索结果和 CLIP 检索结果做融合 rerank。
