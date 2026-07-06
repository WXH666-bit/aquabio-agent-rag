# 系统架构

## 主流程

```text
用户问题 + 可选图片
  -> Router：判定 document_qa / image_analysis / multimodal_qa
  -> Query Builder：组合原问题、图片候选类别和退化描述
  -> Parallel Tools：
       PDF/卡片混合检索
       图像质量分析
       多候选图像增强
       OpenRouter VLM 结构化分析
       可选 Detector
  -> Context Builder：保留来源、页码、分数和工具事实
  -> Answer Generator：基于证据生成中文答案
  -> Guard：检查引用、低置信度声明和工具结果一致性
```

## 状态对象

```json
{
  "query": "",
  "route": "",
  "image_path": null,
  "retrieval": [],
  "image_quality": null,
  "enhancements": [],
  "vision_analysis": null,
  "detections": [],
  "tool_trace": [],
  "answer": "",
  "warnings": []
}
```

## 边界

1. OpenRouter 模型负责规划、图像语义理解和答案生成。
2. OpenCV 负责可重复的数值质量指标和增强，不让 LLM 编造图像处理结果。
3. 检索结果必须带 `source`、`page`、`source_type`。
4. VLM 的类别判断与专用检测器结果分开存储。
5. 增强图“看起来更好”不等于检测效果提高。只有接入检测器并比较相同测试集指标后，才能声称增强改善了检测。

## 后续替换点

- `HybridRetriever` -> BGE-M3 + Chroma/FAISS + RRF/reranker。
- `PDFIngestor` -> RAG-Anything/MinerU，用于表格、公式和图像抽取。
- `RuleRouter` -> LangGraph supervisor。
- `NullDetector` -> Ultralytics YOLO 自定义权重。
- 本地 JSONL -> SQLite/PostgreSQL + 对象存储。

