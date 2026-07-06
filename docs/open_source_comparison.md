# 三个参考项目的区别

| 项目 | 本质定位 | 强项 | 不适合直接承担的部分 | 本项目借鉴 |
|---|---|---|---|---|
| RAG-Anything | 可安装的多模态文档 RAG 框架 | PDF/Office/图片解析，图表、表格、公式处理，LightRAG 图谱与混合查询 | 依赖 MinerU/LightRAG，原型阶段较重；不提供水下检测与增强业务逻辑 | 多模态内容单元、页码和模态元数据、图文统一上下文 |
| all-in-rag | 系统化 RAG 教程与案例集合 | 分块、Embedding、向量库、混合检索、RRF、重排和评估讲解完整 | 不是开箱即用的统一框架；示例横跨多套技术栈 | 轻量混合检索、可替换 Embedding、引用和评估思路 |
| agent-craft | Agent 全栈教学项目 | Function Calling、RAG 工具化、LangGraph 状态流、MCP、Streamlit | 教学模块为主，不含领域数据模型和多模态文档解析内核 | 显式状态、任务路由、工具轨迹、失败回退和界面 |

## 选择结论

不建议把三个仓库机械合并。最稳妥的方式是按职责吸收：

1. 文档层采用 RAG-Anything 的多模态知识单元设计；MVP 先用 PyMuPDF，复杂 PDF 再切换 MinerU/RAG-Anything。
2. 检索层采用 all-in-rag 的“稠密/稀疏融合 + 元数据 + 引用”思路；MVP 用字符 TF-IDF 与词法得分，后续替换 BGE-M3 + Chroma/FAISS。
3. 编排层采用 agent-craft 的显式状态图；MVP 不强制安装 LangGraph，节点接口稳定后可平滑迁移。

## 许可证提醒

- RAG-Anything：MIT。
- agent-craft：以仓库 LICENSE 为准，当前本地副本为 MIT。
- all-in-rag：CC BY-NC-SA 4.0，教程内容不能直接当作无约束商业代码资产。

参考：

- https://github.com/HKUDS/RAG-Anything
- https://github.com/datawhalechina/all-in-rag
- https://github.com/Annyfee/agent-craft

