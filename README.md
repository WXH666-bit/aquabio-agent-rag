# AquaBio-AgentRAG

面向水生生物识别、PDF 知识图谱与多模态研究的本地 Agent 工作台。

基于 **all-MiniLM-L6-v2 + Chroma** 向量检索、**LangGraph Agent** 工作流编排、**MCP 协议**工具调用，支持图文联合问答、多轮会话、ReAct 推理与 PDF 图谱检索。

---

## 快速启动

```bash
pip install -r requirements.txt
python run_app.py
```

启动后访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| 聊天界面 | http://127.0.0.1:3000 | Streamlit 前端，用户交互 |
| API 文档 | http://127.0.0.1:8000/docs | FastAPI Swagger，接口调试 |
| MCP 端点 | http://127.0.0.1:8765/mcp | RAG-Anything 图谱检索 |

停止：按 `Ctrl+C` 即可停止所有服务。

---

## 项目架构

```
├── src/
│   ├── aquabio/               # 核心：配置、LLM 客户端、图像处理、向量存储
│   ├── aquabio_mrag/           # RAG 引擎：LangGraph 工作流、Chroma 检索、ReAct
│   ├── aquabio_raganything/    # 图谱：MinerU PDF 解析、LightRAG、实体关系
│   └── aquabio_web/            # Web 层：FastAPI 路由、Streamlit 前端逻辑、会话存储
├── scripts/                    # 数据管道/验证脚本
├── tests/                      # 单元与集成测试
├── data/                       # 向量库、图像、PDF、日志、会话数据
├── configs/                    # 格式定义与 Schema
├── docs/                       # 详细技术文档
├── run_app.py                  # 一键启动
├── requirements.txt            # pip 依赖清单
└── pyproject.toml              # 项目依赖与构建配置
```

### 四层架构

| 层级 | 模块 | 组件 |
|------|------|------|
| 前端 | Streamlit | 聊天 UI、会话管理、附件上传、ReAct 面板 |
| API | FastAPI | 异步任务、会话 CRUD、反馈收集、MCP 工具代理 |
| Agent | LangGraph | StateGraph 工作流、ReAct 规划器、路由决策、Human-in-the-Loop |
| 检索 | Chroma + all-MiniLM-L6-v2 | 多源检索、BM25、LightRAG 图谱、PDF 实体索引 |

---

## 环境配置

复制 `.env.example` 为 `.env`，填入 API Key：

```ini
# 主 LLM（阿里通义千问）
AQUABIO_LLM_PROVIDER=qwen
QWEN_API_KEY=your_qwen_key
QWEN_MODEL=qwen3.6-flash

# 视觉模型（阿里通义千问）
AQUABIO_VISION_PROVIDER=qwen

# 嵌入模型（本地运行，首次自动下载约 120MB）
MRAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> `.env` 已在 `.gitignore` 中排除，不会被提交到版本控制。

---

## CLI 命令

### 向量库信息

```cmd
python mrag_cli.py db-info
```

### 文本问答

```cmd
python mrag_cli.py ask --query "海星有哪些外观特征？" --offline
```

### 图文联合问答

```cmd
python mrag_cli.py ask --query "这张图可能是什么生物？" --image "data\mrag\images\starfish\img_starfish_001.jpg"
```

### 多轮会话

```cmd
python mrag_cli.py ask --session starfish_demo --query "这个生物的外貌是什么样子的？" --image "data\mrag\images\starfish\img_starfish_001.jpg"
python mrag_cli.py ask --session starfish_demo --query "刚才问到的是什么生物？只回答常见颜色"
python mrag_cli.py history --session starfish_demo
```

### RAG-Anything PDF 图谱

```cmd
python raganything_cli.py inspect --segment sa_invertebrates_p0429_0432
python raganything_cli.py audit-storage
python raganything_cli.py status
python raganything_cli.py query --mode hybrid --query "Luidia africana 有哪些可识别的外观特征？"
```

### 启动独立 MCP 服务

```cmd
python -m aquabio_mrag.mcp_server
```

---

## 运行模式

聊天界面侧边栏提供 5 种预设模式：

| 模式 | 记忆 | RAG | 视觉 | PDF | MCP | 适用场景 |
|------|------|-----|------|-----|-----|----------|
| 标准 Agent | ✅ | ✅ | ✅ | ✅ | ✅ | 日常问答 |
| 全 MCP 模式 | ✅ | ✅ | ✅ | ✅ | ✅ | 完整调用链展示 |
| 快速文本 RAG | ✅ | ✅ | ❌ | ❌ | ❌ | 纯文本快速检索 |
| 图谱研究 | ✅ | ✅ | ❌ | ✅ | ✅ | PDF 知识图谱 |
| 图像识别 | ✅ | ✅ | ✅ | ✅ | ✅ | 图片识别为主 |

---

## 运行测试

```cmd
python -m unittest discover -s tests -v
```

---

## 当前数据规模

| 数据类型 | 数量 |
|----------|------|
| 物种卡片 | 20 |
| 物种长文本块 | 160 |
| 图片 caption | 200 |
| 图片-文本配对 | 200 |
| PDF 文本块 | 192 |
| Chroma 向量 | 772 |
| 图谱实体 / 关系 | 192 / 542 |

---

## 依赖项

- **Python** ≥ 3.10
- **嵌入模型**：all-MiniLM-L6-v2（首次运行自动下载约 120MB）
- **LLM API**：阿里通义千问（默认 qwen3.6-flash）
- **视觉 API**：阿里通义千问（图像识别）
- **PDF 解析**：PyMuPDF + MinerU

完整依赖见 [requirements.txt](requirements.txt) 和 [pyproject.toml](pyproject.toml)。

---

## 详细文档

- [启动命令与模块调用手册](docs/启动命令与模块调用手册.md)
- [本地前后端聊天助手运行说明](docs/本地前后端聊天助手运行说明.md)
- [当前版本运行与调用流程](docs/当前版本运行与调用流程.md)
- [完整流程与向量库](docs/完整流程与向量库.md)
- [项目从 PDF 建库到图文问答逐步教学](docs/项目从PDF建库到图文问答逐步教学.md)
- [项目全流程技术架构与实现审计](docs/项目全流程技术架构与实现审计.md)
- [LangGraph 多轮会话与追问解析](docs/LangGraph多轮会话与追问解析.md)
- [下一步-LangGraph 场景路由与 MCP 工具调用设计](docs/下一步-LangGraph场景路由与MCP工具调用设计.md)
- [产品化界面与 Agent 工具流程](docs/产品化界面与Agent工具流程.md)
- [SA 海洋无脊椎动物图鉴 PDF 逐步处理详解](docs/SA海洋无脊椎动物图鉴PDF逐步处理详解.md)
- [PDF 图片实体绑定与向量检索实现](docs/PDF图片实体绑定与向量检索实现.md)
- [验收与稳定性测试记录](docs/验收与稳定性测试记录.md)

---

## 安全

`.env.example` 只包含占位 key。任何已经贴到聊天或公开文件中的 API Key 都应立即吊销并重新创建。
