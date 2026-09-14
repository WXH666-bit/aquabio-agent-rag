# AquaBio-AgentRAG

面向水生生物识别、PDF 知识图谱与多模态研究的本地 Agent 工作台。

基于 **all-MiniLM-L6-v2 + Chroma** 向量检索、**LangGraph Agent** 工作流编排、**MCP 协议**工具调用，支持图文联合问答、多轮会话、ReAct 推理与 PDF 图谱检索。

---

## 快速启动

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[raganything]"
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

### StepFun 在线模型与 MCP 验收

已用 `step-3.7-flash` 实测模型原生工具调用、图像分析、两个 MCP 服务的
16 个工具，以及文本、多轮追问、PDF 和图文 LangGraph 问答。
结果见 [在线验证报告](docs/online-validation.json) 和 [验证说明](docs/在线模型与MCP验证.md)。
图谱验证使用现有 PDF 第 429 页的一条真实物种记录，当前生成 24 个实体、32 条关系；
这不代表整套 PDF 已完成图谱索引。

在 `.env` 中配置（密钥自行填写）：

```ini
AQUABIO_LLM_PROVIDER=stepfun
AQUABIO_VISION_PROVIDER=stepfun
RAGANYTHING_TEXT_LLM_PROVIDER=stepfun
STEPFUN_API_KEY=your_key
STEPFUN_MODEL=step-3.7-flash
STEPFUN_BASE_URL=https://api.stepfun.com/step_plan/v1
MRAG_EMBEDDING_BACKEND=onnx
AQUABIO_LLM_READ_TIMEOUT=180
RAGANYTHING_QUERY_TIMEOUT=180
RAGANYTHING_MCP_TIMEOUT=210
```

```powershell
.\.venv\Scripts\python.exe raganything_cli.py index-book-native --book sa_invertebrates --unit sa_taxon_lucafr_p0429 --resume
.\.venv\Scripts\python.exe scripts/verify_online_stack.py
```

第二条命令会产生真实模型调用费用，自动启动临时图谱 MCP 服务并清理测试会话。
模型凭据只从环境变量或忽略的 `.env` 读取，不写入报告。

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
python scripts/evaluate_retrieval.py --output docs/retrieval-baseline.json
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

## 索引维护与接口约束

旧索引缺少物种和来源元数据时，可运行 `python mrag_cli.py repair-metadata`。
该命令先验证原始文档与索引的 ID、正文完全一致，再恢复元数据；不会重新计算嵌入。
重建索引使用新集合，校验完成后通过 manifest 原子切换，保留旧集合供回滚。

`requirements.txt` 从 `pyproject.toml` 安装项目。安装后的 CLI 支持
`aquabio-mrag-cli --root 项目目录 db-info`；源码入口仍可直接使用。
会话 ID 必须为 1–80 位小写 ASCII 字母、数字、点、下划线或连字符，首位为字母或数字；
不再把不同输入静默转换为同一个 ID。Windows 保留文件名不允许作为会话 ID。
图片上传上限为 20 MiB，PDF 为 50 MiB，并检查文件内容。

后台任务最多同时执行 4 个，总排队与执行容量为 16 个；满额返回 HTTP 429。
取消会在当前外部调用结束后停止后续节点，尚未提交的会话记忆不会落盘。
SSE 返回 `node_start`、`node_progress`、`final`／`error` 事件。

详细改动和验证范围见 [逐项修复与验证记录](docs/逐项修复与验证记录.md)。

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
