import subprocess
import sys
import time
import urllib.request
import urllib.error
import json
import os
import importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

API_PORT = 8000
UI_PORT = 3000
API_URL = f"http://127.0.0.1:{API_PORT}"

procs = []

REQUIRED_PACKAGES = {
    "fitz": "PyMuPDF>=1.24",
    "numpy": "numpy>=1.26,<2",
    "cv2": "opencv-python>=4.9",
    "PIL": "Pillow>=10.0",
    "scipy": "scipy>=1.11",
    "sklearn": "scikit-learn>=1.4",
    "joblib": "joblib>=1.3",
    "requests": "requests>=2.31",
    "httpx": "httpx>=0.27,<1",
    "sentence_transformers": "sentence-transformers>=3.4,<6",
    "chromadb": "chromadb>=1.0,<2",
    "langgraph": "langgraph>=1.0,<2",
    "langgraph.checkpoint.sqlite": "langgraph-checkpoint-sqlite>=3.0,<4",
    "mcp": "mcp>=1.20,<2",
    "pydantic": "pydantic>=2.10,<3",
    "fastapi": "fastapi>=0.115,<1",
    "uvicorn": "uvicorn>=0.30,<1",
    "streamlit": "streamlit>=1.40",
    "ultralytics": "ultralytics>=8.1",
    "networkx": "networkx>=3.0",
}


def _pip_install(packages: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", *packages, "--quiet"]
    print(f"  正在安装: {', '.join(packages)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [警告] pip 安装失败: {result.stderr.strip()[:200]}")
        return False
    return True


def check_and_install_dependencies():
    missing = {}
    for mod, spec in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing[mod] = spec

    if not missing:
        print("[OK] 所有 Python 依赖包已安装。")
        return

    print(f"[设置] 检测到 {len(missing)} 个依赖包缺失：")
    for mod, spec in missing.items():
        print(f"  - {spec}")
    print("  请先运行: pip install -r requirements.txt")



def check_env_file():
    env_path = os.path.join(ROOT, ".env")
    example_path = os.path.join(ROOT, ".env.example")
    if os.path.isfile(env_path):
        return True
    if os.path.isfile(example_path):
        import shutil
        shutil.copy2(example_path, env_path)
        print("[设置] 已从 .env.example 创建 .env 文件")
        print("  请编辑 .env，填入你的 API 密钥（QWEN_API_KEY 等）")
        return True
    print("[错误] .env 文件不存在，请手动创建并填入 API 密钥。")
    return False


def check_vector_db():
    db_dir = os.path.join(ROOT, "data", "mrag", "vector_db", "chroma")
    if os.path.isdir(db_dir) and os.listdir(db_dir):
        return True
    print("[设置] 未找到向量数据库，将在首次请求时自动构建。")
    return False


def stop_all():
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def api_get(path, timeout=5):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{API_PORT}{path}", timeout=timeout
        ) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def main():
    env = os.environ.copy()
    env["AQUABIO_API_URL"] = API_URL

    print("=" * 50)
    print("  AquaBio-AgentRAG 启动器")
    print("=" * 50)
    print()

    # Step 0: Environment checks
    print("[0/5] 检查运行环境...")

    if not check_env_file():
        sys.exit(1)

    check_and_install_dependencies()

    check_vector_db()
    print()

    # Step 1: Start FastAPI
    print(f"[1/5] 启动 FastAPI 后端（端口 {API_PORT}）...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_app:app",
         "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=ROOT,
        env=env,
    )
    procs.append(api_proc)

    # Step 2: Wait for health
    print("[2/5] 等待 API 健康检查...")
    for i in range(60):
        time.sleep(1)
        result = api_get("/api/health", timeout=3)
        if result and result.get("status") == "ok":
            print(f"  API 就绪（耗时 {i + 1}s）")
            break
        if i % 10 == 9:
            print(f"  仍在等待...（{i + 1}s）")
    else:
        print("  [警告] API 60秒内未响应，继续启动前端...")

    # Step 3: Warmup models
    print("[3/5] 预热模型（嵌入模型、Chroma、LangGraph）...")
    print("  首次启动可能需要几分钟，请耐心等待。")
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{API_PORT}/api/system/warmup",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            warmup = json.loads(r.read().decode())
        if warmup.get("status") == "ready":
            print(f"  预热完成（耗时 {warmup.get('elapsed_seconds', 0):.1f}s）")
        else:
            print(f"  预热异常: {warmup.get('detail', 'unknown')}")
    except Exception as e:
        print(f"  [警告] 预热失败: {e}")
        print("  前端仍会启动，但首次请求可能较慢。")

    # Step 4: Start Streamlit
    print(f"[4/5] 启动 Streamlit 前端（端口 {UI_PORT}）...")
    ui_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "mrag_app.py",
         "--server.port", str(UI_PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT,
        env=env,
    )
    procs.append(ui_proc)

    print()
    print("全部服务已启动！")
    print(f"  API 文档:  http://127.0.0.1:{API_PORT}/docs")
    print(f"  前端页面:  http://127.0.0.1:{UI_PORT}")
    print()
    print("按 Ctrl+C 停止所有服务。")
    print()

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        stop_all()
        print("所有服务已停止。")


if __name__ == "__main__":
    main()
