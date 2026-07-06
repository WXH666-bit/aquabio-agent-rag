from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AquaBioAgent
from .pdf_ingest import ingest_directory
from .vector_store import LocalVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="AquaBio Agentic RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse PDFs into JSONL chunks")
    ingest.add_argument("path")
    ingest.add_argument("--output", default="data/index/pdf_chunks.jsonl")

    build_vector_db = subparsers.add_parser(
        "build-vector-db", help="Build the persistent local vector database"
    )
    build_vector_db.add_argument("--knowledge-dir", default="data/knowledge")
    build_vector_db.add_argument("--index-dir", default="data/index")
    build_vector_db.add_argument("--output", default="data/vector_db")

    vector_db_info = subparsers.add_parser(
        "vector-db-info", help="Show persistent vector database metadata"
    )
    vector_db_info.add_argument("--path", default="data/vector_db")

    ask = subparsers.add_parser("ask", help="Ask a text or image question")
    ask.add_argument("query")
    ask.add_argument("--image")
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--offline", action="store_true", help="Skip OpenRouter and run local tools only")

    args = parser.parse_args()
    if args.command == "ingest":
        count = ingest_directory(args.path, args.output)
        print(f"Ingested {count} PDF chunks into {args.output}")
        print("PDF 内容已更新，请继续执行：python -m aquabio.cli build-vector-db")
        return 0
    if args.command == "build-vector-db":
        manifest = LocalVectorStore(args.output).build(
            args.knowledge_dir, args.index_dir
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "vector-db-info":
        print(
            json.dumps(
                LocalVectorStore(args.path).info(), ensure_ascii=False, indent=2
            )
        )
        return 0

    mode = "本地离线工具" if args.offline else "OpenRouter + 本地知识库"
    print(f"正在运行：{mode}，请稍候...", file=sys.stderr, flush=True)
    if not args.offline:
        print("免费模型通常需要 20-120 秒；图片请求可能更久。", file=sys.stderr, flush=True)
    agent = AquaBioAgent(Path.cwd(), offline=args.offline)
    state = agent.run(args.query, args.image)
    print("处理完成。", file=sys.stderr, flush=True)
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print(state["answer"])
        if state["warnings"]:
            print("\nWarnings:")
            for warning in state["warnings"]:
                print(f"- {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
