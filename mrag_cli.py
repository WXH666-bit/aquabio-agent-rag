from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.conversation import ConversationStore
from aquabio_mrag.vector_db import ChromaMRAGStore
from aquabio_mrag.workflow import AquaBioMRAGWorkflow
from aquabio_mrag.mcp_client import project_mcp_client


def print_json(value: object) -> None:
    def encode(item: object) -> object:
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if hasattr(item, "value"):
            return getattr(item, "value")
        return str(item)

    print(
        json.dumps(
            value, ensure_ascii=False, indent=2, default=encode
        )
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="AquaBio-MRAG strict conversational workflow"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask")
    ask.add_argument("--query", default="")
    ask.add_argument("--image")
    ask.add_argument(
        "--session",
        default="default",
        help="Persistent conversation ID; defaults to 'default'.",
    )
    ask.add_argument("--offline", action="store_true")
    ask.add_argument("--json", action="store_true")
    ask.add_argument(
        "--hitl",
        action="store_true",
        help="Enable durable LangGraph interrupt/resume for clarification.",
    )
    ask.add_argument(
        "--mcp-retrieval",
        action="store_true",
        help=(
            "Route Chroma text/image/multimodal/PDF chunk retrieval through "
            "MCP tools so tool_calls can prove the external tool path."
        ),
    )

    resume = sub.add_parser("resume")
    resume.add_argument("--session", required=True)
    resume.add_argument("--answer", required=True)
    resume.add_argument("--offline", action="store_true")
    resume.add_argument("--json", action="store_true")

    pending = sub.add_parser("pending")
    pending.add_argument("--session", required=True)

    mcp_tools = sub.add_parser("mcp-tools")
    mcp_tools.add_argument(
        "--server", choices=["chroma", "raganything", "all"], default="all"
    )

    sub.add_parser("db-info")

    history = sub.add_parser("history")
    history.add_argument("--session", default="default")
    history.add_argument("--json", action="store_true")

    clear = sub.add_parser("clear-session")
    clear.add_argument("--session", default="default")

    sub.add_parser("list-sessions")

    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = MRAGPaths.from_root(root)
    paths.ensure()
    settings = MRAGSettings.from_env()
    conversations = ConversationStore(paths.sessions_dir)

    if args.command == "db-info":
        print_json(ChromaMRAGStore(paths, settings).info())
        return 0
    if args.command == "history":
        session = conversations.load(args.session)
        if args.json:
            print_json(session)
        else:
            print(f"会话：{session['session_id']}")
            for turn in session.get("turns", []):
                print(f"\n[{turn.get('turn_index')}] 用户：{turn.get('user_query')}")
                print(f"助手：{turn.get('assistant_answer')}")
        return 0
    if args.command == "clear-session":
        removed = conversations.clear(args.session)
        print(
            f"已清空会话：{args.session}"
            if removed
            else f"会话不存在：{args.session}"
        )
        return 0
    if args.command == "list-sessions":
        print_json(conversations.list_sessions())
        return 0
    if args.command == "mcp-tools":
        client = project_mcp_client(root)
        servers = (
            ["chroma", "raganything"]
            if args.server == "all"
            else [args.server]
        )
        print_json(
            {
                server: client.list_tools_sync(server)
                for server in servers
            }
        )
        return 0

    workflow = AquaBioMRAGWorkflow(
        paths, settings, offline=getattr(args, "offline", False)
    )
    if args.command == "pending":
        print_json(workflow.pending(args.session))
        return 0
    if args.command == "resume":
        result = workflow.resume(args.session, args.answer)
    else:
        result = workflow.invoke(
            args.query,
            args.image,
            session_id=args.session,
            hitl=args.hitl,
            options={"mcp_retrieval_enabled": args.mcp_retrieval},
        )

    if args.json:
        print_json(result)
    elif result.get("__interrupt__"):
        print("流程已暂停，等待人工输入：")
        for item in result["__interrupt__"]:
            print(getattr(item, "value", item))
        print(
            f'\n恢复命令：python mrag_cli.py resume --session "{args.session}" '
            '--answer "你的补充信息"'
        )
    else:
        print(result["final_answer"])
        image_rows = [
            row
            for row in result.get("image_context", [])
            if row.get("metadata", {}).get("image_path")
        ]
        if image_rows:
            print("\nPDF 实体样例图片：")
            for index, row in enumerate(image_rows[:8], start=1):
                metadata = row.get("metadata", {})
                name = (
                    metadata.get("scientific_name")
                    or metadata.get("common_name")
                    or metadata.get("entity_id")
                    or row.get("id")
                )
                print(
                    f"{index}. {name} | PDF page "
                    f"{metadata.get('page', '')}"
                )
                print(f"   {metadata.get('absolute_image_path') or metadata.get('image_path')}")
        if result.get("warnings"):
            print("\n警告：")
            for warning in result["warnings"]:
                print(f"- {warning}")
        print(f"\n会话：{args.session}")
        print(f"记录：{result.get('session_file', '')}")
        print("\n执行轨迹：")
        for item in result.get("trace", []):
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
