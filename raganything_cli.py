from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aquabio_raganything.config import (
    RAGAnythingPaths,
    RAGAnythingSettings,
)
from aquabio_raganything.audit import inspect_segment
from aquabio_raganything.book_native import build_book_native
from aquabio_raganything.indexer import (
    index_book_native_units,
    index_segments,
)
from aquabio_raganything.inventory import build_inventory
from aquabio_raganything.image_rag import (
    PDFImageVectorStore,
    build_pdf_image_assets,
    query_pdf_images,
)
from aquabio_raganything.query_adapter import (
    graph_neighbors,
    hybrid_search,
    index_status,
)
from aquabio_raganything.storage_audit import audit_persistent_storages


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AquaBio PDF RAG-Anything + LightRAG CLI"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")

    book_native = commands.add_parser(
        "book-native",
        help="Build deterministic book sections, taxa, page units and RAG chunks.",
    )
    book_native.add_argument(
        "--book",
        choices=["all", "sa_invertebrates", "living_guide"],
        default="all",
    )
    book_native.add_argument(
        "--extract-images",
        action="store_true",
        help="Write large embedded images to disk; metadata-only by default.",
    )
    book_native.add_argument("--min-image-dimension", type=int, default=128)

    index = commands.add_parser("index")
    index.add_argument("--scope", choices=["relevant", "full"], default="relevant")
    index.add_argument("--resume", action="store_true")
    index.add_argument("--limit", type=int)
    index.add_argument("--segment")

    native_index = commands.add_parser("index-book-native")
    native_index.add_argument(
        "--book",
        choices=["sa_invertebrates", "living_guide"],
        default="sa_invertebrates",
    )
    native_index.add_argument("--resume", action="store_true")
    native_index.add_argument("--limit-units", type=int)

    image_assets = commands.add_parser(
        "image-assets",
        help="Extract PDF images and bind them to taxon entities and pages.",
    )
    image_assets.add_argument(
        "--book",
        choices=["sa_invertebrates"],
        default="sa_invertebrates",
    )
    image_assets.add_argument("--min-image-dimension", type=int, default=128)
    image_assets.add_argument("--limit-units", type=int)
    image_assets.add_argument("--overwrite", action="store_true")

    image_index = commands.add_parser(
        "index-images",
        help="Embed PDF image captions into the dedicated Chroma collection.",
    )
    image_index.add_argument(
        "--book",
        choices=["sa_invertebrates"],
        default="sa_invertebrates",
    )
    image_index.add_argument("--batch-size", type=int, default=16)
    image_index.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the PDF image collection.",
    )

    image_query = commands.add_parser(
        "image-query",
        help="Retrieve entity-aligned sample images extracted from the PDF.",
    )
    image_query.add_argument("--query", required=True)
    image_query.add_argument("--entity", default="")
    image_query.add_argument("--top-k", type=int, default=5)

    commands.add_parser("status")
    commands.add_parser("audit-storage")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--segment", required=True)
    inspect.add_argument(
        "--scope", choices=["relevant", "full"], default="relevant"
    )

    query = commands.add_parser("query")
    query.add_argument("--mode", choices=["hybrid"], default="hybrid")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=12)

    neighbors = commands.add_parser("neighbors")
    neighbors.add_argument("--entity", required=True)
    neighbors.add_argument("--depth", type=int, default=1)

    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = RAGAnythingPaths.from_root(root)
    settings = RAGAnythingSettings.from_env()
    paths.ensure()

    if args.command == "inventory":
        _print(build_inventory(paths, settings))
    elif args.command == "book-native":
        _print(
            build_book_native(
                paths,
                book_id=args.book,
                extract_images=args.extract_images,
                min_image_dimension=args.min_image_dimension,
            )
        )
    elif args.command == "index":
        _print(
            asyncio.run(
                index_segments(
                    paths,
                    settings,
                    args.scope,
                    resume=args.resume,
                    limit=args.limit,
                    segment_id=args.segment,
                )
            )
        )
    elif args.command == "index-book-native":
        _print(
            asyncio.run(
                index_book_native_units(
                    paths,
                    settings,
                    args.book,
                    resume=args.resume,
                    limit_units=args.limit_units,
                )
            )
        )
    elif args.command == "image-assets":
        _print(
            build_pdf_image_assets(
                paths,
                book_id=args.book,
                min_image_dimension=args.min_image_dimension,
                limit_units=args.limit_units,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "index-images":
        _print(
            PDFImageVectorStore(paths, settings).build(
                book_id=args.book,
                batch_size=args.batch_size,
                reset=args.reset,
            )
        )
    elif args.command == "image-query":
        _print(
            query_pdf_images(
                paths,
                settings,
                query=args.query,
                top_k=args.top_k,
                entity=args.entity,
            )
        )
    elif args.command == "status":
        _print(index_status(paths))
    elif args.command == "audit-storage":
        _print(audit_persistent_storages(paths))
    elif args.command == "inspect":
        _print(inspect_segment(paths, args.segment, args.scope))
    elif args.command == "query":
        _print(
            asyncio.run(
                hybrid_search(paths, settings, args.query, args.top_k)
            )
        )
    elif args.command == "neighbors":
        _print(graph_neighbors(paths, args.entity, args.depth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
