import argparse
import json
from pathlib import Path

from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.vector_db import ChromaMRAGStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--document-file")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = MRAGPaths.from_root(root)
    settings = MRAGSettings.from_env()
    manifest = ChromaMRAGStore(paths, settings).build(
        document_file=args.document_file,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
