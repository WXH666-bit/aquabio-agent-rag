import argparse
import json
from pathlib import Path

from aquabio_mrag.config import MRAGPaths
from aquabio_mrag.pdf_pipeline import (
    download_registered_pdfs,
    parse_registered_pdfs,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-figures", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = MRAGPaths.from_root(root)
    download = download_registered_pdfs(paths)
    parsed = parse_registered_pdfs(paths, extract_figures=args.extract_figures)
    print(
        json.dumps(
            {"downloaded": download["registered"], **parsed},
            ensure_ascii=False,
            indent=2,
        )
    )
