from pathlib import Path

from aquabio_mrag.config import MRAGPaths
from aquabio_mrag.data_pipeline import crawl_authoritative_text


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(crawl_authoritative_text(MRAGPaths.from_root(root)))

