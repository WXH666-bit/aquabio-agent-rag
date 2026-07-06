from pathlib import Path

from aquabio_mrag.config import MRAGPaths
from aquabio_mrag.data_pipeline import build_multimodal_documents


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(build_multimodal_documents(MRAGPaths.from_root(root)))
