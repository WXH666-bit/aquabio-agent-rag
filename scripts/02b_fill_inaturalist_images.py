import argparse
from pathlib import Path

from aquabio_mrag.config import MRAGPaths
from aquabio_mrag.data_pipeline import fill_images_from_inaturalist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-species", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(
        fill_images_from_inaturalist(
            MRAGPaths.from_root(root), per_species=args.per_species
        )
    )
