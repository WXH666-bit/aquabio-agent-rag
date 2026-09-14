"""Local retrieval benchmark. Caption cases test retrieval, not VLM accuracy."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aquabio_mrag.config import MRAGPaths, MRAGSettings
from aquabio_mrag.retrieval import MultiSourceRetriever, RetrievalRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/retrieval_cases.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    retriever = MultiSourceRetriever(MRAGPaths.from_root(ROOT), MRAGSettings.from_env())
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            rows = retriever.search(RetrievalRequest(
                query=case["query"], task_type="image_qa" if case["kind"] == "caption" else "text_qa",
                top_k=args.top_k, species_ids=case.get("resolved_species"),
                source_types=["image_doc", "multimodal_pair"] if case["kind"] == "caption" else ["species_card", "species_text_chunk"],
            ))
            matches = [i for i, row in enumerate(rows, 1) if row["metadata"].get("species_id") == case["expected_species"]]
            results.append({"id": case["id"], "kind": case["kind"], "hit": bool(matches),
                            "reciprocal_rank": 1 / matches[0] if matches else 0,
                            "precision": len(matches) / args.top_k,
                            "document_ids": [row["id"] for row in rows],
                            "seconds": round(time.perf_counter()-started, 3)})
        except Exception as error:
            results.append({"id": case["id"], "kind": case["kind"], "hit": False,
                            "reciprocal_rank": 0, "precision": 0, "error": f"{type(error).__name__}: {error}"})
    report = {"top_k": args.top_k, "cases": len(results),
              "hit_rate": sum(row["hit"] for row in results) / max(1, len(results)),
              "mrr": sum(row["reciprocal_rank"] for row in results) / max(1, len(results)),
              "precision": sum(row["precision"] for row in results) / max(1, len(results)),
              "results": results}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return int(any("error" in row for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
