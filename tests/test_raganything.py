import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from aquabio_raganything.config import RAGAnythingPaths
from aquabio_raganything.image_rag import requested_page
from aquabio_raganything.query_adapter import (
    _build_evidence,
    _parse_graph_context,
)
from aquabio_raganything.storage_audit import audit_persistent_storages


class RAGAnythingAdapterTests(unittest.TestCase):
    def test_requested_page_supports_chinese_and_english_forms(self):
        self.assertEqual(requested_page("PDF 印刷页413 的生物图片"), 413)
        self.assertEqual(requested_page("请看第 413 页"), 413)
        self.assertEqual(requested_page("printed page 413 specimen"), 413)

    def test_graph_context_and_page_evidence(self):
        content = """
Knowledge Graph Data (Entity):
```json
{"entity": "Luidia Africana", "type": "species"}
```
Knowledge Graph Data (Relationship):
```json
{"entity1": "Luidia Africana", "entity2": "Arms",
 "description": "has long arms"}
```
[DOC_ID=doc_x][SOURCE=guide.pdf][PAGE=9]
distinguishing features
[DOC_ID=doc_x][SOURCE=guide.pdf][PAGE=9]
Long flattened arms.
"""
        entities, relations = _parse_graph_context(content)
        evidence = _build_evidence(content, entities, relations, 5)
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(relations), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["page"], 9)
        self.assertIn("Long flattened arms", evidence[0]["content"])

    def test_storage_audit_requires_every_layer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = RAGAnythingPaths.from_root(root)
            paths.ensure()
            working = paths.working_dir
            doc_id = "doc_x"
            (working / "kv_store_doc_status.json").write_text(
                json.dumps({doc_id: {"status": "processed"}}),
                encoding="utf-8",
            )
            self.assertFalse(
                audit_persistent_storages(paths, doc_id)["valid"]
            )

            for filename in (
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            ):
                (working / filename).write_text(
                    json.dumps({"data": [{"id": "x"}]}),
                    encoding="utf-8",
                )
            for filename in (
                "kv_store_text_chunks.json",
                "kv_store_full_entities.json",
                "kv_store_full_relations.json",
            ):
                (working / filename).write_text(
                    json.dumps({"x": {}}),
                    encoding="utf-8",
                )
            graph = nx.Graph()
            graph.add_edge("species", "feature")
            nx.write_graphml(
                graph,
                working / "graph_chunk_entity_relation.graphml",
            )
            self.assertTrue(
                audit_persistent_storages(paths, doc_id)["valid"]
            )


if __name__ == "__main__":
    unittest.main()
