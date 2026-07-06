from __future__ import annotations

import unittest
from pathlib import Path

import fitz

from aquabio_mrag.evidence import EvidenceItem, weighted_rrf
from aquabio_mrag.mcp_client import project_mcp_client
from aquabio_raganything.book_native import parse_sa_taxa_catalog
from aquabio_raganything.config import BOOKS
from aquabio_raganything.config import RAGAnythingPaths
from aquabio_raganything.image_rag import (
    asks_for_reference_images,
    entity_image_records,
    expand_image_query,
)


ROOT = Path(__file__).resolve().parents[1]


class BookNativeTests(unittest.TestCase):
    def test_sa_table_layout_recovers_full_catalog(self):
        source = ROOT / "data" / "mrag" / "pdfs" / BOOKS["sa_invertebrates"]
        document = fitz.open(source)
        try:
            rows = parse_sa_taxa_catalog(document, source.name)
        finally:
            document.close()
        self.assertEqual(len(rows), 409)
        self.assertEqual(rows[0].scientific_name, "Haliclona (Haliclona) anonyma")
        self.assertEqual(rows[0].printed_page, 41)
        self.assertEqual(rows[-1].printed_page, 493)


class AgentInfrastructureTests(unittest.TestCase):
    def test_pdf_image_intent_and_chinese_query_expansion(self):
        self.assertTrue(asks_for_reference_images("给我海星的样例图片"))
        self.assertIn("Asteroidea", expand_image_query("海星图片"))

    def test_entity_registry_returns_real_jasus_images(self):
        paths = RAGAnythingPaths.from_root(ROOT)
        rows = entity_image_records(paths, "Jasus lalandii", top_k=3)
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(
            all(row["scientific_name"] == "Jasus lalandii" for row in rows)
        )
        self.assertTrue(all(row["image_exists"] for row in rows))

    def test_weighted_rrf_deduplicates_same_evidence(self):
        common = dict(
            id="one",
            doc_id="doc",
            page=10,
            content="same evidence",
        )
        result = weighted_rrf(
            {
                "graph": [
                    EvidenceItem(source="lightrag_graph", **common)
                ],
                "chroma": [EvidenceItem(source="chroma", **common)],
            },
            {"graph": 0.55, "chroma": 0.45},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].metadata["fused_sources"],
            ["chroma", "graph"],
        )

    def test_project_mcp_client_registers_both_servers(self):
        client = project_mcp_client(ROOT)
        self.assertEqual(set(client.servers), {"chroma", "raganything"})


if __name__ == "__main__":
    unittest.main()
