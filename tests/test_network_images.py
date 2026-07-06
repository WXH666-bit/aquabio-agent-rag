from __future__ import annotations

import unittest

from aquabio_web.network_images import (
    _is_distribution_candidate,
    preferred_taxon_query,
)


class NetworkImageTests(unittest.TestCase):
    def test_taxonomic_group_uses_english_search_name(self) -> None:
        self.assertEqual(
            preferred_taxon_query("Selachimorpha", "Shark"),
            "Shark",
        )
        self.assertEqual(
            preferred_taxon_query("Carcharodon carcharias", "White shark"),
            "Carcharodon carcharias",
        )

    def test_distribution_candidate_requires_map_and_subject(self) -> None:
        self.assertTrue(
            _is_distribution_candidate(
                "Sphyrnidae distribution map.svg",
                "Distribution map of hammerhead sharks.",
                "Shark",
            )
        )
        self.assertFalse(
            _is_distribution_candidate(
                "S. aeolus distribution map.png",
                (
                    "Distribution of Oriental trumpeter whiting using map "
                    "based on http://example/Tiger_shark_distribution.png"
                ),
                "Shark",
            )
        )
        self.assertFalse(
            _is_distribution_candidate(
                "Tiger shark.jpg",
                "Underwater photograph of a tiger shark.",
                "Shark",
            )
        )


if __name__ == "__main__":
    unittest.main()
