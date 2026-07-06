from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import requests


HEADERS = {
    "User-Agent": (
        "AquaBio-AgentRAG/1.0 "
        "(local research assistant; Wikimedia Commons image retrieval)"
    ),
    "Referer": "https://commons.wikimedia.org/",
}

MAP_MARKERS = (
    "distribution",
    "range map",
    "range-map",
    "distmap",
    "native range",
    "分布图",
    "分布地图",
)
GENERIC_QUERY_TERMS = {
    "animal",
    "fish",
    "image",
    "map",
    "marine",
    "range",
    "species",
    "distribution",
}


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(without_tags.casefold().split())


def _subject_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z][a-z.-]{2,}", query.casefold())
    return [
        term
        for term in terms
        if term not in GENERIC_QUERY_TERMS
    ]


def preferred_taxon_query(
    scientific_name: str,
    english_name: str,
) -> str:
    scientific = scientific_name.strip()
    english = english_name.strip()
    if re.fullmatch(
        r"[A-Z][a-z.-]{2,}\s+[a-z][a-z.-]{2,}",
        scientific,
    ):
        return scientific
    return english or scientific


def _is_distribution_candidate(
    title: str,
    description: str,
    query: str,
) -> bool:
    plain_title = _plain_text(title)
    plain_description = _plain_text(description)
    # Commons descriptions may cite an unrelated source map URL near the
    # end. Only the title and leading subject description identify what the
    # uploaded map actually depicts.
    subject_description = plain_description.split("http", 1)[0][:320]
    searchable = f"{plain_title} {subject_description}"
    if not any(marker in searchable for marker in MAP_MARKERS):
        return False
    terms = _subject_terms(query)
    return not terms or any(term in searchable for term in terms)


def fetch_commons_images(
    root: Path,
    species_id: str,
    query: str,
    top_k: int = 3,
    image_role: str = "specimen",
) -> tuple[list[dict[str, Any]], list[str]]:
    target_dir = (
        root / "data" / "mrag" / "network_images" / species_id
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"v2|{query}|{image_role}".encode("utf-8")
    ).hexdigest()[:12]
    manifest_path = target_dir / f"manifest_{cache_key}.json"
    if manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [
            row
            for row in cached
            if (root / row.get("image_path", "")).is_file()
            and (
                image_role != "distribution_map"
                or row.get("map_verified") is True
            )
        ]
        if rows:
            return rows[:top_k], []

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": (
            f"{query} distribution map"
            if image_role == "distribution_map"
            else query
        ),
        "gsrnamespace": 6,
        "gsrlimit": min(50, max(top_k * 20, 30)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": 800,
        "format": "json",
        "formatversion": 2,
        "origin": "*",
    }
    try:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=(8, 25),
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
    except Exception as error:
        return [], [
            f"Network image fallback failed: "
            f"{type(error).__name__}: {error}"
        ]

    rows = []
    for page in pages:
        if len(rows) >= top_k:
            break
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata", {})
        description = metadata.get("ImageDescription", {}).get(
            "value", ""
        )
        title = str(page.get("title", query))
        if (
            image_role == "distribution_map"
            and not _is_distribution_candidate(
                title, description, query
            )
        ):
            continue
        url = info.get("thumburl") or info.get("url", "")
        mime = info.get("mime", "")
        if not url or mime not in {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/svg+xml",
        }:
            continue
        try:
            image_response = requests.get(
                url, headers=HEADERS, timeout=(8, 25)
            )
            image_response.raise_for_status()
        except Exception:
            continue
        downloaded_mime = image_response.headers.get(
            "content-type", ""
        ).split(";", 1)[0]
        extension = (
            mimetypes.guess_extension(downloaded_mime)
            or mimetypes.guess_extension(mime)
            or ".jpg"
        )
        if extension == ".svg" and info.get("thumburl"):
            extension = ".png"
        digest = hashlib.sha256(image_response.content).hexdigest()
        target = target_dir / f"{digest[:16]}{extension}"
        target.write_bytes(image_response.content)
        relative = str(target.relative_to(root)).replace("\\", "/")
        rows.append(
            {
                "image_id": f"network_{digest[:12]}",
                "image_path": relative,
                "image_url": f"/files/{relative}",
                "caption": title,
                "scientific_name": query,
                "common_name": "",
                "page": None,
                "score": 0.0,
                "image_role": image_role,
                "source": "wikimedia_commons",
                "source_page": info.get("descriptionurl", ""),
                "license": (
                    metadata.get("LicenseShortName", {}).get(
                        "value", ""
                    )
                ),
                "map_verified": image_role == "distribution_map",
            }
        )
    manifest_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    warnings = [] if rows else [
        "No usable Wikimedia Commons image was found."
    ]
    return rows, warnings
