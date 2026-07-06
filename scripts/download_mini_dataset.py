from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples"
PDFS = ROOT / "data/pdfs"
USER_AGENT = "AquaBio-AgentRAG/0.1 (educational prototype)"

QUERIES = {
    "starfish": "underwater starfish filetype:bitmap",
    "sea_urchin": "underwater sea urchin filetype:bitmap",
    "sea_cucumber": "underwater sea cucumber filetype:bitmap",
    "scallop": "underwater scallop filetype:bitmap",
    "jellyfish": "underwater jellyfish filetype:bitmap",
}

PAPERS = {
    "UIEB_paper.pdf": "https://arxiv.org/pdf/1901.05495",
    "DUO_paper.pdf": "https://arxiv.org/pdf/2106.05681",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
    ),
)


def plain(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def fetch_sample(label: str, query: str) -> dict:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 8,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 800,
        "format": "json",
        "origin": "*",
    }
    response = SESSION.get(
        "https://commons.wikimedia.org/w/api.php",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if not url or Path(url.split("?")[0]).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        metadata = info.get("extmetadata", {})
        extension = Path(url.split("?")[0]).suffix.lower().replace(".jpeg", ".jpg")
        path = SAMPLES / f"{label}_01{extension}"
        try:
            time.sleep(1.5)
            image = SESSION.get(url, timeout=90)
            image.raise_for_status()
        except requests.RequestException:
            continue
        path.write_bytes(image.content)
        return {
            "id": f"sample_{label}_01",
            "weak_label": label,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "title": page.get("title"),
            "artist": plain(metadata.get("Artist", {}).get("value", "")),
            "license": plain(metadata.get("LicenseShortName", {}).get("value", "")),
            "license_url": plain(metadata.get("LicenseUrl", {}).get("value", "")),
            "description": plain(metadata.get("ImageDescription", {}).get("value", "")),
            "source_page": info.get("descriptionurl"),
            "download_url": url,
        }
    raise RuntimeError(f"No reusable bitmap result found for {label}: {query}")


def download_papers() -> None:
    PDFS.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS.items():
        response = SESSION.get(url, timeout=120)
        response.raise_for_status()
        (PDFS / name).write_bytes(response.content)
        print(f"Downloaded {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", action="store_true", help="Also download two open-access papers")
    args = parser.parse_args()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    records = []
    for label, query in QUERIES.items():
        record = fetch_sample(label, query)
        records.append(record)
        print(f"Downloaded {record['path']} ({record['license']})")
    with (SAMPLES / "attribution.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.papers:
        download_papers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
