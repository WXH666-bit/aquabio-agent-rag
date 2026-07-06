from __future__ import annotations

import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import MRAGPaths
from .io_utils import read_jsonl, write_jsonl
from .models import ImageDocument, MultimodalPair, RAGDocument


USER_AGENT = "AquaBio-MRAG/0.2 (educational research project)"
ALLOWED_LICENSE_MARKERS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "creative commons",
)


def _session(max_retries: int = 3) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=max_retries,
                backoff_factor=0.8,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
        ),
    )
    return session


def _plain(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def _decode_literal_unicode(value: str) -> str:
    if "\\u" not in value:
        return value
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )


def load_species(paths: MRAGPaths) -> list[dict]:
    species_list = json.loads(paths.species_list.read_text(encoding="utf-8"))
    for species in species_list:
        species["keywords"] = [
            _decode_literal_unicode(keyword) for keyword in species["keywords"]
        ]
    return species_list


def crawl_authoritative_text(paths: MRAGPaths, delay: float = 0.4) -> dict:
    paths.ensure()
    session = _session()
    wiki_records = []
    worms_records = []
    for species in load_species(paths):
        species_id = species["species_id"]
        wiki_title = species["wiki_title"]
        wiki_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + quote_plus(wiki_title).replace("+", "_")
        )
        wiki_response = session.get(wiki_url, timeout=60)
        wiki_data = (
            wiki_response.json()
            if wiki_response.ok
            else {"error": wiki_response.status_code}
        )
        wiki_records.append(
            {
                "species_id": species_id,
                "source_url": species["source_urls"]["wikipedia"],
                "retrieved_from": wiki_url,
                "status": wiki_response.status_code,
                "title": wiki_data.get("title", ""),
                "extract": wiki_data.get("extract", ""),
                "description": wiki_data.get("description", ""),
                "content_urls": wiki_data.get("content_urls", {}),
            }
        )

        worms_name = quote_plus(species["worms_name"])
        worms_url = (
            f"https://www.marinespecies.org/rest/AphiaRecordsByName/{worms_name}"
            "?like=false&marine_only=true"
        )
        worms_response = session.get(worms_url, timeout=60)
        worms_data = (
            worms_response.json()
            if worms_response.ok
            else {"error": worms_response.status_code}
        )
        worms_records.append(
            {
                "species_id": species_id,
                "source_url": species["source_urls"]["worms"],
                "retrieved_from": worms_url,
                "status": worms_response.status_code,
                "records": worms_data,
            }
        )
        print(f"text sources: {species_id}")
        time.sleep(delay)

    wiki_count = write_jsonl(
        paths.raw_dir / "wikipedia_records.jsonl", wiki_records
    )
    worms_count = write_jsonl(paths.raw_dir / "worms_records.jsonl", worms_records)
    return {"wikipedia": wiki_count, "worms": worms_count}


def _commons_candidates(
    session: requests.Session, query: str, limit: int = 50
) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": 640,
        "format": "json",
        "formatversion": 2,
        "origin": "*",
    }
    response = session.get(
        "https://commons.wikimedia.org/w/api.php", params=params, timeout=90
    )
    response.raise_for_status()
    return response.json().get("query", {}).get("pages", [])


def _candidate_relevance(candidate: dict, terms: list[str]) -> int:
    info = candidate.get("imageinfo", [{}])[0]
    metadata = info.get("extmetadata", {})
    haystack = " ".join(
        [
            candidate.get("title", ""),
            _plain(metadata.get("ImageDescription", {}).get("value", "")),
            _plain(metadata.get("Categories", {}).get("value", "")),
        ]
    ).lower()
    return sum(1 for term in terms if term.lower() in haystack)


def crawl_commons_images(
    paths: MRAGPaths,
    per_species: int = 10,
    delay: float = 0.8,
) -> dict:
    """Download license-traceable images; Bing URLs are retained for discovery."""
    paths.ensure()
    session = _session()
    download_session = _session(max_retries=0)
    image_docs_path = paths.knowledge_dir / "image_docs.jsonl"
    failure_path = paths.raw_dir / "image_crawl_failures.jsonl"
    all_records: list[dict] = read_jsonl(image_docs_path)
    failures: list[dict] = read_jsonl(failure_path)
    completed_species = {
        species_id
        for species_id in {row["species_id"] for row in all_records}
        if sum(row["species_id"] == species_id for row in all_records) >= per_species
    }

    for species in load_species(paths):
        species_id = species["species_id"]
        if species_id in completed_species:
            print(f"images: {species_id} {per_species}/{per_species} cached")
            continue
        english_name = species["english_name"]
        chinese_name = species["chinese_name"]
        query = f"{english_name} underwater"
        bing_url = (
            "https://www.bing.com/images/search?q=" + quote_plus(query)
        )
        search_terms = [
            english_name.lower(),
            species["scientific_name"].split("/")[0].strip().lower(),
            *[term.lower() for term in species["keywords"][:2]],
        ]
        existing = [
            row for row in all_records if row["species_id"] == species_id
        ]
        candidates = _commons_candidates(session, query, limit=50)
        candidates.sort(
            key=lambda item: _candidate_relevance(item, search_terms), reverse=True
        )

        species_dir = paths.images_dir / species_id
        species_dir.mkdir(parents=True, exist_ok=True)
        accepted = len(existing)
        seen_hashes: set[str] = set()
        existing_urls = {row["image_url"] for row in existing}
        for candidate in candidates:
            if accepted >= per_species:
                break
            info = candidate.get("imageinfo", [{}])[0]
            metadata = info.get("extmetadata", {})
            license_name = _plain(
                metadata.get("LicenseShortName", {}).get("value", "")
            )
            if not license_name or not any(
                marker in license_name.lower() for marker in ALLOWED_LICENSE_MARKERS
            ):
                continue
            url = info.get("thumburl") or info.get("url")
            mime = info.get("mime", "")
            if (
                not url
                or url in existing_urls
                or mime not in {"image/jpeg", "image/png", "image/webp"}
            ):
                continue
            if _candidate_relevance(candidate, search_terms) <= 0:
                continue

            try:
                response = download_session.get(url, timeout=(8, 15))
                response.raise_for_status()
                digest = hashlib.sha256(response.content).hexdigest()
                if digest in seen_hashes:
                    continue
                extension = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                }[mime]
                image_id = f"img_{species_id}_{accepted + 1:03d}"
                image_path = species_dir / f"{image_id}{extension}"
                image_path.write_bytes(response.content)
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                if width < 256 or height < 256:
                    image_path.unlink(missing_ok=True)
                    continue
            except Exception as error:
                failures.append(
                    {
                        "species_id": species_id,
                        "url": url,
                        "error": str(error),
                    }
                )
                continue

            description = _plain(
                metadata.get("ImageDescription", {}).get("value", "")
            )
            caption = (
                description
                or f"Underwater reference image of {english_name} ({chinese_name})."
            )
            keywords = list(
                dict.fromkeys(
                    [
                        english_name.lower(),
                        chinese_name,
                        species["scientific_name"],
                        "underwater",
                        *species["keywords"],
                    ]
                )
            )
            record = ImageDocument(
                id=image_id,
                species_id=species_id,
                english_name=english_name,
                chinese_name=chinese_name,
                image_path=str(image_path.relative_to(paths.root)).replace("\\", "/"),
                image_url=url,
                source_page=info.get("descriptionurl", ""),
                license=license_name,
                license_url=_plain(
                    metadata.get("LicenseUrl", {}).get("value", "")
                ),
                author=_plain(metadata.get("Artist", {}).get("value", "")),
                caption=caption,
                visual_keywords=keywords,
                embedding_text=" ".join([english_name, chinese_name, *keywords, caption]),
                width=width,
                height=height,
                bing_discovery_url=bing_url,
            )
            all_records.append(record.model_dump())
            write_jsonl(image_docs_path, all_records)
            seen_hashes.add(digest)
            existing_urls.add(url)
            accepted += 1
            time.sleep(delay)

        if accepted < per_species:
            failures.append(
                {
                    "species_id": species_id,
                    "error": f"only {accepted}/{per_species} accepted images",
                }
            )
        print(f"images: {species_id} {accepted}/{per_species}")
        write_jsonl(failure_path, failures)

    image_count = write_jsonl(image_docs_path, all_records)
    write_jsonl(failure_path, failures)
    return {
        "species": len(load_species(paths)),
        "images": image_count,
        "expected": len(load_species(paths)) * per_species,
        "failures": len(failures),
    }


def fill_images_from_inaturalist(
    paths: MRAGPaths,
    per_species: int = 10,
    delay: float = 0.15,
) -> dict:
    """Fill Commons gaps with licensed iNaturalist research observations."""
    paths.ensure()
    session = _session(max_retries=2)
    image_docs_path = paths.knowledge_dir / "image_docs.jsonl"
    all_records = read_jsonl(image_docs_path)
    species_list = load_species(paths)
    species_by_id = {item["species_id"]: item for item in species_list}

    # Repair legacy literal Unicode keywords before they enter the vector store.
    for record in all_records:
        species = species_by_id[record["species_id"]]
        keywords = list(
            dict.fromkeys(
                [
                    species["english_name"].lower(),
                    species["chinese_name"],
                    species["scientific_name"],
                    "underwater",
                    *species["keywords"],
                ]
            )
        )
        record["visual_keywords"] = keywords
        record["embedding_text"] = " ".join(
            [
                species["english_name"],
                species["chinese_name"],
                *keywords,
                record["caption"],
            ]
        )
    write_jsonl(image_docs_path, all_records)

    failures: list[dict] = []
    for species in species_list:
        species_id = species["species_id"]
        existing = [
            row for row in all_records if row["species_id"] == species_id
        ]
        if len(existing) >= per_species:
            print(f"iNaturalist: {species_id} {per_species}/{per_species} cached")
            continue

        # iNaturalist resolves these broad common names more reliably than
        # family/order names such as Nephropidae or Anguilliformes.
        taxon_name = species["english_name"]
        response = session.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_name": taxon_name,
                "photos": "true",
                "quality_grade": "research",
                "per_page": 200,
                "order_by": "votes",
                "order": "desc",
            },
            timeout=(10, 40),
        )
        response.raise_for_status()
        observations = response.json().get("results", [])
        existing_urls = {row["image_url"] for row in existing}
        accepted = len(existing)

        for observation in observations:
            if accepted >= per_species:
                break
            taxon = observation.get("taxon") or {}
            place = observation.get("place_guess") or ""
            observed_on = observation.get("observed_on_string") or ""
            source_page = observation.get("uri") or (
                "https://www.inaturalist.org/observations/"
                + str(observation.get("id", ""))
            )
            for photo in observation.get("photos", []):
                if accepted >= per_species:
                    break
                license_code = (photo.get("license_code") or "").lower()
                if license_code not in {
                    "cc0",
                    "cc-by",
                    "cc-by-sa",
                    "cc-by-nc",
                    "cc-by-nc-sa",
                }:
                    continue
                image_url = (photo.get("url") or "").replace(
                    "/square.", "/medium."
                )
                if not image_url or image_url in existing_urls:
                    continue
                try:
                    image_response = session.get(
                        image_url, timeout=(8, 20)
                    )
                    image_response.raise_for_status()
                    extension = Path(image_url.split("?")[0]).suffix.lower()
                    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                        extension = ".jpg"
                    image_id = f"img_{species_id}_{accepted + 1:03d}"
                    image_path = (
                        paths.images_dir / species_id / f"{image_id}{extension}"
                    )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_response.content)
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        width, height = image.size
                    if width < 256 or height < 256:
                        image_path.unlink(missing_ok=True)
                        continue
                except Exception as error:
                    failures.append(
                        {
                            "species_id": species_id,
                            "url": image_url,
                            "error": str(error),
                        }
                    )
                    continue

                scientific_name = taxon.get("name") or taxon_name
                common_name = taxon.get("preferred_common_name") or ""
                caption_parts = [
                    common_name,
                    f"({scientific_name})" if scientific_name else "",
                    f"observed at {place}" if place else "",
                    f"on {observed_on}" if observed_on else "",
                ]
                caption = " ".join(part for part in caption_parts if part).strip()
                keywords = list(
                    dict.fromkeys(
                        [
                            species["english_name"].lower(),
                            species["chinese_name"],
                            species["scientific_name"],
                            scientific_name,
                            common_name,
                            "underwater",
                            *species["keywords"],
                        ]
                    )
                )
                record = ImageDocument(
                    id=image_id,
                    species_id=species_id,
                    english_name=species["english_name"],
                    chinese_name=species["chinese_name"],
                    image_path=str(image_path.relative_to(paths.root)).replace(
                        "\\", "/"
                    ),
                    image_url=image_url,
                    source_page=source_page,
                    license=license_code.upper(),
                    license_url=(
                        "https://creativecommons.org/publicdomain/zero/1.0/"
                        if license_code == "cc0"
                        else "https://creativecommons.org/licenses/"
                        + license_code.removeprefix("cc-")
                        + "/4.0/"
                    ),
                    author=photo.get("attribution") or "",
                    caption=caption
                    or f"Research-grade observation of {species['english_name']}.",
                    visual_keywords=keywords,
                    embedding_text=" ".join(
                        [
                            species["english_name"],
                            species["chinese_name"],
                            *keywords,
                            caption,
                        ]
                    ),
                    width=width,
                    height=height,
                    bing_discovery_url=(
                        "https://www.bing.com/images/search?q="
                        + quote_plus(f"{species['english_name']} underwater")
                    ),
                )
                all_records.append(record.model_dump())
                existing_urls.add(image_url)
                accepted += 1
                write_jsonl(image_docs_path, all_records)
                time.sleep(delay)

        if accepted < per_species:
            failures.append(
                {
                    "species_id": species_id,
                    "error": f"only {accepted}/{per_species} images after fallback",
                }
            )
        print(f"iNaturalist: {species_id} {accepted}/{per_species}")

    write_jsonl(paths.raw_dir / "inaturalist_failures.jsonl", failures)
    return {
        "species": len(species_list),
        "images": len(all_records),
        "expected": len(species_list) * per_species,
        "failures": len(failures),
    }


def build_multimodal_documents(paths: MRAGPaths) -> dict:
    paths.ensure()
    seed_knowledge = paths.seed_db / "data" / "knowledge"
    cards = read_jsonl(seed_knowledge / "species_cards.jsonl")
    text_docs = read_jsonl(seed_knowledge / "species_text_docs.jsonl")
    image_docs = read_jsonl(paths.knowledge_dir / "image_docs.jsonl")
    pdf_chunks = read_jsonl(paths.knowledge_dir / "pdf_chunks.jsonl")
    pdf_figures = read_jsonl(paths.knowledge_dir / "pdf_figures.jsonl")
    wiki_records = {
        row["species_id"]: row
        for row in read_jsonl(paths.raw_dir / "wikipedia_records.jsonl")
    }
    worms_records = {
        row["species_id"]: row
        for row in read_jsonl(paths.raw_dir / "worms_records.jsonl")
    }

    text_by_species: dict[str, list[dict]] = {}
    for document in text_docs:
        text_by_species.setdefault(document["species_id"], []).append(document)

    pairs = []
    for image in image_docs:
        species_docs = text_by_species.get(image["species_id"], [])
        preferred = [
            item
            for item in species_docs
            if item.get("chunk_type") in {"overview", "visual_features", "habitat"}
        ]
        text_ids = [item["id"] for item in preferred]
        rag_context = "\n".join(item["content"] for item in preferred)
        pair = MultimodalPair(
            id=f"pair_{image['id']}",
            species_id=image["species_id"],
            image_id=image["id"],
            text_ids=text_ids,
            image_caption=image["caption"],
            rag_context=rag_context,
            embedding_text=" ".join(
                [
                    image["embedding_text"],
                    *[item["content"] for item in preferred],
                ]
            ),
        )
        pairs.append(pair.model_dump())

    write_jsonl(paths.knowledge_dir / "multimodal_pairs.jsonl", pairs)

    combined: list[dict] = []
    for card in cards:
        species_id = card["species_id"]
        wiki = wiki_records.get(species_id, {})
        worms = worms_records.get(species_id, {})
        source_metadata = {
            "chinese_name": card.get("chinese_name", ""),
            "english_name": card.get("english_name", ""),
            "scientific_name": card.get("scientific_name", ""),
            "chunk_type": "species_card",
            "wikipedia_url": card.get("source_urls", {}).get("wikipedia", ""),
            "worms_url": card.get("source_urls", {}).get("worms", ""),
            "wikipedia_extract": wiki.get("extract", ""),
            "worms_status": worms.get("status", ""),
        }
        combined.append(
            RAGDocument(
                id=card["id"],
                source_type="species_card",
                species_id=species_id,
                modality="species_card",
                content=card["overview"],
                embedding_text=" ".join(
                    [
                        card["english_name"],
                        card["chinese_name"],
                        card["scientific_name"],
                        card["overview"],
                        wiki.get("extract", ""),
                    ]
                ),
                metadata=source_metadata,
            ).model_dump()
        )

    for document in text_docs:
        combined.append(
            RAGDocument(
                id=document["id"],
                source_type="species_text_chunk",
                species_id=document["species_id"],
                modality="text",
                content=document["content"],
                embedding_text=" ".join(
                    [
                        document.get("english_name", ""),
                        document.get("chinese_name", ""),
                        document.get("scientific_name", ""),
                        document.get("title", ""),
                        document["content"],
                        " ".join(document.get("keywords", [])),
                    ]
                ),
                metadata={
                    "chunk_type": document.get("chunk_type", ""),
                    "title": document.get("title", ""),
                    "wikipedia_url": document.get("source_urls", {}).get(
                        "wikipedia", ""
                    ),
                    "worms_url": document.get("source_urls", {}).get("worms", ""),
                },
            ).model_dump()
        )

    for image in image_docs:
        combined.append(
            RAGDocument(
                id=image["id"],
                source_type="image_doc",
                species_id=image["species_id"],
                modality="image_caption",
                content=image["caption"],
                embedding_text=image["embedding_text"],
                metadata={
                    "image_path": image["image_path"],
                    "source_page": image["source_page"],
                    "license": image["license"],
                    "license_url": image.get("license_url", ""),
                    "author": image.get("author", ""),
                    "bing_discovery_url": image.get("bing_discovery_url", ""),
                },
            ).model_dump()
        )

    for pair in pairs:
        combined.append(
            RAGDocument(
                id=pair["id"],
                source_type="multimodal_pair",
                species_id=pair["species_id"],
                modality="image_text_pair",
                content=pair["rag_context"],
                embedding_text=pair["embedding_text"],
                metadata={
                    "image_id": pair["image_id"],
                    "text_ids": pair["text_ids"],
                },
            ).model_dump()
        )

    combined.extend(pdf_chunks)
    combined.extend(pdf_figures)

    count = write_jsonl(
        paths.knowledge_dir / "rag_documents_combined.jsonl", combined
    )
    return {
        "species_cards": len(cards),
        "species_text_docs": len(text_docs),
        "image_docs": len(image_docs),
        "multimodal_pairs": len(pairs),
        "pdf_chunks": len(pdf_chunks),
        "pdf_figures": len(pdf_figures),
        "combined_documents": count,
    }
