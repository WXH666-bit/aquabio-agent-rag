from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any

import requests


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "AquaBio-AgentRAG/1.0 "
        "(local research assistant; Wikipedia evidence retrieval)"
    ),
    "Referer": "https://en.wikipedia.org/",
}
WEB_RESEARCH_MARKERS = (
    "天敌",
    "捕食者",
    "谁吃",
    "被什么吃",
    "食物链",
    "网上",
    "网络",
    "查资料",
    "最新资料",
    "natural predator",
    "predators",
    "web search",
    "internet",
)
RELATED_IMAGE_MARKERS = (
    "天敌",
    "捕食者",
    "谁吃",
    "被什么吃",
    "食物链",
    "predators",
)
KNOWN_RELATED_IMAGE_TARGETS = {
    "starfish": "sea otter",
    "sea star": "sea otter",
    "asteroidea": "sea otter",
}


def needs_web_research(query: str) -> bool:
    lowered = query.casefold()
    return any(marker in lowered for marker in WEB_RESEARCH_MARKERS)


def asks_for_related_entity_image(query: str) -> bool:
    lowered = query.casefold()
    return any(marker in lowered for marker in RELATED_IMAGE_MARKERS)


def plan_web_research(
    llm: Any,
    subject: str,
    query: str,
) -> dict[str, str]:
    subject_key = subject.casefold()
    known_target = next(
        (
            target
            for marker, target in KNOWN_RELATED_IMAGE_TARGETS.items()
            if marker in subject_key
        ),
        "",
    )
    fallback_target = known_target or f"{subject} predator"
    fallback = {
        "search_query": subject,
        "image_query": fallback_target,
        "image_target": fallback_target,
        "focus": "natural predators",
    }
    if not getattr(llm, "enabled", False):
        return fallback
    prompt = {
        "subject": subject,
        "question": query,
        "task": (
            "Create a concise English Wikipedia search query. If the user "
            "asks for an image of a related organism such as a predator, "
            "choose one concrete, visually identifiable related organism "
            "as image_target. The image_target must not be the subject "
            "itself. Prefer a well-documented predator with a clear "
            "Wikimedia Commons photograph. Return JSON only."
        ),
        "schema": {
            "search_query": "string",
            "image_query": "string",
            "image_target": "string",
            "focus": "string",
        },
    }
    try:
        raw = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You plan factual web research. Do not answer the "
                        "question. Return one JSON object only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            max_tokens=300,
            max_continuations=0,
        )
        value = llm._extract_json_object(raw) or {}
        result = {
            key: str(value.get(key, fallback[key])).strip()
            for key in fallback
        }
        if asks_for_related_entity_image(query):
            result["search_query"] = subject
            target = result["image_target"].casefold()
            if (
                not target
                or target in subject_key
                or subject_key in target
                or target in {"predator", "predators", "natural predators"}
            ):
                result["image_target"] = fallback_target
                result["image_query"] = fallback_target
        return result if result["search_query"] else fallback
    except Exception:
        return fallback


def _clean_snippet(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value)))


def _relevant_excerpt(text: str, focus: str, limit: int = 1800) -> str:
    paragraphs = [
        re.sub(r"\s+", " ", item).strip()
        for item in text.split("\n")
        if item.strip()
    ]
    keywords = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{4,}", focus)
    }
    keywords.update({"predator", "predators", "preyed", "predation", "enemy"})
    selected = [
        paragraph
        for paragraph in paragraphs
        if any(keyword in paragraph.casefold() for keyword in keywords)
    ]
    selected.sort(
        key=lambda paragraph: (
            (
                0
                if any(
                    marker in paragraph.casefold()
                    for marker in (
                        "preyed on",
                        "natural predators",
                        "predators include",
                    )
                )
                else 1
                if "predation" in paragraph.casefold()
                else 2
            ),
            paragraphs.index(paragraph),
        )
    )
    content = "\n".join(selected[:4] or paragraphs[:3])
    return content[:limit]


def search_wikipedia(
    search_query: str,
    focus: str = "",
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        search_response = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": max(5, top_k),
                "format": "json",
                "formatversion": 2,
            },
            headers=HEADERS,
            timeout=(8, 25),
        )
        search_response.raise_for_status()
        search_rows = search_response.json().get("query", {}).get(
            "search", []
        )
        page_ids = [str(row["pageid"]) for row in search_rows[:top_k]]
        if not page_ids:
            return [], ["Wikipedia API returned no search result."]
        page_response = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "prop": "extracts|info",
                "pageids": "|".join(page_ids),
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
            },
            headers=HEADERS,
            timeout=(8, 25),
        )
        page_response.raise_for_status()
        pages = page_response.json().get("query", {}).get("pages", [])
    except Exception as error:
        return [], [
            f"Wikipedia API failed: {type(error).__name__}: {error}"
        ]

    rows = []
    snippets = {
        str(row["pageid"]): _clean_snippet(row.get("snippet", ""))
        for row in search_rows
    }
    for rank, page in enumerate(pages, start=1):
        page_id = str(page.get("pageid", ""))
        content = _relevant_excerpt(
            page.get("extract", "") or snippets.get(page_id, ""),
            focus or search_query,
        )
        if not content:
            continue
        title = page.get("title", "")
        source_page = page.get("fullurl", "")
        digest = hashlib.sha256(
            f"{page_id}|{content}".encode("utf-8")
        ).hexdigest()[:12]
        rows.append(
            {
                "id": f"web_wikipedia_{digest}",
                "content": content,
                "semantic_similarity": max(0.1, 1.0 - rank * 0.1),
                "final_score": max(0.1, 1.0 - rank * 0.1),
                "metadata": {
                    "source_type": "web_knowledge",
                    "retrieval_source": "wikipedia_api",
                    "title": title,
                    "source_page": source_page,
                    "search_query": search_query,
                    "rank": rank,
                },
            }
        )
    return rows, [] if rows else ["Wikipedia pages contained no usable text."]
