from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz

from .config import BOOKS, RAGAnythingPaths


SA_BOOK_ID = "sa_invertebrates"
KENYA_BOOK_ID = "living_guide"
SA_TABLE_PAGE_INDEXES = range(26, 39)
SA_PAGE_OFFSET = 3
TABLE_COLUMNS = (
    ("class", 55.0, 105.0),
    ("order", 105.0, 153.0),
    ("family", 153.0, 204.0),
    ("genus", 204.0, 268.0),
    ("species", 268.0, 318.0),
    ("common_name", 318.0, 382.0),
    ("authority", 382.0, 441.0),
    ("fb_code", 441.0, 475.0),
)
FIELD_LABELS = (
    ("distinguishing_features", r"distinguishing features"),
    ("colour", r"colou?r"),
    ("size", r"size"),
    ("distribution", r"distribution"),
    ("similar_species", r"similar species"),
    ("references", r"references?"),
)
TAXONOMY_LABELS = (
    "phylum",
    "subphylum",
    "class",
    "subclass",
    "order",
    "suborder",
    "infraorder",
    "superfamily",
    "family",
    "subfamily",
    "genus",
    "genera",
    "species",
    "common name",
    "common",
    "alternate",
)


@dataclass
class TaxaCatalogRow:
    catalog_id: str
    source_file: str
    table_pdf_page: int
    printed_page: int
    expected_pdf_page: int
    taxon_level: str
    class_name: str = ""
    order: str = ""
    family: str = ""
    genus: str = ""
    species: str = ""
    scientific_name: str = ""
    common_name: str = ""
    authority: str = ""
    fb_code: str = ""
    phylum: str = ""


@dataclass
class SpeciesPageUnit:
    unit_id: str
    doc_id: str
    source_file: str
    pdf_page: int
    printed_page: int | None
    title: str
    fb_code: str
    scientific_name: str
    common_name: str
    taxon_level: str
    taxonomy: dict[str, str] = field(default_factory=dict)
    distinguishing_features: str = ""
    colour: str = ""
    size: str = ""
    distribution: str = ""
    similar_species: str = ""
    references: str = ""
    image_count: int = 0
    image_records: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    extraction_warnings: list[str] = field(default_factory=list)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized:
        return normalized[:80]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _text_spans(page: fitz.Page) -> list[dict[str, Any]]:
    spans = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = _clean(span.get("text", ""))
                if not text:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                spans.append(
                    {
                        "text": text,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "cy": (y0 + y1) / 2,
                        "size": span.get("size", 0),
                        "font": span.get("font", ""),
                    }
                )
    return spans


def parse_sa_taxa_catalog(document: fitz.Document, source_file: str) -> list[TaxaCatalogRow]:
    rows: list[TaxaCatalogRow] = []
    for page_index in SA_TABLE_PAGE_INDEXES:
        spans = _text_spans(document[page_index])
        anchors = sorted(
            (
                (span["cy"], int(span["text"]))
                for span in spans
                if span["x0"] >= 475
                and re.fullmatch(r"\d{1,3}", span["text"])
                # Identification entries begin at printed page 41. Smaller
                # right-margin values are alternating page footers.
                and 41 <= int(span["text"]) <= document.page_count
            ),
            key=lambda item: item[0],
        )
        for index, (center_y, printed_page) in enumerate(anchors):
            lower = (
                (anchors[index - 1][0] + center_y) / 2
                if index
                else center_y - 9
            )
            upper = (
                (center_y + anchors[index + 1][0]) / 2
                if index + 1 < len(anchors)
                else center_y + 9
            )
            values: dict[str, str] = {}
            for name, x_start, x_end in TABLE_COLUMNS:
                parts = sorted(
                    (
                        (span["y0"], span["x0"], span["text"])
                        for span in spans
                        if lower <= span["cy"] < upper
                        and x_start - 1 <= span["x0"] < x_end
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                values[name] = _clean(" ".join(item[2] for item in parts))

            genus = values["genus"]
            species = values["species"]
            scientific_name = _clean(f"{genus} {species}")
            taxon_level = "species"
            if not species or re.search(r"\bspp?\.?\b", species, re.I):
                taxon_level = "group"
            elif not genus:
                taxon_level = "higher_taxon"
            fb_code = values["fb_code"] or f"page_{printed_page}"
            rows.append(
                TaxaCatalogRow(
                    catalog_id=f"catalog_{_slug(fb_code)}_{printed_page}",
                    source_file=source_file,
                    table_pdf_page=page_index + 1,
                    printed_page=printed_page,
                    expected_pdf_page=printed_page + SA_PAGE_OFFSET,
                    taxon_level=taxon_level,
                    class_name=values["class"],
                    order=values["order"],
                    family=values["family"],
                    genus=genus,
                    species=species,
                    scientific_name=scientific_name,
                    common_name=values["common_name"],
                    authority=values["authority"],
                    fb_code=fb_code,
                )
            )
    return rows


def _extract_labeled_sections(text: str) -> dict[str, str]:
    matches: list[tuple[int, int, str]] = []
    for field_name, pattern in FIELD_LABELS:
        match = re.search(rf"(?im)^\s*{pattern}\s*$", text)
        if match:
            matches.append((match.start(), match.end(), field_name))
    matches.sort()
    result = {name: "" for name, _ in FIELD_LABELS}
    for index, (_, end, field_name) in enumerate(matches):
        next_start = (
            matches[index + 1][0] if index + 1 < len(matches) else len(text)
        )
        value = text[end:next_start]
        if field_name == "references":
            title_match = re.search(
                r"(?m)^.{2,140}\([A-Za-z0-9 -]{3,16}\)\s*$\nPhylum\s*:",
                value,
                re.I,
            )
            if title_match:
                value = value[: title_match.start()]
        result[field_name] = _clean(value)
    return result


def _extract_title(text: str) -> tuple[str, str]:
    matches = list(
        re.finditer(
            r"(?m)^(.{2,140})\(([A-Za-z0-9 -]{3,16})\)\s*$\nPhylum\s*:",
            text,
            re.I,
        )
    )
    if not matches:
        return "", ""
    match = matches[-1]
    return _clean(match.group(1)), _clean(match.group(2))


def _extract_taxonomy(text: str) -> dict[str, str]:
    title_match = list(
        re.finditer(
            r"(?m)^.{2,140}\([A-Za-z0-9 -]{3,16}\)\s*$\nPhylum\s*:",
            text,
            re.I,
        )
    )
    if not title_match:
        return {}
    tail = text[title_match[-1].end() - len("Phylum:") :]
    label_pattern = "|".join(
        sorted((re.escape(label) for label in TAXONOMY_LABELS), key=len, reverse=True)
    )
    matches = list(
        re.finditer(rf"(?im)^\s*(?P<label>{label_pattern})\s*:\s*$", tail)
    )
    taxonomy: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tail)
        value_lines = []
        for line in tail[match.end() : end].splitlines():
            value = line.strip()
            if not value:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*mm)?", value, re.I):
                break
            value_lines.append(value)
        if not value_lines:
            continue
        taxonomy[match.group("label").casefold()] = _clean(
            " ".join(value_lines)
        )
    if "genera" in taxonomy and "genus" not in taxonomy:
        taxonomy["genus"] = taxonomy["genera"]
    if "common" in taxonomy and "common name" not in taxonomy:
        taxonomy["common name"] = taxonomy["common"]
    return taxonomy


def _printed_page(text: str) -> int | None:
    for match in re.finditer(r"(?m)^\s*(\d{1,3})\s*$", text):
        value = int(match.group(1))
        if 1 <= value <= 500:
            return value
    return None


def _page_images(
    page: fitz.Page,
    unit_id: str,
    output_dir: Path,
    extract_images: bool,
    min_dimension: int,
) -> list[dict[str, Any]]:
    records = []
    for index, info in enumerate(page.get_image_info(xrefs=True), start=1):
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        if min(width, height) < min_dimension:
            continue
        bbox = [round(float(value), 2) for value in info.get("bbox", ())]
        record: dict[str, Any] = {
            "image_id": f"{unit_id}_img_{index:02d}",
            "xref": int(info.get("xref", 0)),
            "width": width,
            "height": height,
            "bbox": bbox,
            "digest": (
                info.get("digest", b"").hex()
                if isinstance(info.get("digest"), bytes)
                else str(info.get("digest", ""))
            ),
            "extracted_path": "",
        }
        if extract_images and record["xref"]:
            output_dir.mkdir(parents=True, exist_ok=True)
            image = page.parent.extract_image(record["xref"])
            target = output_dir / f"{record['image_id']}.{image['ext']}"
            if not target.exists():
                target.write_bytes(image["image"])
            record["extracted_path"] = str(target)
        records.append(record)
    return records


def parse_sa_species_units(
    document: fitz.Document,
    source_file: str,
    output_dir: Path,
    extract_images: bool = False,
    min_image_dimension: int = 128,
) -> list[SpeciesPageUnit]:
    units = []
    for page_index in range(39, document.page_count):
        page = document[page_index]
        text = page.get_text("text").replace("\x00", " ")
        if (
            page_index + 1 == 385
            and
            "Ornithoteuthis antillarum" in text
            and "Ornithoteuthis volatilis" in text
        ):
            unit_id = "sa_taxon_ornithoteuthis_pair_p0385"
            images = _page_images(
                page,
                unit_id,
                output_dir / "images",
                extract_images,
                min_image_dimension,
            )
            units.append(
                SpeciesPageUnit(
                    unit_id=unit_id,
                    doc_id=f"doc_{SA_BOOK_ID}_p{page_index + 1:04d}",
                    source_file=source_file,
                    pdf_page=page_index + 1,
                    printed_page=382,
                    title=(
                        "Ornithoteuthis antillarum and "
                        "Ornithoteuthis volatilis"
                    ),
                    fb_code="OrnAnt+OrnVol",
                    scientific_name=(
                        "Ornithoteuthis antillarum; "
                        "Ornithoteuthis volatilis"
                    ),
                    common_name=(
                        "Atlantic bird squid; Shiny bird squid"
                    ),
                    taxon_level="species_pair",
                    taxonomy={
                        "phylum": "Mollusca",
                        "class": "Cephalopoda",
                        "order": "Oegopsida",
                        "family": "Ommastrephidae",
                        "genus": "Ornithoteuthis",
                        "species": "antillarum; volatilis",
                        "common name": (
                            "Atlantic bird squid; Shiny bird squid"
                        ),
                    },
                    distinguishing_features=_clean(text),
                    distribution=_clean(
                        "O. antillarum: North Atlantic south to at least "
                        "28 degrees S off Namibia. O. volatilis: Indo-West "
                        "Pacific to east Africa and the Benguela region."
                    ),
                    similar_species=(
                        "The two species are separated by the number of "
                        "hectocotylus depression columns and pits."
                    ),
                    image_count=len(images),
                    image_records=images,
                    raw_text=_clean(text),
                )
            )
            continue
        if not (
            re.search(r"(?im)^distinguishing features\s*$", text)
            and re.search(r"(?im)^Phylum\s*:", text)
        ):
            continue
        if re.search(r"(?im)^\s*xxx\s*$", text):
            continue
        title, fb_code = _extract_title(text)
        taxonomy = _extract_taxonomy(text)
        if "phylum" not in taxonomy:
            header = re.search(r"(?im)^Phylum:\s*([^\n]+)", text)
            if header:
                taxonomy["phylum"] = _clean(header.group(1))
        sections = _extract_labeled_sections(text)
        printed_page = _printed_page(text)
        scientific_name = _clean(
            f"{taxonomy.get('genus', '')} {taxonomy.get('species', '')}"
        )
        common_name = taxonomy.get("common name", "")
        warnings = []
        if not title:
            title = scientific_name or common_name or f"PDF page {page_index + 1}"
            warnings.append("title_not_matched")
        if not fb_code:
            fb_code = f"page_{printed_page or page_index + 1}"
            warnings.append("fb_code_not_matched")
        taxon_level = (
            "group"
            if re.search(r"\bspp?\.?\b", taxonomy.get("species", ""), re.I)
            else "species"
        )
        unit_id = f"sa_taxon_{_slug(fb_code)}_p{page_index + 1:04d}"
        image_records = _page_images(
            page,
            unit_id,
            output_dir / "images",
            extract_images,
            min_image_dimension,
        )
        units.append(
            SpeciesPageUnit(
                unit_id=unit_id,
                doc_id=f"doc_{SA_BOOK_ID}_p{page_index + 1:04d}",
                source_file=source_file,
                pdf_page=page_index + 1,
                printed_page=printed_page,
                title=title,
                fb_code=fb_code,
                scientific_name=scientific_name,
                common_name=common_name,
                taxon_level=taxon_level,
                taxonomy=taxonomy,
                image_count=len(image_records),
                image_records=image_records,
                raw_text=_clean(text),
                extraction_warnings=warnings,
                **sections,
            )
        )
    return units


def _join_catalog_and_units(
    catalog: list[TaxaCatalogRow],
    units: list[SpeciesPageUnit],
) -> list[dict[str, Any]]:
    by_page = {unit.printed_page: unit for unit in units if unit.printed_page}
    by_code = {unit.fb_code.casefold(): unit for unit in units if unit.fb_code}
    records = []
    for row in catalog:
        unit = by_page.get(row.printed_page) or by_code.get(row.fb_code.casefold())
        data = asdict(row)
        if unit:
            data["phylum"] = unit.taxonomy.get("phylum", "")
            data["page_unit_id"] = unit.unit_id
            data["actual_pdf_page"] = unit.pdf_page
            data["page_matched"] = True
            if unit.scientific_name:
                data["scientific_name"] = unit.scientific_name
            if unit.common_name:
                data["common_name"] = unit.common_name
            data["taxon_level"] = unit.taxon_level
        else:
            data["page_unit_id"] = ""
            data["actual_pdf_page"] = None
            data["page_matched"] = False
        records.append(data)
    return records


def _enrich_units_from_catalog(
    catalog: list[TaxaCatalogRow],
    units: list[SpeciesPageUnit],
) -> None:
    by_page = {row.printed_page: row for row in catalog}
    by_code = {row.fb_code.casefold(): row for row in catalog if row.fb_code}
    for unit in units:
        row = by_page.get(unit.printed_page or -1) or by_code.get(
            unit.fb_code.casefold()
        )
        if row is None:
            continue
        if row.fb_code and unit.fb_code.startswith("page_"):
            unit.fb_code = row.fb_code
        if row.scientific_name and not unit.scientific_name:
            unit.scientific_name = row.scientific_name
        if row.common_name:
            unit.common_name = row.common_name
            unit.taxonomy["common name"] = row.common_name
        if not unit.title or unit.title.startswith("PDF page"):
            unit.title = (
                unit.scientific_name or unit.common_name or unit.title
            )
        for key, value in (
            ("class", row.class_name),
            ("order", row.order),
            ("family", row.family),
            ("genus", row.genus),
            ("species", row.species),
        ):
            if value and not unit.taxonomy.get(key):
                unit.taxonomy[key] = value
        unit.taxon_level = row.taxon_level


def _species_chunks(units: list[SpeciesPageUnit]) -> list[dict[str, Any]]:
    chunks = []
    fields = (
        "distinguishing_features",
        "colour",
        "size",
        "distribution",
        "similar_species",
        "references",
    )
    for unit in units:
        base = {
            "doc_id": unit.doc_id,
            "unit_id": unit.unit_id,
            "source_file": unit.source_file,
            "page": unit.pdf_page,
            "printed_page": unit.printed_page,
            "title": unit.title,
            "scientific_name": unit.scientific_name,
            "common_name": unit.common_name,
            "fb_code": unit.fb_code,
            "phylum": unit.taxonomy.get("phylum", ""),
            "modality": "text",
        }
        taxonomy_text = "; ".join(
            f"{key}: {value}" for key, value in unit.taxonomy.items() if value
        )
        chunks.append(
            {
                **base,
                "chunk_id": f"{unit.unit_id}_taxonomy",
                "chunk_type": "taxonomy",
                "content": taxonomy_text,
            }
        )
        for field_name in fields:
            value = getattr(unit, field_name)
            if not value:
                continue
            chunks.append(
                {
                    **base,
                    "chunk_id": f"{unit.unit_id}_{field_name}",
                    "chunk_type": field_name,
                    "content": value,
                }
            )
        for image in unit.image_records:
            chunks.append(
                {
                    **base,
                    "chunk_id": image["image_id"],
                    "chunk_type": "pdf_image",
                    "modality": "image",
                    "content": (
                        f"Image on the identification page for {unit.title}. "
                        f"Dimensions {image['width']}x{image['height']}."
                    ),
                    "image": image,
                }
            )
    return chunks


def _relation_triples(units: list[SpeciesPageUnit]) -> list[dict[str, Any]]:
    triples = []
    seen = set()

    def add(unit: SpeciesPageUnit, source: str, relation: str, target: str) -> None:
        source = _clean(source)
        target = _clean(target)
        if not source or not target:
            return
        key = (source.casefold(), relation, target.casefold(), unit.pdf_page)
        if key in seen:
            return
        seen.add(key)
        triples.append(
            {
                "triple_id": "rel_"
                + hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16],
                "source": source,
                "relation": relation,
                "target": target,
                "doc_id": unit.doc_id,
                "page": unit.pdf_page,
                "unit_id": unit.unit_id,
                "evidence": (
                    unit.distinguishing_features
                    if relation == "has_feature"
                    else getattr(unit, relation.removesuffix("_in"), "")
                ),
            }
        )

    for unit in units:
        taxon = unit.scientific_name or unit.title
        for rank in ("genus", "family", "order", "class", "phylum"):
            add(unit, taxon, "is_a", unit.taxonomy.get(rank, ""))
        add(unit, taxon, "has_common_name", unit.common_name)
        add(unit, taxon, "has_feature", unit.distinguishing_features)
        add(unit, taxon, "distributed_in", unit.distribution)
        add(unit, taxon, "similar_to", unit.similar_species)
        add(unit, taxon, "described_in", f"{unit.source_file} page {unit.pdf_page}")
        for image in unit.image_records:
            add(unit, image["image_id"], "illustrates", taxon)
            add(unit, image["image_id"], "belongs_to", unit.unit_id)
    return triples


def _raganything_content_items(
    units: list[SpeciesPageUnit],
) -> list[dict[str, Any]]:
    items = []
    for chunk in _species_chunks(units):
        if chunk["modality"] != "text" or not chunk["content"]:
            continue
        items.append(
            {
                "type": "text",
                "text": (
                    f"[DOC_ID={chunk['doc_id']}]"
                    f"[SOURCE={chunk['source_file']}]"
                    f"[PAGE={chunk['page']}]"
                    f"[UNIT_ID={chunk['unit_id']}]"
                    f"[CHUNK_TYPE={chunk['chunk_type']}]\n"
                    f"Taxon: {chunk['title']}\n"
                    f"{chunk['content']}"
                ),
                "page_idx": chunk["page"] - 1,
                "unit_id": chunk["unit_id"],
                "doc_id": chunk["doc_id"],
                "chunk_type": chunk["chunk_type"],
            }
        )
    return items


def _book_sections_from_units(units: list[SpeciesPageUnit]) -> list[dict[str, Any]]:
    sections = [
        {
            "section_id": "sa_front_matter",
            "title": "Front matter and identification instructions",
            "start_pdf_page": 1,
            "end_pdf_page": 25,
            "section_type": "front_matter",
        },
        {
            "section_id": "sa_table_of_taxa",
            "title": "Table of Taxa in Field Guide",
            "start_pdf_page": 26,
            "end_pdf_page": 39,
            "section_type": "taxa_catalog",
        },
    ]
    current: dict[str, Any] | None = None
    for unit in sorted(units, key=lambda item: item.pdf_page):
        phylum = unit.taxonomy.get("phylum", "") or "Unclassified"
        if current and current["title"].casefold() == phylum.casefold():
            current["end_pdf_page"] = unit.pdf_page
            current["taxon_units"] += 1
            continue
        current = {
            "section_id": f"sa_phylum_{_slug(phylum)}",
            "title": phylum,
            "start_pdf_page": unit.pdf_page,
            "end_pdf_page": unit.pdf_page,
            "section_type": "phylum_identification_pages",
            "taxon_units": 1,
        }
        sections.append(current)
    return sections


def _parse_kenya_toc(document: fitz.Document) -> list[dict[str, Any]]:
    rows = []
    for page_index in range(8, min(13, document.page_count)):
        lines = document[page_index].get_text("text").splitlines()
        pending = ""
        for raw_line in lines:
            line = _clean(re.sub(r"\.{2,}", " ", raw_line))
            if not line:
                continue
            inline = re.match(r"^(.*?\D)\s*(\d{1,3})$", line)
            if inline:
                candidate = _clean(inline.group(1))
                number = inline.group(2)
            elif re.fullmatch(r"\d{1,3}", line) and pending:
                candidate = pending
                number = line
            else:
                letters = [char for char in line if char.isalpha()]
                uppercase_ratio = (
                    sum(char.isupper() for char in letters) / len(letters)
                    if letters
                    else 0
                )
                pending = line if uppercase_ratio >= 0.8 else ""
                continue
            letters = [char for char in candidate if char.isalpha()]
            uppercase_ratio = (
                sum(char.isupper() for char in letters) / len(letters)
                if letters
                else 0
            )
            if uppercase_ratio < 0.8:
                pending = ""
                continue
            title = _clean(candidate)
            printed_page = int(number)
            if title in {"TABLE OF CONTENTS"}:
                pending = ""
                continue
            rows.append(
                {
                    "section_id": f"kenya_{_slug(title)}_{printed_page}",
                    "title": title,
                    "printed_page": printed_page,
                    "expected_pdf_page": printed_page + 12,
                    "toc_pdf_page": page_index + 1,
                    "section_type": "resource_group",
                }
            )
            pending = ""
    deduped = {}
    for row in rows:
        deduped[(row["title"], row["printed_page"])] = row
    result = sorted(deduped.values(), key=lambda row: row["printed_page"])
    for index, row in enumerate(result):
        row["end_printed_page"] = (
            result[index + 1]["printed_page"] - 1
            if index + 1 < len(result)
            else document.page_count - 12
        )
    return result


def _kenya_page_units(
    document: fitz.Document, source_file: str, sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    units = []
    for page_index in range(document.page_count):
        text = _clean(document[page_index].get_text("text").replace("\x00", " "))
        if not text:
            continue
        printed_page = page_index - 11 if page_index >= 12 else None
        section = next(
            (
                row
                for row in reversed(sections)
                if printed_page is not None
                and printed_page >= row["printed_page"]
            ),
            None,
        )
        units.append(
            {
                "unit_id": f"kenya_page_{page_index + 1:04d}",
                "doc_id": f"doc_{KENYA_BOOK_ID}_p{page_index + 1:04d}",
                "source_file": source_file,
                "pdf_page": page_index + 1,
                "printed_page": printed_page,
                "section_id": section["section_id"] if section else "",
                "section_title": section["title"] if section else "Front matter",
                "modality": "text",
                "content": text,
                "image_count": len(document[page_index].get_images(full=True)),
            }
        )
    return units


def build_book_native(
    paths: RAGAnythingPaths,
    book_id: str = "all",
    extract_images: bool = False,
    min_image_dimension: int = 128,
) -> dict[str, Any]:
    paths.ensure()
    selected = BOOKS if book_id == "all" else {book_id: BOOKS[book_id]}
    summary: dict[str, Any] = {"books": {}, "extract_images": extract_images}
    for current_book_id, filename in selected.items():
        source = paths.pdf_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        output_dir = paths.book_native_dir / current_book_id
        output_dir.mkdir(parents=True, exist_ok=True)
        document = fitz.open(source)
        try:
            if current_book_id == SA_BOOK_ID:
                catalog = parse_sa_taxa_catalog(document, filename)
                units = parse_sa_species_units(
                    document,
                    filename,
                    output_dir,
                    extract_images=extract_images,
                    min_image_dimension=min_image_dimension,
                )
                _enrich_units_from_catalog(catalog, units)
                catalog_records = _join_catalog_and_units(catalog, units)
                chunks = _species_chunks(units)
                relations = _relation_triples(units)
                sections = _book_sections_from_units(units)
                _write_jsonl(
                    output_dir / "book_taxa_catalog.jsonl", catalog_records
                )
                _write_jsonl(
                    output_dir / "species_page_units.jsonl",
                    (asdict(unit) for unit in units),
                )
                _write_jsonl(output_dir / "rag_chunks.jsonl", chunks)
                _write_jsonl(
                    output_dir / "raganything_content_list.jsonl",
                    _raganything_content_items(units),
                )
                _write_jsonl(
                    output_dir / "relation_triples.jsonl", relations
                )
                _write_jsonl(output_dir / "book_sections.jsonl", sections)
                report = {
                    "book_id": current_book_id,
                    "source_file": filename,
                    "pdf_pages": document.page_count,
                    "catalog_rows": len(catalog),
                    "species_page_units": len(units),
                    "catalog_page_matches": sum(
                        bool(row["page_matched"]) for row in catalog_records
                    ),
                    "chunks": len(chunks),
                    "relations": len(relations),
                    "image_records": sum(unit.image_count for unit in units),
                    "units_with_warnings": sum(
                        bool(unit.extraction_warnings) for unit in units
                    ),
                    "phyla": dict(
                        Counter(
                            unit.taxonomy.get("phylum", "Unclassified")
                            for unit in units
                        )
                    ),
                }
            else:
                sections = _parse_kenya_toc(document)
                page_units = _kenya_page_units(document, filename, sections)
                _write_jsonl(output_dir / "book_sections.jsonl", sections)
                _write_jsonl(output_dir / "book_page_units.jsonl", page_units)
                _write_jsonl(output_dir / "rag_chunks.jsonl", page_units)
                _write_jsonl(
                    output_dir / "raganything_content_list.jsonl",
                    (
                        {
                            "type": "text",
                            "text": (
                                f"[DOC_ID={row['doc_id']}]"
                                f"[SOURCE={row['source_file']}]"
                                f"[PAGE={row['pdf_page']}]"
                                f"[UNIT_ID={row['unit_id']}]\n"
                                f"Section: {row['section_title']}\n"
                                f"{row['content']}"
                            ),
                            "page_idx": row["pdf_page"] - 1,
                            "unit_id": row["unit_id"],
                            "doc_id": row["doc_id"],
                            "chunk_type": "book_page",
                        }
                        for row in page_units
                    ),
                )
                report = {
                    "book_id": current_book_id,
                    "source_file": filename,
                    "pdf_pages": document.page_count,
                    "sections": len(sections),
                    "page_units": len(page_units),
                    "chunks": len(page_units),
                    "image_objects": sum(
                        row["image_count"] for row in page_units
                    ),
                }
            _write_json(output_dir / "extraction_report.json", report)
            summary["books"][current_book_id] = report
        finally:
            document.close()
    _write_json(paths.book_native_dir / "build_summary.json", summary)
    return summary
