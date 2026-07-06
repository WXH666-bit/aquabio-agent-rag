"""
可选：联网扩展脚本。

用途：
- 使用 Wikipedia REST API 获取页面摘要
- 使用 WoRMS REST API 获取分类学记录
- 把结果保存为 raw records，后续可用于替换/增强 seed 文本

注意：
- 该脚本需要本地网络环境。
- 不建议直接把外部网页大段复制为作业内容，应保存 source_url，并对文本做摘要化/结构化。
"""

import json
import time
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECIES_LIST = ROOT / "data" / "species_list.json"
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "AquaBio-MRAG/0.1 student project"}

def get_wikipedia_summary(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 200:
        return r.json()
    return {"error": r.status_code, "title": title}

def get_worms_records(name):
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{name}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 200:
        return r.json()
    return {"error": r.status_code, "name": name}

def main():
    species = json.loads(SPECIES_LIST.read_text(encoding="utf-8"))

    wiki_out = OUT_DIR / "wikipedia_records.jsonl"
    worms_out = OUT_DIR / "worms_records.jsonl"

    with wiki_out.open("w", encoding="utf-8") as fw, worms_out.open("w", encoding="utf-8") as ft:
        for item in species:
            wiki = get_wikipedia_summary(item["wiki_title"])
            wiki_record = {
                "species_id": item["species_id"],
                "wiki_title": item["wiki_title"],
                "record": wiki
            }
            fw.write(json.dumps(wiki_record, ensure_ascii=False) + "\n")

            worms = get_worms_records(item["worms_name"])
            worms_record = {
                "species_id": item["species_id"],
                "worms_name": item["worms_name"],
                "record": worms
            }
            ft.write(json.dumps(worms_record, ensure_ascii=False) + "\n")

            print("done", item["species_id"])
            time.sleep(1)

if __name__ == "__main__":
    main()
