"""Minimal OpenRouter image-and-text question answering demo.

CMD example:
    .\.venv\Scripts\python.exe demo_multimodal_api.py ^
      --image data\samples\starfish_01.jpg ^
      --question "图中是什么水下生物？图像有什么质量问题？"
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

import requests


def load_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without an extra dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def ask_image(question: str, image_path: Path) -> str:
    load_env()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2-pro:free")

    if not api_key or "replace_with" in api_key:
        raise RuntimeError("请先在 .env 中配置 OPENROUTER_API_KEY。")
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片：{image_path}")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是水下图像分析助手。请基于图片回答问题；"
                    "不确定时明确说明，不要虚构物种、数量或检测框。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AquaBio Multimodal Demo",
    }

    print(f"正在调用模型：{model}", file=sys.stderr, flush=True)
    print("免费模型可能需要等待 20-180 秒。", file=sys.stderr, flush=True)

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter 请求失败：HTTP {response.status_code}\n{response.text[:1000]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenRouter 免费模型图文问答 Demo")
    parser.add_argument("--image", required=True, help="本地图片路径")
    parser.add_argument("--question", required=True, help="需要询问的问题")
    args = parser.parse_args()

    try:
        answer = ask_image(args.question, Path(args.image))
    except (OSError, RuntimeError, requests.RequestException) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    print("\n模型回答：\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
