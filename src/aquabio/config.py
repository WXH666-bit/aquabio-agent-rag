from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "nex-agi/nex-n2-pro:free"
    site_url: str = "http://localhost:8501"
    app_name: str = "AquaBio-AgentRAG"
    provider: str = "openrouter"

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        provider = os.getenv(
            "AQUABIO_LLM_PROVIDER", cls.provider
        ).lower()
        if provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            key_file = os.getenv("QWEN_KEY_FILE", "")
            if not api_key and key_file:
                path = Path(key_file)
                if path.is_file():
                    match = re.search(
                        r"api\s*key\s*[：:]\s*(sk-[A-Za-z0-9_-]+)",
                        path.read_text(encoding="utf-8", errors="ignore"),
                        flags=re.IGNORECASE,
                    )
                    if match:
                        api_key = match.group(1)
            return cls(
                api_key=api_key,
                base_url=os.getenv(
                    "QWEN_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                model=os.getenv(
                    "QWEN_MODEL", "qwen3.7-plus"
                ),
                site_url="",
                app_name=cls.app_name,
                provider=provider,
            )
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL", cls.base_url),
            model=os.getenv("OPENROUTER_MODEL", cls.model),
            site_url=os.getenv("OPENROUTER_SITE_URL", cls.site_url),
            app_name=os.getenv("OPENROUTER_APP_NAME", cls.app_name),
            provider=provider,
        )


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str = ""
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.5-flash-lite"

    @classmethod
    def from_env(cls) -> "GeminiSettings":
        load_env()
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url=os.getenv("GEMINI_BASE_URL", cls.base_url),
            model=os.getenv("GEMINI_VISION_MODEL", cls.model),
            fallback_model=os.getenv(
                "GEMINI_VISION_FALLBACK_MODEL",
                cls.fallback_model,
            ),
        )
