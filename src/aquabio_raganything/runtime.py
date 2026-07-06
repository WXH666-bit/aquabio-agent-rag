from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .config import (
    ENTITY_GUIDANCE,
    ENTITY_TYPES,
    RAGAnythingPaths,
    RAGAnythingSettings,
)


class LocalBGEEmbedding:
    def __init__(self, settings: RAGAnythingSettings):
        self.settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {
                "local_files_only": self.settings.local_files_only,
            }
            if self.settings.model_cache:
                kwargs["cache_folder"] = self.settings.model_cache
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                **kwargs,
            )
        return self._model

    async def __call__(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return await asyncio.to_thread(
            model.encode,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def _gemini_request(
    settings: RAGAnythingSettings,
    prompt: str,
    image_data: str | None,
) -> str:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if image_data:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_data,
                }
            }
        )
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.1},
    }
    models = [settings.gemini_model, settings.gemini_fallback_model]
    last_response = None
    for model in dict.fromkeys(item for item in models if item):
        url = (
            f"{settings.gemini_base_url.rstrip('/')}/models/"
            f"{model}:generateContent"
        )
        for delay in (0, 2, 4):
            if delay:
                import time

                time.sleep(delay)
            last_response = requests.post(
                url,
                headers={
                    "x-goog-api-key": settings.gemini_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            if last_response.status_code not in {
                429,
                500,
                502,
                503,
                504,
            }:
                break
        if last_response.ok:
            body = last_response.json()
            return "".join(
                part.get("text", "")
                for part in body["candidates"][0]["content"]["parts"]
            )
    if last_response is None:
        raise RuntimeError("Gemini request was not sent.")
    last_response.raise_for_status()
    raise RuntimeError("Gemini returned no text.")


def _messages_to_prompt(messages: list[dict]) -> tuple[str, str | None]:
    prompt_parts = []
    image_data = None
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        for part in content or []:
            if part.get("type") == "text":
                prompt_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if ";base64," in url:
                    image_data = url.split(";base64,", 1)[1]
    return "\n".join(prompt_parts), image_data


def create_rag(
    paths: RAGAnythingPaths,
    settings: RAGAnythingSettings,
):
    scripts_dir = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if scripts_dir.lower() not in {item.lower() for item in path_entries}:
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get(
            "PATH", ""
        )
    try:
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig
    except ImportError as exc:
        raise RuntimeError(
            "RAG-Anything environment is not installed. Run this command with "
            ".venv-raganything after installing the local RAG-Anything source."
        ) from exc

    use_deepseek = settings.text_llm_provider == "deepseek"
    use_qwen = settings.text_llm_provider == "qwen"
    qwen_settings = None
    if use_qwen:
        from aquabio.config import Settings

        qwen_settings = Settings.from_env()
        if qwen_settings.provider != "qwen" or not qwen_settings.api_key:
            raise RuntimeError(
                "Qwen is selected for LightRAG queries, but no Qwen key "
                "was found."
            )
    if use_deepseek and not settings.deepseek_api_key:
        raise RuntimeError(
            "DeepSeek is selected for graph extraction, but no key was found."
        )
    if (
        not use_deepseek
        and not use_qwen
        and not settings.openrouter_api_key
    ):
        raise RuntimeError("OPENROUTER_API_KEY is required for graph extraction.")

    if use_qwen:
        text_model = qwen_settings.model
        text_api_key = qwen_settings.api_key
        text_base_url = qwen_settings.base_url
    elif use_deepseek:
        text_model = settings.deepseek_model
        text_api_key = settings.deepseek_api_key
        text_base_url = settings.deepseek_base_url
    else:
        text_model = settings.openrouter_model
        text_api_key = settings.openrouter_api_key
        text_base_url = settings.openrouter_base_url

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs,
    ):
        kwargs.pop("hashing_kv", None)
        keyword_extraction = bool(
            kwargs.pop("keyword_extraction", False)
        )
        return await openai_complete_if_cache(
            text_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=text_api_key,
            base_url=text_base_url,
            # DeepSeek currently rejects OpenAI's Pydantic response_format.
            # The LightRAG keyword prompt already requests JSON and repairs it
            # after the call, so plain chat completion is compatible.
            keyword_extraction=(
                keyword_extraction if not use_deepseek else False
            ),
            **kwargs,
        )

    async def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        image_data: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ):
        if not settings.gemini_api_key:
            return await llm_model_func(
                prompt,
                system_prompt,
                history_messages,
                **kwargs,
            )
        if messages:
            message_prompt, message_image = _messages_to_prompt(messages)
            prompt = message_prompt or prompt
            image_data = message_image or image_data
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        return await asyncio.to_thread(
            _gemini_request,
            settings,
            prompt,
            image_data,
        )

    embedding = LocalBGEEmbedding(settings)
    embedding_func = EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=8192,
        func=embedding,
    )
    config = RAGAnythingConfig(
        working_dir=str(paths.working_dir),
        parser=settings.parser,
        parse_method=settings.parse_method,
        parser_output_dir=str(paths.parser_output_dir),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=1,
        context_window=1,
        context_mode="page",
        max_context_tokens=2000,
        include_headers=True,
        include_captions=True,
        use_full_path=False,
    )
    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "kv_storage": "JsonKVStorage",
            "vector_storage": "NanoVectorDBStorage",
            "graph_storage": "NetworkXStorage",
            "doc_status_storage": "JsonDocStatusStorage",
            "llm_model_name": text_model,
            "llm_model_max_async": 1,
            "embedding_func_max_async": 1,
            "max_parallel_insert": 1,
            "enable_llm_cache": True,
            "addon_params": {
                "entity_types": ENTITY_TYPES,
                "entity_types_guidance": ENTITY_GUIDANCE,
            },
        },
    )


async def ensure_initialized(rag) -> None:
    result = await rag._ensure_lightrag_initialized()
    if not result or not result.get("success"):
        raise RuntimeError(
            f"LightRAG initialization failed: "
            f"{(result or {}).get('error', 'unknown error')}"
        )
