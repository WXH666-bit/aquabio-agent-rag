from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import requests

from .config import Settings


class OpenRouterClient:
    def __init__(self, settings: Settings, offline: bool = False):
        self.settings = settings
        self.offline = offline

    @property
    def enabled(self) -> bool:
        return not self.offline and bool(
            self.settings.api_key and "replace_with" not in self.settings.api_key
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.provider == "openrouter":
            headers["HTTP-Referer"] = self.settings.site_url
            headers["X-Title"] = self.settings.app_name
        return headers

    @staticmethod
    def _timeout() -> tuple[float, float]:
        return (
            float(os.getenv("AQUABIO_LLM_CONNECT_TIMEOUT", "10")),
            float(os.getenv("AQUABIO_LLM_READ_TIMEOUT", "45")),
        )

    @staticmethod
    def _choice_text(choice: dict[str, Any]) -> str:
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content or "")

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return True
        if re.search(r"\[E(?:\d+)?$", stripped):
            return True
        if stripped.count("[") != stripped.count("]"):
            return True
        return False

    @staticmethod
    def _provider_error(response: requests.Response) -> RuntimeError:
        try:
            error = response.json().get("error", {})
            code = (
                error.get("code")
                or error.get("type")
                or response.status_code
            )
            message = error.get("message") or response.text
        except (ValueError, AttributeError):
            code = response.status_code
            message = response.text
        return RuntimeError(
            f"LLM API request failed ({code}): {str(message)[:500]}"
        )

    def chat(
        self,
        messages: list[dict],
        response_schema: dict | None = None,
        max_tokens: int = 1600,
        max_continuations: int = 2,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if self.settings.provider == "openrouter":
            payload["reasoning"] = {"effort": "low", "exclude": True}
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "aquabio_response", "strict": True, "schema": response_schema},
            }
        response = requests.post(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self._timeout(),
        )
        if (
            response_schema
            and response.status_code in {400, 404, 422}
            and self.settings.provider == "openrouter"
        ):
            payload.pop("response_format", None)
            messages[-1]["content"] = [
                {
                    "type": "text",
                    "text": (
                        "Return only valid JSON matching this schema: "
                        + json.dumps(response_schema, ensure_ascii=False)
                    ),
                },
                *(
                    messages[-1]["content"]
                    if isinstance(messages[-1]["content"], list)
                    else [{"type": "text", "text": messages[-1]["content"]}]
                ),
            ]
            response = requests.post(
                f"{self.settings.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout(),
            )
        if not response.ok:
            raise self._provider_error(response)
        choice = response.json()["choices"][0]
        answer = self._choice_text(choice)
        if response_schema:
            return answer

        continuations = 0
        finish_reason = choice.get("finish_reason", "")
        while continuations < max_continuations and (
            finish_reason == "length" or self._looks_truncated(answer)
        ):
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "上一次生成不完整。请从头重新生成一份简洁但完整的最终答案，"
                        "不要提及续写或截断；确保每个关键结论都有完整的[E数字]引用。"
                    ),
                },
            ]
            continuation_response = requests.post(
                f"{self.settings.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.settings.model,
                    "messages": retry_messages,
                    "temperature": 0.1,
                    "max_tokens": max(max_tokens, 3200),
                    **(
                        {"reasoning": {"effort": "low", "exclude": True}}
                        if self.settings.provider == "openrouter"
                        else {}
                    ),
                },
                timeout=self._timeout(),
            )
            continuation_response.raise_for_status()
            choice = continuation_response.json()["choices"][0]
            replacement = self._choice_text(choice).strip()
            if not replacement:
                break
            answer = replacement
            finish_reason = choice.get("finish_reason", "")
            continuations += 1
        return answer

    def select_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 500,
    ) -> list[dict[str, Any]]:
        """Use the provider's native function-calling response."""
        if not self.enabled:
            return []
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self._timeout(),
        )
        if not response.ok:
            raise self._provider_error(response)
        message = response.json()["choices"][0].get("message", {})
        calls = []
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                {
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                }
            )
        return calls

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any] | None:
        text = (raw or "").strip()
        if not text:
            return None

        candidates = [text]
        candidates.extend(
            re.findall(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
        )

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _normalize_image_analysis(
        value: dict[str, Any],
        fallback_description: str = "",
    ) -> dict[str, Any]:
        def string_list(key: str) -> list[str]:
            raw_value = value.get(key, [])
            if isinstance(raw_value, str):
                return [raw_value] if raw_value.strip() else []
            if isinstance(raw_value, list):
                return [
                    str(item).strip()
                    for item in raw_value
                    if str(item).strip()
                ]
            return []

        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        description = str(
            value.get("description")
            or value.get("detailed_description")
            or value.get("summary")
            or fallback_description
        ).strip()
        return {
            "description": description,
            "possible_species": string_list("possible_species"),
            "visible_features": string_list("visible_features"),
            "degradation": string_list("degradation"),
            "confidence": min(1.0, max(0.0, confidence)),
            "limitations": string_list("limitations"),
        }

    def analyze_image(self, image_path: str | Path, prompt: str) -> dict:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "possible_species": {"type": "array", "items": {"type": "string"}},
                "visible_features": {"type": "array", "items": {"type": "string"}},
                "degradation": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "description",
                "possible_species",
                "visible_features",
                "degradation",
                "confidence",
                "limitations",
            ],
            "additionalProperties": False,
        }
        content = [
            {
                "type": "text",
                "text": (
                    prompt
                    + "\nReturn exactly one JSON object. Do not use Markdown "
                    "code fences or explanatory text."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
        raw = self.chat([{"role": "user", "content": content}], response_schema=schema)
        parsed = self._extract_json_object(raw)
        if parsed is not None:
            analysis = self._normalize_image_analysis(parsed)
            if analysis["description"]:
                analysis["structured_output"] = True
                return analysis

        if raw.strip():
            return {
                "description": raw.strip(),
                "possible_species": [],
                "visible_features": [],
                "degradation": [],
                "confidence": 0.0,
                "limitations": [
                    "视觉模型未返回约定 JSON，已将普通文本响应作为图片 caption。"
                ],
                "structured_output": False,
            }
        raise ValueError("Vision model returned an empty response.")
