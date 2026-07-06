from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests

from .config import GeminiSettings


class GeminiVisionClient:
    def __init__(self, settings: GeminiSettings, offline: bool = False):
        self.settings = settings
        self.offline = offline

    @property
    def enabled(self) -> bool:
        return not self.offline and bool(
            self.settings.api_key
            and "replace_with" not in self.settings.api_key
        )

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def analyze_image(self, image_path: str | Path, prompt: str) -> dict:
        if not self.enabled:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        schema = {
            "type": "OBJECT",
            "properties": {
                "description": {"type": "STRING"},
                "possible_species": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "visible_features": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "degradation": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "confidence": {"type": "NUMBER"},
                "limitations": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "required": [
                "description",
                "possible_species",
                "visible_features",
                "degradation",
                "confidence",
                "limitations",
            ],
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"{prompt}\n"
                                "Return the requested analysis as valid JSON."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        headers = {
            "x-goog-api-key": self.settings.api_key,
            "Content-Type": "application/json",
        }
        models = [self.settings.model]
        if (
            self.settings.fallback_model
            and self.settings.fallback_model != self.settings.model
        ):
            models.append(self.settings.fallback_model)

        response = None
        for model in models:
            url = (
                f"{self.settings.base_url.rstrip('/')}/models/"
                f"{model}:generateContent"
            )
            for attempt in range(3):
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if response.status_code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    break
                if attempt < 2:
                    time.sleep(2**attempt)
            if response.ok:
                break

        if response is None:
            raise RuntimeError("Gemini vision request was not sent.")
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini vision model returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts") or []
        value = next(
            (part.get("text") for part in parts if isinstance(part, dict)),
            None,
        )
        if not value:
            raise ValueError("Gemini vision model returned an empty response.")

        import json

        parsed = json.loads(value)
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        return {
            "description": str(parsed.get("description", "")).strip(),
            "possible_species": self._string_list(
                parsed.get("possible_species")
            ),
            "visible_features": self._string_list(
                parsed.get("visible_features")
            ),
            "degradation": self._string_list(parsed.get("degradation")),
            "confidence": min(1.0, max(0.0, confidence)),
            "limitations": self._string_list(parsed.get("limitations")),
            "structured_output": True,
        }
