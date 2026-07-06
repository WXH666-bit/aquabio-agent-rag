from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def analyze_quality(image_path: str | Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    means = image.mean(axis=(0, 1)) / 255.0
    dominant = ["blue", "green", "red"][int(np.argmax(means))]
    cast_strength = float(np.max(means) - np.min(means))
    return {
        "brightness": round(float(gray.mean() / 255.0), 4),
        "contrast": round(float(gray.std() / 128.0), 4),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "dominant_channel": dominant,
        "color_cast_score": round(cast_strength, 4),
        "problems": [
            name
            for name, condition in (
                ("low_light", gray.mean() / 255.0 < 0.3),
                ("low_contrast", gray.std() / 128.0 < 0.32),
                ("blur", cv2.Laplacian(gray, cv2.CV_64F).var() < 80),
                ("color_cast", cast_strength > 0.12),
            )
            if condition
        ],
    }


def _white_balance(image: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32)
    channel_means = result.mean(axis=(0, 1))
    gray_mean = channel_means.mean()
    result *= gray_mean / np.maximum(channel_means, 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def _clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def _gamma(image: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def create_enhancements(image_path: str | Path, output_dir: str | Path) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    candidates = {
        "white_balance": _white_balance(image),
        "clahe": _clahe(image),
        "white_balance_clahe": _clahe(_white_balance(image)),
        "gamma": _gamma(image),
    }
    results = []
    for method, candidate in candidates.items():
        path = output / f"{stem}_{method}.jpg"
        cv2.imwrite(str(path), candidate)
        results.append({"method": method, "path": str(path), "quality": analyze_quality(path)})
    return results

