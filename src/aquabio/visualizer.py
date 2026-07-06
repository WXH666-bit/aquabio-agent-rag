from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLOR_PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]


def draw_detections(
    image: str | Path | np.ndarray,
    detections: list[dict[str, Any]],
    output_path: str | Path | None = None,
    line_thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Cannot read image: {image}")
    else:
        img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det.get("class_id", 0)
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        label_parts = [det.get("class_name", str(cls_id))]
        if show_confidence:
            label_parts.append(f"{det['confidence']:.2f}")
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
    return img


def draw_comparison(
    original_path: str | Path,
    enhanced_paths: dict[str, str | Path],
    detection_results: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
    max_cols: int = 3,
) -> np.ndarray:
    panels = []
    if "original" in detection_results:
        det = detection_results["original"].get("detection", {})
        panels.append(("Original", original_path, det.get("detections", [])))
    for method, path in enhanced_paths.items():
        det_data = detection_results.get("enhanced", {}).get(method, {})
        det = det_data.get("detection", {})
        method_labels = {
            "white_balance": "WB",
            "clahe": "CLAHE",
            "white_balance_clahe": "WB+CLAHE",
            "gamma": "Gamma",
        }
        label = method_labels.get(method, method)
        panels.append((label, path, det.get("detections", [])))
    if not panels:
        raise ValueError("No panels to render")
    rendered = []
    target_h = 320
    for label, path, dets in panels:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = draw_detections(img, dets, line_thickness=2, font_scale=0.5)
        h, w = img.shape[:2]
        scale = target_h / h
        img = cv2.resize(img, (int(w * scale), target_h))
        header = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(np.vstack([header, img]))
    if not rendered:
        raise ValueError("All images failed to load")
    max_w = max(p.shape[1] for p in rendered)
    padded = []
    for panel in rendered:
        h, w = panel.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
            panel = np.hstack([panel, pad])
        padded.append(panel)
    cols = min(max_cols, len(padded))
    rows_needed = (len(padded) + cols - 1) // cols
    grid_rows = []
    for row_idx in range(rows_needed):
        row_panels = padded[row_idx * cols : (row_idx + 1) * cols]
        while len(row_panels) < cols:
            row_panels.append(
                np.zeros_like(row_panels[0])
            )
        grid_rows.append(np.hstack(row_panels))
    grid = np.vstack(grid_rows)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), grid)
    return grid


def draw_detection_stats(
    comparison: dict[str, Any],
    output_path: str | Path | None = None,
) -> np.ndarray:
    width, height = 800, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(
        img,
        "Detection Comparison",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    original_det = comparison.get("original", {}).get("detection", {})
    original_count = original_det.get("num_detections", 0)
    original_conf = (
        max(d["confidence"] for d in original_det["detections"])
        if original_det.get("detections")
        else 0.0
    )
    entries = [("Original", original_count, original_conf)]
    for method, data in comparison.get("enhanced", {}).items():
        det = data.get("detection", {})
        count = det.get("num_detections", 0)
        conf = (
            max(d["confidence"] for d in det["detections"])
            if det.get("detections")
            else 0.0
        )
        entries.append((method, count, conf))
    bar_y = 70
    bar_height = 30
    max_count = max(e[1] for e in entries) if entries else 1
    max_count = max(max_count, 1)
    for idx, (label, count, conf) in enumerate(entries):
        bar_width = int((count / max_count) * 500) if max_count > 0 else 0
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        cv2.rectangle(img, (180, bar_y), (180 + bar_width, bar_y + bar_height), color, -1)
        cv2.putText(
            img,
            label,
            (20, bar_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{count} det / {conf:.2f} conf",
            (190 + bar_width, bar_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        bar_y += bar_height + 15
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
    return img
