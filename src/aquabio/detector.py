from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class YOLOv8Detector:
    def __init__(
        self,
        model_variant: str = "yolov8n",
        weights_path: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ):
        self.model_variant = model_variant
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        self._device = device or ("cuda" if self._cuda_available() else "cpu")
        self._weights_path = weights_path
        self._class_names: list[str] = []

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLOv8 detection. "
                "Install with: pip install ultralytics"
            ) from exc
        if self._weights_path and Path(self._weights_path).exists():
            self._model = YOLO(self._weights_path)
        else:
            self._model = YOLO(f"{self.model_variant}.pt")
        self._class_names = list(getattr(self._model, "names", {}).values())
        return self._model

    def detect(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> dict[str, Any]:
        model = self._load_model()
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold
        if isinstance(image, (str, Path)):
            image_input = str(image)
        else:
            image_input = image
        results = model(
            image_input,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        if not results:
            return self._empty_result()
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._empty_result()
        bboxes = boxes.xyxy.cpu().numpy().tolist()
        scores = boxes.conf.cpu().numpy().tolist()
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        names = getattr(result, "names", {})
        detections = []
        for bbox, score, cls_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = bbox
            detections.append({
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                "confidence": round(float(score), 4),
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), str(cls_id)),
            })
        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_shape": (
                list(result.orig_shape) if result.orig_shape is not None else []
            ),
        }

    def detect_batch(
        self,
        images: list[str | Path | np.ndarray],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        model = self._load_model()
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold
        paths = [str(img) if isinstance(img, (str, Path)) else img for img in images]
        results = model(
            paths,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        output = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                output.append(self._empty_result())
                continue
            bboxes = boxes.xyxy.cpu().numpy().tolist()
            scores = boxes.conf.cpu().numpy().tolist()
            class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            names = getattr(result, "names", {})
            detections = []
            for bbox, score, cls_id in zip(bboxes, scores, class_ids):
                x1, y1, x2, y2 = bbox
                detections.append({
                    "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                    "confidence": round(float(score), 4),
                    "class_id": int(cls_id),
                    "class_name": names.get(int(cls_id), str(cls_id)),
                })
            output.append({
                "detections": detections,
                "num_detections": len(detections),
                "image_shape": (
                    list(result.orig_shape) if result.orig_shape is not None else []
                ),
            })
        return output

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "detections": [],
            "num_detections": 0,
            "image_shape": [],
        }


class AgentDetectionOrchestrator:
    def __init__(
        self,
        detector: YOLOv8Detector,
        enhancement_methods: list[str] | None = None,
    ):
        self.detector = detector
        self.enhancement_methods = enhancement_methods or [
            "original",
            "white_balance",
            "clahe",
            "white_balance_clahe",
            "gamma",
        ]

    def analyze_and_detect(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        from .image_tools import analyze_quality, create_enhancements

        image_path = Path(image_path)
        quality = analyze_quality(image_path)
        original_result = self.detector.detect(image_path)
        enhanced_results = {}
        if output_dir and len(self.enhancement_methods) > 1:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            enhancements = create_enhancements(image_path, output)
            for enh in enhancements:
                method = enh["method"]
                if method not in self.enhancement_methods:
                    continue
                det = self.detector.detect(enh["path"])
                enhanced_results[method] = {
                    "detection": det,
                    "quality": enh["quality"],
                    "enhanced_path": enh["path"],
                }
        best_method = self._select_best_method(original_result, enhanced_results)
        return {
            "original_quality": quality,
            "original_detection": original_result,
            "enhanced_results": enhanced_results,
            "best_method": best_method,
            "recommendation": self._build_recommendation(
                quality, original_result, enhanced_results, best_method
            ),
        }

    def _select_best_method(
        self,
        original: dict[str, Any],
        enhanced: dict[str, Any],
    ) -> str:
        best_score = 0.0
        best_method = "original"
        if original["detections"]:
            best_score = max(
                d["confidence"] for d in original["detections"]
            )
        for method, data in enhanced.items():
            det = data["detection"]
            if det["detections"]:
                top_conf = max(d["confidence"] for d in det["detections"])
                if top_conf > best_score:
                    best_score = top_conf
                    best_method = method
        return best_method

    def _build_recommendation(
        self,
        quality: dict[str, Any],
        original: dict[str, Any],
        enhanced: dict[str, Any],
        best_method: str,
    ) -> str:
        problems = quality.get("problems", [])
        if not problems:
            return "Image quality is good, recommend using original for detection."
        parts = [f"Detected image problems: {', '.join(problems)}. "]
        if best_method == "original":
            parts.append("Enhancement did not improve detection confidence, recommend original.")
        else:
            method_labels = {
                "white_balance": "White Balance",
                "clahe": "CLAHE",
                "white_balance_clahe": "WB+CLAHE",
                "gamma": "Gamma",
            }
            label = method_labels.get(best_method, best_method)
            parts.append(f"Recommend {label} enhancement for best detection confidence.")
        return "".join(parts)