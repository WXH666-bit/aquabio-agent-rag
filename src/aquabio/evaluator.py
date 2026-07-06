from __future__ import annotations

from pathlib import Path
from typing import Any

from .detector import YOLOv8Detector


class DetectionEvaluator:
    def __init__(self, detector: YOLOv8Detector):
        self.detector = detector

    def compare_enhanced_detection(
        self,
        original_path: str | Path,
        enhanced_paths: dict[str, str | Path],
    ) -> dict[str, Any]:
        original_path = Path(original_path)
        original_det = self.detector.detect(original_path)
        comparison: dict[str, Any] = {
            "original": {
                "path": str(original_path),
                "detection": original_det,
            },
            "enhanced": {},
        }
        for method, path in enhanced_paths.items():
            det = self.detector.detect(path)
            comparison["enhanced"][method] = {
                "path": str(path),
                "detection": det,
            }
        comparison["summary"] = self._summarize(comparison)
        return comparison

    def evaluate_dataset(
        self,
        image_dir: str | Path,
        label_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(
            p for p in image_dir.rglob("*") if p.suffix.lower() in image_suffixes
        )
        if not images:
            return {"error": "No images found", "total": 0, "results": []}
        results: list[dict[str, Any]] = []
        total_detections = 0
        confidence_sum = 0.0
        for img_path in images:
            det = self.detector.detect(img_path)
            num = det["num_detections"]
            total_detections += num
            if det["detections"]:
                confidence_sum += max(
                    d["confidence"] for d in det["detections"]
                )
            results.append({
                "image": str(img_path.relative_to(image_dir)),
                "num_detections": num,
                "detections": det["detections"],
            })
        avg_confidence = (
            confidence_sum / len(images) if images else 0.0
        )
        return {
            "total_images": len(images),
            "total_detections": total_detections,
            "avg_detections_per_image": round(
                total_detections / len(images), 2
            ),
            "avg_top_confidence": round(avg_confidence, 4),
            "results": results,
        }

    def evaluate_enhancement_impact(
        self,
        original_dir: str | Path,
        enhanced_dirs: dict[str, str | Path],
    ) -> dict[str, Any]:
        original_dir = Path(original_dir)
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(
            p for p in original_dir.rglob("*") if p.suffix.lower() in image_suffixes
        )
        if not images:
            return {"error": "No images found", "total": 0}
        baseline = self._batch_stats(images)
        enhanced_stats = {}
        for method, dir_path in enhanced_dirs.items():
            dir_path = Path(dir_path)
            if not dir_path.exists():
                enhanced_stats[method] = {"error": f"Directory not found: {dir_path}"}
                continue
            enhanced_images = sorted(
                p
                for p in dir_path.rglob("*")
                if p.suffix.lower() in image_suffixes
            )
            if not enhanced_images:
                enhanced_stats[method] = {"error": "No enhanced images found"}
                continue
            enhanced_stats[method] = self._batch_stats(enhanced_images)
        delta = {}
        for method, stats in enhanced_stats.items():
            if "error" in stats:
                continue
            delta[method] = {
                "detection_delta": round(
                    stats["avg_detections"] - baseline["avg_detections"], 2
                ),
                "confidence_delta": round(
                    stats["avg_top_confidence"] - baseline["avg_top_confidence"], 4
                ),
                "detection_rate_delta": round(
                    stats["detection_rate"] - baseline["detection_rate"], 4
                ),
            }
        return {
            "baseline": baseline,
            "enhanced": enhanced_stats,
            "delta": delta,
            "total_images": len(images),
        }

    def _batch_stats(self, images: list[Path]) -> dict[str, Any]:
        total_detections = 0
        confidence_sum = 0.0
        images_with_detections = 0
        for img_path in images:
            det = self.detector.detect(img_path)
            total_detections += det["num_detections"]
            if det["detections"]:
                confidence_sum += max(
                    d["confidence"] for d in det["detections"]
                )
                images_with_detections += 1
        return {
            "total_images": len(images),
            "total_detections": total_detections,
            "avg_detections": round(total_detections / len(images), 2),
            "avg_top_confidence": round(confidence_sum / len(images), 4),
            "detection_rate": round(images_with_detections / len(images), 4),
        }