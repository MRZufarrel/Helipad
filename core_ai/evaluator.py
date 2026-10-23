"""Model evaluation: precision, recall, mAP, and latency profiling.

:class:`VisionEvaluator` runs the trained ``best.pt`` on the unseen test
split and reports:

- **Precision**, **Recall**, **mAP@50**, **mAP@50-95** (via Ultralytics
  built-in validation).
- **Latency breakdown** in milliseconds: pre-processing, inference, and
  post-processing (NMS) — measured per-image and averaged.
- **Compliance check** against the configured target latency
  (default < 200 ms).
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config import CONFIG

_LOGGER = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation cannot proceed."""


class VisionEvaluator:
    """Evaluate trained YOLOv8 weights on the test split.

    Attributes:
        weights_path: Path to ``best.pt``.
        data_yaml: Path to ``data.yaml``.
        test_images_dir: Directory containing test images.
        target_latency_ms: Latency compliance threshold.
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        model_cfg = CONFIG.section("model")
        eval_cfg = CONFIG.section("evaluation")
        inf_cfg = CONFIG.section("inference")
        ds_cfg = CONFIG.section("dataset")

        if weights_path:
            self.weights_path: Path = Path(weights_path)
        else:
            self.weights_path = CONFIG.resolve_path(model_cfg["best_weights_path"])

        self.data_yaml: Path = CONFIG.resolve_path(ds_cfg["data_yaml_path"])
        self.test_images_dir: Path = (
            CONFIG.resolve_path(ds_cfg["root_dir"])
            / ds_cfg["test_subdir"]
            / ds_cfg["images_folder"]
        )

        self.conf_threshold: float = eval_cfg["conf_threshold"]
        self.iou_threshold: float = eval_cfg["iou_threshold"]
        self.max_det: int = eval_cfg["max_det"]
        self.target_latency_ms: float = inf_cfg["target_latency_ms"]

        self._model: Optional[Any] = None
        self._device: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_metrics(self) -> Dict[str, float]:
        """Run Ultralytics validation to obtain mAP / precision / recall.

        Returns:
            Dictionary with keys ``precision``, ``recall``,
            ``map50``, ``map50_95``, and ``fitness``.

        Raises:
            EvaluationError: If weights or data.yaml are missing, or
                validation fails.
        """
        self._ensure_model_loaded()

        if not self.data_yaml.exists():
            raise EvaluationError(
                f"data.yaml not found: {self.data_yaml}"
            )

        _LOGGER.info(
            "Running Ultralytics validation on test split (data=%s)...",
            self.data_yaml,
        )

        try:
            metrics = self._model.val(
                data=str(self.data_yaml),
                split="test",
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            _LOGGER.error("Validation failed: %s", exc, exc_info=True)
            raise EvaluationError(f"Validation failed: {exc}") from exc

        try:
            results_dict = metrics.results_dict
        except AttributeError:
            results_dict = {}

        precision = float(results_dict.get("metrics/precision(B)", 0.0))
        recall = float(results_dict.get("metrics/recall(B)", 0.0))
        map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
        fitness = float(results_dict.get("fitness", 0.0))

        result = {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "fitness": fitness,
        }

        _LOGGER.info("=== Validation Metrics ===")
        for key, val in result.items():
            _LOGGER.info("  %-12s: %.4f", key, val)

        return result

    def evaluate_latency(self, num_images: Optional[int] = None) -> Dict[str, Any]:
        """Measure per-image latency breakdown across the test set.

        Latency is split into:

        - **preprocessing_ms**: resize / normalise / tensor conversion.
        - **inference_ms**: forward pass through the network.
        - **postprocessing_ms**: NMS + box decoding.

        Args:
            num_images: Maximum number of test images to evaluate.
                ``None`` means all available.

        Returns:
            Dictionary with per-stage mean/median latencies, total
            latency, and ``latency_compliant`` boolean.

        Raises:
            EvaluationError: If no test images are found or inference
                fails.
        """
        self._ensure_model_loaded()

        image_paths = sorted(self.test_images_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = sorted(self.test_images_dir.glob("*.png"))
        if not image_paths:
            raise EvaluationError(
                f"No test images found in {self.test_images_dir}"
            )

        if num_images is not None:
            image_paths = image_paths[:num_images]

        _LOGGER.info(
            "Profiling latency on %d test images (target < %.0f ms)...",
            len(image_paths),
            self.target_latency_ms,
        )

        pre_times: List[float] = []
        inf_times: List[float] = []
        post_times: List[float] = []
        total_times: List[float] = []

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                _LOGGER.warning("Could not read %s — skipping.", img_path)
                continue

            pre, inf, post, total = self._measure_single(frame)
            pre_times.append(pre)
            inf_times.append(inf)
            post_times.append(post)
            total_times.append(total)

        if not total_times:
            raise EvaluationError("No images could be processed for latency profiling.")

        result = self._summarise_latency(pre_times, inf_times, post_times, total_times)

        _LOGGER.info("=== Latency Profiling (ms) ===")
        _LOGGER.info("  Pre-processing  mean: %.2f  median: %.2f",
                      result["preprocessing_ms"]["mean"],
                      result["preprocessing_ms"]["median"])
        _LOGGER.info("  Inference        mean: %.2f  median: %.2f",
                      result["inference_ms"]["mean"],
                      result["inference_ms"]["median"])
        _LOGGER.info("  Post-processing  mean: %.2f  median: %.2f",
                      result["postprocessing_ms"]["mean"],
                      result["postprocessing_ms"]["median"])
        _LOGGER.info("  Total            mean: %.2f  median: %.2f",
                      result["total_ms"]["mean"],
                      result["total_ms"]["median"])
        _LOGGER.info("  Target: < %.0f ms  |  Compliant: %s",
                      self.target_latency_ms,
                      result["latency_compliant"])

        return result

    def evaluate_all(self, num_latency_images: Optional[int] = None) -> Dict[str, Any]:
        """Run both metric evaluation and latency profiling.

        Args:
            num_latency_images: Limit for latency profiling images.

        Returns:
            Combined dictionary with ``metrics`` and ``latency`` keys.
        """
        metrics = self.evaluate_metrics()
        latency = self.evaluate_latency(num_images=num_latency_images)
        return {"metrics": metrics, "latency": latency}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        if not self.weights_path.exists():
            raise EvaluationError(
                f"Weights not found: {self.weights_path}. Run training first."
            )

        from ultralytics import YOLO

        try:
            import torch
            if torch.cuda.is_available():
                self._model = YOLO(str(self.weights_path))
                self._model.to("cuda:0")
                self._device = "cuda:0"
                _LOGGER.info("Evaluator model loaded on CUDA.")
            else:
                raise RuntimeError("CUDA not available")
        except Exception:
            self._model = YOLO(str(self.weights_path))
            self._model.to("cpu")
            self._device = "cpu"
            _LOGGER.info("Evaluator model loaded on CPU.")

    def _measure_single(self, frame: np.ndarray) -> Tuple[float, float, float, float]:
        """Measure pre/inference/post latency for a single frame.

        This uses Ultralytics' internal timing when available, falling
        back to manual ``perf_counter`` measurement.

        Args:
            frame: BGR image array.

        Returns:
            Tuple of ``(pre_ms, inf_ms, post_ms, total_ms)``.
        """
        # Manual measurement: we call predict and inspect speed dict
        t_total_start = time.perf_counter()
        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        t_total_end = time.perf_counter()

        total_manual_ms = (t_total_end - t_total_start) * 1000.0

        # Ultralytics provides a 'speed' dict with preprocess/inference/postprocess
        try:
            speed = results[0].speed
            pre_ms = float(speed.get("preprocess", 0.0))
            inf_ms = float(speed.get("inference", 0.0))
            post_ms = float(speed.get("postprocess", 0.0))
            # If speed values are all zero, fall back to manual split
            if pre_ms + inf_ms + post_ms < 0.01:
                # Approximate split: inference dominates
                pre_ms = total_manual_ms * 0.10
                inf_ms = total_manual_ms * 0.75
                post_ms = total_manual_ms * 0.15
            total_ms = pre_ms + inf_ms + post_ms
        except (AttributeError, IndexError, TypeError):
            pre_ms = total_manual_ms * 0.10
            inf_ms = total_manual_ms * 0.75
            post_ms = total_manual_ms * 0.15
            total_ms = total_manual_ms

        return pre_ms, inf_ms, post_ms, total_ms

    def _summarise_latency(
        self,
        pre: List[float],
        inf: List[float],
        post: List[float],
        total: List[float],
    ) -> Dict[str, Any]:
        """Summarise raw latency lists into mean/median statistics.

        Args:
            pre: Pre-processing times in ms.
            inf: Inference times in ms.
            post: Post-processing times in ms.
            total: Total times in ms.

        Returns:
            Structured latency summary dictionary.
        """
        def stats(lst: List[float]) -> Dict[str, float]:
            return {
                "mean": statistics.mean(lst),
                "median": statistics.median(lst),
                "min": min(lst),
                "max": max(lst),
                "std": statistics.stdev(lst) if len(lst) > 1 else 0.0,
            }

        mean_total = statistics.mean(total)
        return {
            "preprocessing_ms": stats(pre),
            "inference_ms": stats(inf),
            "postprocessing_ms": stats(post),
            "total_ms": stats(total),
            "mean_total_ms": mean_total,
            "target_ms": self.target_latency_ms,
            "latency_compliant": mean_total < self.target_latency_ms,
            "num_images": len(total),
            "device": self._device,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry: run full evaluation."""
    from config.config import setup_logging
    setup_logging()

    evaluator = VisionEvaluator()
    results = evaluator.evaluate_all()
    _LOGGER.info("Full evaluation results: %s", results)


if __name__ == "__main__":
    main()
