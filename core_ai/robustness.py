"""Robustness testing under synthetic environmental perturbations.

:class:`RobustnessTester` applies four categories of degradation to test
images and evaluates detection performance under each:

1.  **Dim lighting** — brightness × 0.3
2.  **Glare / overexposure** — brightness × 1.7
3.  **Perspective rotation** — 15° to 45°
4.  **Motion blur** — Gaussian kernel 15×15

For each condition, the detector is run on every perturbed image and the
detection rate plus mean confidence are recorded.  A final summary
report is logged, analysing performance degradation and limitations.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config import CONFIG
from core_ai.detector import HelipadDetector

_LOGGER = logging.getLogger(__name__)


class RobustnessTestError(Exception):
    """Raised when robustness testing cannot proceed."""


class RobustnessTester:
    """Test helipad detector robustness under environmental perturbations.

    Attributes:
        detector: The :class:`HelipadDetector` instance to test.
        test_images_dir: Directory containing baseline test images.
        conditions: Ordered list of perturbation condition names.
    """

    def __init__(self, detector: Optional[HelipadDetector] = None) -> None:
        rb_cfg = CONFIG.section("robustness")
        ds_cfg = CONFIG.section("dataset")

        self.dim_factor: float = rb_cfg["dim_brightness_factor"]
        self.glare_factor: float = rb_cfg["glare_brightness_factor"]
        self.rot_min_deg: float = rb_cfg["rotation_min_deg"]
        self.rot_max_deg: float = rb_cfg["rotation_max_deg"]
        self.blur_kernel: int = rb_cfg["blur_kernel_size"]
        self.blur_sigma: float = rb_cfg["blur_sigma"]

        self.test_images_dir: Path = (
            CONFIG.resolve_path(ds_cfg["root_dir"])
            / ds_cfg["test_subdir"]
            / ds_cfg["images_folder"]
        )

        self.conditions: List[str] = [
            "baseline",
            "dim_lighting",
            "glare_overexposure",
            "perspective_rotation",
            "motion_blur",
        ]

        if detector is not None:
            self.detector: HelipadDetector = detector
        else:
            self.detector = HelipadDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_all(self, num_images: Optional[int] = None) -> Dict[str, Any]:
        """Run robustness evaluation across all perturbation conditions.

        Args:
            num_images: Maximum number of test images per condition.
                ``None`` means all available.

        Returns:
            Dictionary keyed by condition name, each containing
            ``detection_rate``, ``mean_confidence``, ``num_images``,
            and ``latency_ms``.

        Raises:
            RobustnessTestError: If no test images are found or the
                detector cannot be loaded.
        """
        self.detector.load()

        image_paths = sorted(self.test_images_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = sorted(self.test_images_dir.glob("*.png"))
        if not image_paths:
            raise RobustnessTestError(
                f"No test images found in {self.test_images_dir}"
            )

        if num_images is not None:
            image_paths = image_paths[:num_images]

        _LOGGER.info(
            "Robustness testing on %d images across %d conditions...",
            len(image_paths),
            len(self.conditions),
        )

        results: Dict[str, Any] = {}
        for condition in self.conditions:
            results[condition] = self._evaluate_condition(
                condition=condition,
                image_paths=image_paths,
            )

        self._log_summary(results)
        return results

    # ------------------------------------------------------------------
    # Perturbation methods
    # ------------------------------------------------------------------
    def apply_dim_lighting(self, frame: np.ndarray) -> np.ndarray:
        """Reduce brightness to simulate dim lighting conditions.

        Args:
            frame: BGR image array.

        Returns:
            Darkened BGR image.
        """
        return np.clip(frame.astype(np.float32) * self.dim_factor, 0, 255).astype(np.uint8)

    def apply_glare(self, frame: np.ndarray) -> np.ndarray:
        """Increase brightness to simulate glare / overexposure.

        Args:
            frame: BGR image array.

        Returns:
            Brightened BGR image.
        """
        return np.clip(frame.astype(np.float32) * self.glare_factor, 0, 255).astype(np.uint8)

    def apply_perspective_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Apply a random perspective rotation between configured angles.

        The rotation angle is randomly chosen between
        ``rot_min_deg`` and ``rot_max_deg``.

        Args:
            frame: BGR image array.

        Returns:
            Rotated BGR image (same dimensions, black fill).
        """
        angle = np.random.uniform(self.rot_min_deg, self.rot_max_deg)
        h, w = frame.shape[:2]
        centre = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(
            frame, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return rotated

    def apply_motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian motion blur with the configured kernel size.

        Args:
            frame: BGR image array.

        Returns:
            Blurred BGR image.
        """
        ksize = self.blur_kernel
        # Ensure odd kernel size
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(frame, (ksize, ksize), self.blur_sigma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_condition(
        self, condition: str, image_paths: List[Path]
    ) -> Dict[str, Any]:
        """Evaluate detector on all images under a single condition.

        Args:
            condition: Name of the perturbation condition.
            image_paths: List of test image paths.

        Returns:
            Dictionary with detection rate, mean confidence, and latency.
        """
        _LOGGER.info("  Condition: %s (%d images)", condition, len(image_paths))

        num_detected = 0
        confidences: List[float] = []
        latencies: List[float] = []

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                _LOGGER.warning("  Could not read %s — skipping.", img_path)
                continue

            perturbed = self._apply_perturbation(condition, frame)

            try:
                result = self.detector.detect_and_map_spatial(perturbed)
            except Exception as exc:
                _LOGGER.warning("  Detection failed on %s [%s]: %s",
                                img_path.name, condition, exc)
                continue

            dets = result["detections"]
            latencies.append(result["inference_latency_ms"])

            if len(dets) > 0:
                num_detected += 1
                confidences.extend(d["confidence"] for d in dets)

        total_processed = len(latencies)
        detection_rate = num_detected / total_processed if total_processed > 0 else 0.0
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        mean_latency = float(np.mean(latencies)) if latencies else 0.0

        return {
            "detection_rate": detection_rate,
            "mean_confidence": mean_conf,
            "num_images": total_processed,
            "num_detected": num_detected,
            "latency_ms": mean_latency,
        }

    def _apply_perturbation(self, condition: str, frame: np.ndarray) -> np.ndarray:
        """Dispatch to the correct perturbation method.

        Args:
            condition: Condition name.
            frame: Original BGR image.

        Returns:
            Perturbed BGR image.
        """
        if condition == "baseline":
            return frame
        elif condition == "dim_lighting":
            return self.apply_dim_lighting(frame)
        elif condition == "glare_overexposure":
            return self.apply_glare(frame)
        elif condition == "perspective_rotation":
            return self.apply_perspective_rotation(frame)
        elif condition == "motion_blur":
            return self.apply_motion_blur(frame)
        else:
            _LOGGER.warning("Unknown condition '%s' — returning original.", condition)
            return frame

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log a formatted summary table of robustness results.

        Also logs an analytical narrative of limitations.

        Args:
            results: Full results dictionary from :meth:`run_all`.
        """
        _LOGGER.info("=" * 70)
        _LOGGER.info("ROBUSTNESS TEST SUMMARY")
        _LOGGER.info("=" * 70)
        _LOGGER.info(
            "%-25s | %8s | %10s | %10s | %8s",
            "Condition", "Det.Rate", "Mean Conf", "Latency(ms)", "Images",
        )
        _LOGGER.info("-" * 70)

        baseline = results.get("baseline", {})
        baseline_rate = baseline.get("detection_rate", 0.0)
        baseline_conf = baseline.get("mean_confidence", 0.0)

        for condition in self.conditions:
            r = results.get(condition, {})
            _LOGGER.info(
                "%-25s | %8.1f%% | %10.4f | %10.2f | %8d",
                condition,
                r.get("detection_rate", 0.0) * 100,
                r.get("mean_confidence", 0.0),
                r.get("latency_ms", 0.0),
                r.get("num_images", 0),
            )

        _LOGGER.info("=" * 70)

        # Analytical narrative
        _LOGGER.info("ANALYSIS OF LIMITATIONS:")
        for condition in self.conditions[1:]:  # skip baseline
            r = results.get(condition, {})
            rate = r.get("detection_rate", 0.0)
            conf = r.get("mean_confidence", 0.0)
            rate_drop = baseline_rate - rate
            conf_drop = baseline_conf - conf

            if rate_drop > 0.3:
                severity = "SEVERE"
            elif rate_drop > 0.15:
                severity = "MODERATE"
            else:
                severity = "MINOR"

            _LOGGER.info(
                "  [%s] %-25s — detection rate drop: %.1f%%, "
                "confidence drop: %.4f",
                severity,
                condition,
                rate_drop * 100,
                conf_drop,
            )

        _LOGGER.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry: run robustness testing."""
    from config.config import setup_logging
    setup_logging()

    tester = RobustnessTester()
    results = tester.run_all()
    _LOGGER.info("Robustness testing complete.")


if __name__ == "__main__":
    main()
