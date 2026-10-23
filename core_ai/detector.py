"""Helipad detection and spatial mapping module.

Provides :class:`HelipadDetector`, which loads trained YOLOv8 weights
(with automatic CUDA→CPU fallback), runs inference on individual frames,
and maps pixel-space detections into real-world metric offsets using a
pinhole camera model.

The spatial mapping formula:

    delta_X = altitude * tan((e_x / W) * fov_x)
    delta_Y = altitude * tan((e_y / H) * fov_y)

where ``e_x``, ``e_y`` are pixel errors from the frame centre, ``W``,
``H`` are frame dimensions, and ``fov_x``, ``fov_y`` are the camera's
horizontal and vertical fields of view (in radians).
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config import CONFIG

_LOGGER = logging.getLogger(__name__)


class DetectorWeightsNotFoundError(Exception):
    """Raised when the trained ``best.pt`` weights file is missing."""


class InferenceError(Exception):
    """Raised when inference fails due to model or frame issues."""


class HelipadDetector:
    """YOLOv8-based helipad detector with spatial mapping.

    The detector loads the best trained weights, preferring CUDA and
    falling back to CPU if CUDA is unavailable or runs out of memory.
    Each call to :meth:`detect_and_map_spatial` returns a dictionary
    with bounding boxes, confidence scores, pixel centroids, metric
    offsets (ΔX, ΔY), and inference latency.

    Attributes:
        weights_path: Absolute path to ``best.pt``.
        confidence_threshold: Minimum confidence for a detection.
        iou_threshold: NMS IoU threshold.
        device: Resolved device string (``'cuda:0'`` or ``'cpu'``).
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        model_cfg = CONFIG.section("model")
        inf_cfg = CONFIG.section("inference")

        if weights_path:
            self.weights_path: Path = Path(weights_path)
        else:
            self.weights_path = CONFIG.resolve_path(model_cfg["best_weights_path"])

        self.confidence_threshold: float = model_cfg["confidence_threshold"]
        self.iou_threshold: float = model_cfg["iou_threshold"]
        self._device_preferred: str = model_cfg["device_preferred"]
        self._device_fallback: str = model_cfg["device_fallback"]

        self._default_altitude: float = inf_cfg["default_altitude_m"]
        self._default_fov_x_deg: float = inf_cfg["default_fov_x_deg"]
        self._default_fov_y_deg: float = inf_cfg["default_fov_y_deg"]

        self.device: str = ""
        self._model: Optional[Any] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the YOLOv8 model with CUDA→CPU fallback.

        Raises:
            DetectorWeightsNotFoundError: If ``best.pt`` does not exist.
            InferenceError: If the model cannot be loaded on any device.
        """
        if not self.weights_path.exists():
            raise DetectorWeightsNotFoundError(
                f"Trained weights not found: {self.weights_path}. "
                "Run training first."
            )

        from ultralytics import YOLO

        # Try CUDA first
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if cuda_available:
            try:
                self._model = YOLO(str(self.weights_path))
                self._model.to(self._device_preferred)
                self.device = self._device_preferred
                _LOGGER.info(
                    "Model loaded on CUDA: %s",
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU",
                )
                self._loaded = True
                return
            except Exception as exc:
                _LOGGER.warning(
                    "CUDA load failed (%s) — falling back to CPU.", exc
                )

        # CPU fallback
        try:
            self._model = YOLO(str(self.weights_path))
            self._model.to(self._device_fallback)
            self.device = self._device_fallback
            _LOGGER.info("Model loaded on CPU (fallback).")
            self._loaded = True
        except Exception as exc:
            raise InferenceError(
                f"Failed to load model on any device: {exc}"
            ) from exc

    def _ensure_loaded(self) -> None:
        """Lazy-load the model if not yet loaded."""
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Detection + spatial mapping
    # ------------------------------------------------------------------
    def detect_and_map_spatial(
        self,
        frame: np.ndarray,
        altitude_m: Optional[float] = None,
        fov_x: Optional[float] = None,
        fov_y: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run detection and compute metric spatial offsets.

        Args:
            frame: BGR image array (H, W, 3).
            altitude_m: Drone altitude above ground in metres.
                Defaults to the configured value.
            fov_x: Horizontal field of view in **degrees**.
                Defaults to the configured value.
            fov_y: Vertical field of view in **degrees**.
                Defaults to the configured value.

        Returns:
            Dictionary with keys:

            - ``detections``: List of per-detection dicts (bbox, confidence,
              centre_pixel, metric_offset).
            - ``frame_shape``: ``(height, width)``.
            - ``altitude_m``: Altitude used.
            - ``device``: Device string used for inference.
            - ``inference_latency_ms``: Total inference latency in ms.

        Raises:
            InferenceError: If the frame is invalid or inference fails.
        """
        self._ensure_loaded()

        if frame is None or frame.size == 0:
            raise InferenceError("Received empty or None frame.")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise InferenceError(
                f"Expected BGR frame of shape (H,W,3), got {frame.shape}"
            )

        altitude = altitude_m if altitude_m is not None else self._default_altitude
        fov_x_deg = fov_x if fov_x is not None else self._default_fov_x_deg
        fov_y_deg = fov_y if fov_y is not None else self._default_fov_y_deg
        fov_x_rad = math.radians(fov_x_deg)
        fov_y_rad = math.radians(fov_y_deg)

        h, w = frame.shape[:2]

        try:
            t_start = time.perf_counter()
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            t_end = time.perf_counter()
        except Exception as exc:
            _LOGGER.error("Inference failed: %s", exc, exc_info=True)
            raise InferenceError(f"Inference failed: {exc}") from exc

        latency_ms = (t_end - t_start) * 1000.0
        detections = self._parse_results(
            results=results,
            frame_w=w,
            frame_h=h,
            altitude=altitude,
            fov_x_rad=fov_x_rad,
            fov_y_rad=fov_y_rad,
        )

        _LOGGER.debug(
            "Detected %d helipad(s) in %.2f ms on %s",
            len(detections),
            latency_ms,
            self.device,
        )

        return {
            "detections": detections,
            "frame_shape": (h, w),
            "altitude_m": altitude,
            "device": self.device,
            "inference_latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def draw_detections(
        self, frame: np.ndarray, result: Dict[str, Any]
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of *frame*.

        Args:
            frame: Original BGR image.
            result: Output dictionary from :meth:`detect_and_map_spatial`.

        Returns:
            New image array with detections drawn.
        """
        annotated = frame.copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cx, cy = det["centre_pixel"]
            dx, dy = det["metric_offset"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"helipad {conf:.2f} dX={dx:.2f}m dY={dy:.2f}m"
            cv2.putText(
                annotated, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )
            # Crosshair at centre
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            # Frame centre
            h, w = frame.shape[:2]
            cv2.drawMarker(annotated, (w // 2, h // 2), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

        return annotated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_results(
        self,
        results: Any,
        frame_w: int,
        frame_h: int,
        altitude: float,
        fov_x_rad: float,
        fov_y_rad: float,
    ) -> List[Dict[str, Any]]:
        """Parse Ultralytics results into structured detection dicts.

        Args:
            results: Raw Ultralytics results list.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.
            altitude: Altitude in metres.
            fov_x_rad: Horizontal FOV in radians.
            fov_y_rad: Vertical FOV in radians.

        Returns:
            List of detection dictionaries.
        """
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
        except Exception as exc:
            _LOGGER.error("Failed to parse detection boxes: %s", exc)
            return detections

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            conf = float(confs[i])
            cls_id = int(clss[i])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Pixel error from frame centre
            e_x = cx - frame_w / 2
            e_y = cy - frame_h / 2

            # Pinhole model → metric offset
            delta_x = altitude * math.tan((e_x / frame_w) * fov_x_rad)
            delta_y = altitude * math.tan((e_y / frame_h) * fov_y_rad)

            detections.append({
                "class_id": cls_id,
                "class_name": "helipad",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "centre_pixel": (cx, cy),
                "pixel_error": (e_x, e_y),
                "metric_offset": (delta_x, delta_y),
            })

        return detections


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry: run detection on a single test image."""
    from config.config import setup_logging
    setup_logging()

    from core_ai.dataset_setup import DatasetSetup

    setup = DatasetSetup()
    test_images = sorted(
        (setup.root_dir / setup.splits[2] / setup.images_folder).glob("*.jpg")
    )
    if not test_images:
        _LOGGER.error("No test images found. Run dataset_setup first.")
        return

    frame = cv2.imread(str(test_images[0]))
    if frame is None:
        _LOGGER.error("Failed to read image: %s", test_images[0])
        return

    detector = HelipadDetector()
    detector.load()
    result = detector.detect_and_map_spatial(frame)
    _LOGGER.info("Result: %s", result)

    annotated = detector.draw_detections(frame, result)
    out_path = CONFIG.resolve_path("logs/test_detection_output.jpg")
    cv2.imwrite(str(out_path), annotated)
    _LOGGER.info("Annotated image saved to %s", out_path)


if __name__ == "__main__":
    main()
