"""End-to-end pipeline orchestrator for the Helipad Vision Engine.

This script runs the complete Modul D pipeline in order:

1.  Dataset setup (validate + synthetic generation + data.yaml)
2.  Training (YOLOv8 fine-tuning)
3.  Detection demo (single-frame inference + spatial mapping)
4.  Evaluation (precision / recall / mAP + latency profiling)
5.  Robustness testing (environmental perturbations)

Usage::

    python run_pipeline.py            # full pipeline
    python run_pipeline.py --skip-train   # skip training (use existing weights)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config import CONFIG, setup_logging

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run the full Modul D pipeline."""
    parser = argparse.ArgumentParser(
        description="Cargo Drone Helipad Vision Engine — Pipeline Runner"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training step (use existing best.pt if available)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip robustness testing step",
    )
    args = parser.parse_args()

    setup_logging()
    _LOGGER.info("=" * 70)
    _LOGGER.info("CARGO DRONE HELIPAD VISION ENGINE — PIPELINE START")
    _LOGGER.info("=" * 70)

    # --- Step 1: Dataset Setup ---
    _LOGGER.info("[STEP 1/5] Dataset Setup")
    from core_ai.dataset_setup import DatasetSetup
    ds = DatasetSetup()
    data_yaml = ds.ensure_ready()
    _LOGGER.info("  data.yaml: %s", data_yaml)

    # --- Step 2: Training ---
    if args.skip_train:
        _LOGGER.info("[STEP 2/5] Training — SKIPPED by user")
    else:
        _LOGGER.info("[STEP 2/5] Training")
        from core_ai.train import HelipadTrainer
        trainer = HelipadTrainer()
        trainer.load_model()
        trainer.train()

    # --- Step 3: Detection Demo ---
    _LOGGER.info("[STEP 3/5] Detection Demo")
    from core_ai.detector import HelipadDetector
    import cv2
    test_images = sorted(
        (ds.root_dir / ds.splits[2] / ds.images_folder).glob("*.jpg")
    )
    if test_images:
        detector = HelipadDetector()
        detector.load()
        frame = cv2.imread(str(test_images[0]))
        if frame is not None:
            result = detector.detect_and_map_spatial(frame)
            _LOGGER.info("  Detections: %d, Latency: %.2f ms",
                         len(result["detections"]),
                         result["inference_latency_ms"])
            annotated = detector.draw_detections(frame, result)
            out_path = CONFIG.resolve_path("logs/pipeline_detection_demo.jpg")
            cv2.imwrite(str(out_path), annotated)
            _LOGGER.info("  Annotated output: %s", out_path)

    # --- Step 4: Evaluation ---
    if args.skip_eval:
        _LOGGER.info("[STEP 4/5] Evaluation — SKIPPED by user")
    else:
        _LOGGER.info("[STEP 4/5] Evaluation")
        from core_ai.evaluator import VisionEvaluator
        evaluator = VisionEvaluator()
        try:
            eval_results = evaluator.evaluate_all()
            _LOGGER.info("  Evaluation complete.")
        except Exception as exc:
            _LOGGER.warning("  Evaluation failed: %s", exc)

    # --- Step 5: Robustness ---
    if args.skip_robustness:
        _LOGGER.info("[STEP 5/5] Robustness Testing — SKIPPED by user")
    else:
        _LOGGER.info("[STEP 5/5] Robustness Testing")
        from core_ai.robustness import RobustnessTester
        tester = RobustnessTester()
        try:
            tester.run_all()
        except Exception as exc:
            _LOGGER.warning("  Robustness testing failed: %s", exc)

    _LOGGER.info("=" * 70)
    _LOGGER.info("PIPELINE COMPLETE")
    _LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
