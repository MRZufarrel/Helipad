"""Cargo Drone Helipad Vision Engine — core AI package.

Modules
-------
- :mod:`dataset_setup`  — dataset validation, ``data.yaml`` generation, synthetic data.
- :mod:`train`          — YOLOv8 training wrapper (``HelipadTrainer``).
- :mod:`detector`       — inference + spatial mapping (``HelipadDetector``).
- :mod:`evaluator`      — precision/recall/mAP + latency profiling (``VisionEvaluator``).
- :mod:`robustness`     — environmental perturbation testing (``RobustnessTester``).
"""

from __future__ import annotations

__all__ = [
    "dataset_setup",
    "train",
    "detector",
    "evaluator",
    "robustness",
]

__version__ = "1.0.0"
