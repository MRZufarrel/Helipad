"""YOLOv8 training wrapper for helipad detection.

Wraps the Ultralytics YOLOv8 training pipeline with configurable
hyperparameters loaded from the central ``config.yaml``.  All training
artefacts (best weights, metrics logs) are written to predictable
locations for downstream modules to consume.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config import CONFIG

_LOGGER = logging.getLogger(__name__)


class ModelWeightsNotFoundError(Exception):
    """Raised when pre-trained or trained weight files are missing."""


class TrainingError(Exception):
    """Raised when the YOLOv8 training process fails."""


class HelipadTrainer:
    """Encapsulates the YOLOv8 training lifecycle for helipad detection.

    This class loads pre-trained ``yolov8n.pt`` weights, launches
    training with hyperparameters from configuration, and verifies that
    the best weights are produced.

    Attributes:
        data_yaml: Absolute path to the dataset ``data.yaml``.
        pretrained_weights: Path to the pre-trained weights file.
        best_weights_path: Expected path for the trained best weights.
    """

    def __init__(self) -> None:
        model_cfg = CONFIG.section("model")
        train_cfg = CONFIG.section("training")
        ds_cfg = CONFIG.section("dataset")

        self.data_yaml: Path = CONFIG.resolve_path(ds_cfg["data_yaml_path"])
        self.pretrained_weights: str = model_cfg["pretrained_weights"]
        self.best_weights_path: Path = CONFIG.resolve_path(
            model_cfg["best_weights_path"]
        )
        self.last_weights_path: Path = CONFIG.resolve_path(
            model_cfg["last_weights_path"]
        )

        # Hyperparameters
        self.epochs: int = train_cfg["epochs"]
        self.image_size: int = train_cfg["image_size"]
        self.batch_size: int = train_cfg["batch_size"]
        self.patience: int = train_cfg["patience"]
        self.optimizer: str = train_cfg["optimizer"]
        self.learning_rate: float = train_cfg["learning_rate"]
        self.weight_decay: float = train_cfg["weight_decay"]
        self.momentum: float = train_cfg["momentum"]
        self.warmup_epochs: int = train_cfg["warmup_epochs"]
        self.cos_lr: bool = train_cfg["cos_lr"]
        self.project_name: str = train_cfg["project_name"]
        self.experiment_name: str = train_cfg["experiment_name"]
        self.run_task: str = train_cfg.get("run_task", "detect")

        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_model(self) -> Any:
        """Load the pre-trained YOLOv8 nano model.

        Returns:
            The Ultralytics YOLO model instance.

        Raises:
            ModelWeightsNotFoundError: If the pre-trained weights file
                cannot be found.
            TrainingError: If the model cannot be initialised.
        """
        from ultralytics import YOLO

        weights_path = Path(self.pretrained_weights)
        if not weights_path.exists():
            _LOGGER.warning(
                "Pre-trained weights '%s' not found locally — Ultralytics "
                "will attempt auto-download.",
                self.pretrained_weights,
            )

        try:
            self._model = YOLO(self.pretrained_weights)
            _LOGGER.info(
                "Pre-trained model loaded: %s", self.pretrained_weights
            )
        except Exception as exc:
            raise TrainingError(
                f"Failed to load pre-trained model '{self.pretrained_weights}': {exc}"
            ) from exc
        return self._model

    def train(self) -> Dict[str, Any]:
        """Run YOLOv8 training with configured hyperparameters.

        Returns:
            Dictionary of training results/metrics returned by
            Ultralytics.

        Raises:
            TrainingError: If training fails or best weights are not
                produced.
            ModelWeightsNotFoundError: If the data.yaml is missing.
        """
        if self._model is None:
            self.load_model()

        if not self.data_yaml.exists():
            raise ModelWeightsNotFoundError(
                f"data.yaml not found at {self.data_yaml}. "
                "Run dataset_setup first."
            )

        _LOGGER.info(
            "Starting training: epochs=%d, imgsz=%d, batch=%d, optimizer=%s",
            self.epochs,
            self.image_size,
            self.batch_size,
            self.optimizer,
        )

        try:
            results = self._model.train(
                data=str(self.data_yaml),
                epochs=self.epochs,
                imgsz=self.image_size,
                batch=self.batch_size,
                patience=self.patience,
                optimizer=self.optimizer,
                lr0=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                warmup_epochs=self.warmup_epochs,
                cos_lr=self.cos_lr,
                project=self.project_name,
                name=self.experiment_name,
                exist_ok=True,
            )
        except Exception as exc:
            _LOGGER.error("Training failed: %s", exc, exc_info=True)
            raise TrainingError(f"YOLOv8 training failed: {exc}") from exc

        # Verify best weights. Ultralytics resolves the run directory as
        # <project>/<task>/<name>, so probe the documented path first and
        # then the task-qualified location.
        if not self.best_weights_path.exists():
            candidates = [
                CONFIG.resolve_path(
                    f"{self.project_name}/{self.run_task}/"
                    f"{self.experiment_name}/weights/best.pt"
                ),
                CONFIG.resolve_path(
                    f"{self.project_name}/{self.experiment_name}/weights/best.pt"
                ),
            ]
            alt_path = next((p for p in candidates if p.exists()), None)
            if alt_path is not None:
                self.best_weights_path = alt_path
            else:
                raise TrainingError(
                    "Training completed but best weights not found at "
                    f"{self.best_weights_path}, "
                    + " or ".join(str(p) for p in candidates)
                    + "."
                )

        _LOGGER.info(
            "Training complete. Best weights: %s", self.best_weights_path
        )
        self._log_metrics(results)
        return results

    def get_best_weights_path(self) -> Path:
        """Return the resolved path to the best trained weights.

        Returns:
            Absolute path to ``best.pt``.

        Raises:
            ModelWeightsNotFoundError: If the file does not exist.
        """
        if not self.best_weights_path.exists():
            raise ModelWeightsNotFoundError(
                f"Best weights not found: {self.best_weights_path}. "
                "Has training been run?"
            )
        return self.best_weights_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_metrics(self, results: Any) -> None:
        """Log key training metrics from the Ultralytics results object.

        Args:
            results: Results object returned by ``model.train()``.
        """
        try:
            results_dict = results.results_dict if hasattr(results, "results_dict") else {}
        except Exception:
            results_dict = {}

        if results_dict:
            for key, value in results_dict.items():
                _LOGGER.info("  Metric — %s: %.4f", key, float(value))
        else:
            _LOGGER.info("Training results object returned (no metrics dict available).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry: run full training pipeline."""
    from config.config import setup_logging
    setup_logging()

    trainer = HelipadTrainer()
    trainer.load_model()
    trainer.train()
    _LOGGER.info("Best weights: %s", trainer.get_best_weights_path())


if __name__ == "__main__":
    main()
