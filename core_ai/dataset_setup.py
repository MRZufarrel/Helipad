"""Dataset setup, validation, and synthetic helipad generation.

This module is responsible for:

1.  Validating the YOLO directory structure (``images/`` + ``labels/``
    for each split).
2.  Auto-generating ``config/data.yaml`` with ``nc: 1`` and
    ``names: ['helipad']``.
3.  Producing a synthetic helipad dataset (random background + painted
    "H" shape) so the full pipeline can be exercised end-to-end without
    external data.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

# Ensure project root is on sys.path so ``config`` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config import CONFIG

_LOGGER = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when the dataset directory structure is invalid."""


class DatasetSetup:
    """Validate, configure, and (optionally) synthesise a YOLO dataset.

    Attributes:
        root_dir: Absolute path to the dataset root.
        splits: Ordered list of split names (``train``, ``val``, ``test``).
    """

    def __init__(self) -> None:
        ds_cfg = CONFIG.section("dataset")
        self.root_dir: Path = CONFIG.resolve_path(ds_cfg["root_dir"])
        self.splits: List[str] = [
            ds_cfg["train_subdir"],
            ds_cfg["val_subdir"],
            ds_cfg["test_subdir"],
        ]
        self.images_folder: str = ds_cfg["images_folder"]
        self.labels_folder: str = ds_cfg["labels_folder"]
        self.num_classes: int = ds_cfg["num_classes"]
        self.class_names: List[str] = ds_cfg["class_names"]
        self.data_yaml_path: Path = CONFIG.resolve_path(ds_cfg["data_yaml_path"])

        syn = ds_cfg["synthetic"]
        self._syn_image_size: int = syn["image_size"]
        self._syn_counts: Dict[str, int] = {
            ds_cfg["train_subdir"]: syn["num_train"],
            ds_cfg["val_subdir"]: syn["num_val"],
            ds_cfg["test_subdir"]: syn["num_test"],
        }
        self._syn_min_ratio: float = syn["helipad_min_size_ratio"]
        self._syn_max_ratio: float = syn["helipad_max_size_ratio"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_structure(self) -> bool:
        """Validate that every split has ``images/`` and ``labels/`` dirs.

        Returns:
            ``True`` if the structure is valid.

        Raises:
            DatasetValidationError: If any required directory is missing.
        """
        _LOGGER.info("Validating dataset structure at %s", self.root_dir)
        if not self.root_dir.exists():
            raise DatasetValidationError(
                f"Dataset root does not exist: {self.root_dir}"
            )

        for split in self.splits:
            split_dir = self.root_dir / split
            images_dir = split_dir / self.images_folder
            labels_dir = split_dir / self.labels_folder

            if not split_dir.exists():
                raise DatasetValidationError(
                    f"Split directory missing: {split_dir}"
                )
            if not images_dir.exists():
                raise DatasetValidationError(
                    f"Images directory missing: {images_dir}"
                )
            if not labels_dir.exists():
                raise DatasetValidationError(
                    f"Labels directory missing: {labels_dir}"
                )

            image_files = self._list_images(images_dir)
            if len(image_files) == 0:
                _LOGGER.warning(
                    "Split '%s' has 0 images in %s — synthetic generation "
                    "should be run.",
                    split,
                    images_dir,
                )
            _LOGGER.info(
                "Split '%s': %d images, labels dir OK.", split, len(image_files)
            )

        _LOGGER.info("Dataset structure validation passed.")
        return True

    def generate_data_yaml(self) -> Path:
        """Generate ``data.yaml`` for Ultralytics YOLOv8 training.

        The file is written to the path specified in configuration
        (``dataset.data_yaml_path``).

        Returns:
            Absolute path to the generated ``data.yaml``.
        """
        data = {
            "path": str(self.root_dir.resolve()),
            "train": f"{self.splits[0]}/{self.images_folder}",
            "val": f"{self.splits[1]}/{self.images_folder}",
            "test": f"{self.splits[2]}/{self.images_folder}",
            "nc": self.num_classes,
            "names": self.class_names,
        }

        self.data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_yaml_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)

        _LOGGER.info("Generated data.yaml at %s", self.data_yaml_path)
        return self.data_yaml_path

    def generate_synthetic_dataset(self) -> None:
        """Generate a synthetic helipad dataset if image dirs are empty.

        For each split, if the ``images/`` directory contains no files,
        ``num_*`` synthetic images are created.  Each image has a random
        background and a painted "H" helipad marker with a corresponding
        YOLO label file.
        """
        _LOGGER.info("Starting synthetic dataset generation...")
        for split in self.splits:
            images_dir = self.root_dir / split / self.images_folder
            labels_dir = self.root_dir / split / self.labels_folder
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            existing = self._list_images(images_dir)
            if len(existing) > 0:
                _LOGGER.info(
                    "Split '%s' already has %d images — skipping synthesis.",
                    split,
                    len(existing),
                )
                continue

            count = self._syn_counts.get(split, 0)
            _LOGGER.info(
                "Generating %d synthetic images for split '%s'...", count, split
            )
            for idx in range(count):
                self._generate_single_image(
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    filename=f"helipad_{split}_{idx:04d}",
                )
        _LOGGER.info("Synthetic dataset generation complete.")

    def ensure_ready(self) -> Path:
        """Convenience: validate, synthesise if needed, generate YAML.

        Returns:
            Path to the generated ``data.yaml``.
        """
        # Create root if it doesn't exist at all
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for split in self.splits:
            (self.root_dir / split / self.images_folder).mkdir(parents=True, exist_ok=True)
            (self.root_dir / split / self.labels_folder).mkdir(parents=True, exist_ok=True)

        self.generate_synthetic_dataset()
        self.validate_structure()
        return self.generate_data_yaml()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        """Return a sorted list of image files in *directory*.

        Args:
            directory: Directory to search.

        Returns:
            Sorted list of image ``Path`` objects.
        """
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted(
            p for p in directory.iterdir() if p.suffix.lower() in extensions
        )

    def _generate_single_image(
        self, images_dir: Path, labels_dir: Path, filename: str
    ) -> None:
        """Generate one synthetic helipad image and its YOLO label.

        A random background (sky / terrain gradient) is created, then a
        white "H" helipad marker is painted at a random location and
        scale.  The corresponding YOLO-format label (``class cx cy w h``
        normalised) is written.

        Args:
            images_dir: Directory to write the image file.
            labels_dir: Directory to write the label file.
            filename: Base filename (without extension).
        """
        size = self._syn_image_size
        img = self._random_background(size)

        # Random helipad size
        min_px = int(size * self._syn_min_ratio)
        max_px = int(size * self._syn_max_ratio)
        helipad_px = random.randint(min_px, max_px)

        # Random centre ensuring the helipad stays within bounds
        margin = helipad_px // 2 + 10
        cx = random.randint(margin, size - margin)
        cy = random.randint(margin, size - margin)

        self._draw_helipad(img, cx, cy, helipad_px)

        # Save image
        image_path = images_dir / f"{filename}.jpg"
        cv2.imwrite(str(image_path), img)

        # YOLO label: class cx cy w h (normalised)
        x_norm = cx / size
        y_norm = cy / size
        w_norm = helipad_px / size
        h_norm = helipad_px / size
        label_path = labels_dir / f"{filename}.txt"
        with open(label_path, "w", encoding="utf-8") as handle:
            handle.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    @staticmethod
    def _random_background(size: int) -> np.ndarray:
        """Create a random background image simulating aerial views.

        Args:
            size: Width and height in pixels.

        Returns:
            BGR image array of shape ``(size, size, 3)``.
        """
        bg_type = random.choice(["sky", "terrain", "water"])
        img = np.zeros((size, size, 3), dtype=np.uint8)

        if bg_type == "sky":
            # Vertical gradient from light blue to white
            top = np.array([180, 130, 70], dtype=np.float32)    # sky blue BGR
            bottom = np.array([255, 255, 240], dtype=np.float32)
        elif bg_type == "terrain":
            top = np.array([60, 80, 40], dtype=np.float32)      # dark green
            bottom = np.array([100, 120, 70], dtype=np.float32)
        else:  # water
            top = np.array([120, 60, 20], dtype=np.float32)     # dark blue
            bottom = np.array([180, 100, 40], dtype=np.float32)

        for y in range(size):
            ratio = y / size
            row = top * (1 - ratio) + bottom * ratio
            img[y, :] = row.astype(np.uint8)

        # Add random noise
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _draw_helipad(
        img: np.ndarray, cx: int, cy: int, size_px: int
    ) -> None:
        """Draw a white "H" helipad marker onto *img* in-place.

        The marker consists of an outer circle and a bold "H" shape.

        Args:
            img: BGR image array (modified in-place).
            cx: Centre X pixel coordinate.
            cy: Centre Y pixel coordinate.
            size_px: Overall diameter of the helipad marker in pixels.
        """
        white = (255, 255, 255)
        outline = (200, 200, 200)

        # Outer circle
        radius = size_px // 2
        cv2.circle(img, (cx, cy), radius, outline, 2)
        cv2.circle(img, (cx, cy), radius - 3, white, -1)

        # "H" dimensions
        h_height = int(radius * 1.1)
        h_width = int(radius * 0.7)
        bar_thickness = max(2, size_px // 20)

        left_x = cx - h_width // 2
        right_x = cx + h_width // 2
        top_y = cy - h_height // 2
        bottom_y = cy + h_height // 2

        # Two vertical bars
        cv2.line(img, (left_x, top_y), (left_x, bottom_y), (40, 40, 40), bar_thickness)
        cv2.line(img, (right_x, top_y), (right_x, bottom_y), (40, 40, 40), bar_thickness)
        # Horizontal crossbar
        cv2.line(img, (left_x, cy), (right_x, cy), (40, 40, 40), bar_thickness)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry: validate dataset and generate data.yaml."""
    from config.config import setup_logging
    setup_logging()

    setup = DatasetSetup()
    data_yaml = setup.ensure_ready()
    _LOGGER.info("Dataset ready. data.yaml: %s", data_yaml)


if __name__ == "__main__":
    main()
