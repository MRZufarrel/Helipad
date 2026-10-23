"""Configuration loader for the Cargo Drone Helipad Vision Engine.

This module provides a singleton-style access point to all configuration
values defined in ``config/config.yaml``.  No other module in the
``core_ai`` package should hardcode magic numbers — they must read from
the :data:`CONFIG` object exposed here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_LOGGER = logging.getLogger(__name__)

# Resolve project root relative to this file: config/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


class ConfigurationError(Exception):
    """Raised when the configuration file cannot be loaded or parsed."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Absolute path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If the file does not exist or contains
            invalid YAML.
    """
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found at: {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML configuration at {path}: {exc}"
        ) from exc
    if data is None:
        raise ConfigurationError(
            f"Configuration file is empty: {path}"
        )
    _LOGGER.info("Configuration loaded successfully from %s", path)
    return data


class Config:
    """Thin wrapper around the raw configuration dictionary.

    Provides dot-notation access to top-level sections while keeping the
    underlying dict available for dynamic lookups.
    """

    def __init__(self, data: Dict[str, Any], project_root: Path) -> None:
        self._data = data
        self._project_root = project_root

    @property
    def project_root(self) -> Path:
        """Return the absolute project root directory."""
        return self._project_root

    def get(self, *keys: str, default: Optional[Any] = None) -> Any:
        """Retrieve a nested configuration value.

        Args:
            *keys: Sequential keys to traverse the config tree.
            default: Value to return if any key is missing.

        Returns:
            The resolved value or *default*.
        """
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                _LOGGER.debug(
                    "Config key path %s not found, returning default.",
                    " -> ".join(keys),
                )
                return default
            node = node[key]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        """Return an entire configuration section.

        Args:
            name: Top-level section name (e.g. ``"dataset"``).

        Returns:
            Dictionary of that section.

        Raises:
            ConfigurationError: If the section does not exist.
        """
        if name not in self._data:
            raise ConfigurationError(
                f"Configuration section '{name}' not found."
            )
        return self._data[name]

    def resolve_path(self, relative: str) -> Path:
        """Resolve a project-relative path to an absolute ``Path``.

        Args:
            relative: Path string relative to the project root.

        Returns:
            Absolute :class:`pathlib.Path`.
        """
        p = Path(relative)
        if p.is_absolute():
            return p
        return self._project_root / p


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
CONFIG = Config(_load_yaml(_CONFIG_PATH), _PROJECT_ROOT)


def get_config() -> Config:
    """Return the global :class:`Config` singleton.

    Returns:
        The shared configuration instance.
    """
    return CONFIG


def setup_logging() -> None:
    """Configure the root logger from configuration values.

    This should be called once at application startup.
    """
    log_cfg = CONFIG.section("logging")
    log_dir = CONFIG.resolve_path(log_cfg.get("log_dir", "logs"))
    log_file = CONFIG.resolve_path(log_cfg.get("log_file", "logs/vision_engine.log"))
    os.makedirs(str(log_dir), exist_ok=True)

    level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt=log_cfg.get("log_format", "%(asctime)s | %(levelname)-8s | %(message)s"),
        datefmt=log_cfg.get("date_format", "%Y-%m-%dT%H:%M:%S%z"),
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplication on re-init
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _LOGGER.info("Logging initialised — level=%s, file=%s", logging.getLevelName(level), log_file)
