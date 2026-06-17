"""Shared config loader for the data pipeline.

Finds the repo root (the directory containing ``config.yaml``), loads it, and
resolves every ``paths:`` entry to an absolute path. Import this from every
data script so no module ever hardcodes a path — the same code runs in Colab
and locally.

    from config import load_config, resolve_paths
    cfg = load_config()
    paths = resolve_paths(cfg)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILENAME = "config.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory contains ``config.yaml``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / CONFIG_FILENAME).is_file():
            return parent
    raise FileNotFoundError(f"Could not find {CONFIG_FILENAME} above {here}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` as a dict. Defaults to the one at the repo root."""
    cfg_path = Path(path) if path else find_repo_root() / CONFIG_FILENAME
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Paths:
    """Absolute, resolved project paths (see ``config.yaml`` ``paths:``)."""

    root: Path
    raw_dir: Path
    custom_dir: Path
    interim_dir: Path
    processed_dir: Path
    data_yaml: Path
    data_card: Path
    weights: Path


def resolve_paths(cfg: dict[str, Any], root: Path | None = None) -> Paths:
    """Resolve ``cfg['paths']`` against the repo root into absolute paths."""
    root = (root or find_repo_root()).resolve()
    p = cfg["paths"]

    def r(key: str) -> Path:
        return (root / p[key]).resolve()

    return Paths(
        root=root,
        raw_dir=r("raw_dir"),
        custom_dir=r("custom_dir"),
        interim_dir=r("interim_dir"),
        processed_dir=r("processed_dir"),
        data_yaml=r("data_yaml"),
        data_card=r("data_card"),
        weights=r("weights"),
    )
