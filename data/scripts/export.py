"""Step 6 — write ``data/data.yaml`` for the split dataset (the handoff to Coder 1).

The dataset is already in YOLO format on disk after ``split.py``; this just
emits the ``data.yaml`` that Ultralytics reads, pointing at the processed splits
and listing the locked class names in order.

Run:  python data/scripts/export.py
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from config import Paths, load_config, resolve_paths

logger = logging.getLogger(__name__)


def build_data_yaml(paths: Paths, classes: list[str]) -> dict[str, Any]:
    """Build the ``data.yaml`` dict with paths relative to its own location."""
    # Paths relative to the data.yaml file so it works in Colab and locally.
    base = paths.data_yaml.parent
    rel = lambda d: os.path.relpath(d, base).replace(os.sep, "/")
    return {
        "path": rel(paths.processed_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": list(classes),
    }


def run(cfg: dict[str, Any], paths: Paths) -> Path:
    """Write ``paths.data_yaml`` and return its path."""
    classes = cfg["classes"]
    data = build_data_yaml(paths, classes)
    paths.data_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.data_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    logger.info("Wrote %s (nc=%d, names=%s)", paths.data_yaml, data["nc"], data["names"])
    return paths.data_yaml


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run(cfg, paths)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
