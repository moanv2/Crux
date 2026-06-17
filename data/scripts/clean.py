"""Step 2 — clean and normalize the raw dataset into ``data/interim/``.

Normalizes class names, resizes to 640, fixes EXIF/rotation, dedupes near
-duplicates, and drops corrupt or empty images. Reads knobs from
``config.yaml`` ``clean:``. Writes a flat ``images/`` + ``labels/`` layout.

Run:  python data/scripts/clean.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from config import Paths, load_config, resolve_paths

logger = logging.getLogger(__name__)


def fix_orientation_and_resize(image_path: Path, size: int) -> None:
    """Apply the EXIF orientation, then resize the longest side to ``size``."""
    # TODO: PIL.ImageOps.exif_transpose + resize; keep label coords consistent.
    raise NotImplementedError


def find_near_duplicates(image_paths: list[Path], hash_size: int) -> set[Path]:
    """Return images that are perceptual-hash near-duplicates of an earlier one."""
    # TODO: imagehash.phash with the given hash_size; keep the first of each cluster.
    raise NotImplementedError


def normalize_class_names(label_dir: Path, classes: list[str]) -> None:
    """Remap label class indices/names to the locked ``classes`` order."""
    # TODO: build {old_name -> new_index} and rewrite each YOLO .txt line.
    raise NotImplementedError


def run(cfg: dict[str, Any], paths: Paths) -> Path:
    """Clean ``paths.raw_dir`` into ``paths.interim_dir``; return interim dir."""
    c = cfg["clean"]
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "clean: resize_to=%s fix_exif=%s dedupe=%s drop_corrupt=%s",
        c["resize_to"], c["fix_exif"], c["dedupe"], c["drop_corrupt"],
    )
    # TODO: orchestrate fix_orientation_and_resize -> find_near_duplicates ->
    #       drop corrupt/empty -> normalize_class_names, writing into interim_dir.
    raise NotImplementedError("TODO: implement the cleaning pipeline.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    out = run(cfg, paths)
    logger.info("Cleaned dataset written to %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
