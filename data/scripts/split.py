"""Step 5 — split the cleaned dataset into train/val/test with no leakage.

Reads the flat ``interim_dir`` (``images/`` + ``labels/``) and copies each
image/label pair into ``processed_dir/{train,val,test}/{images,labels}``.
Ratios and seed come from ``config.yaml`` ``split:``. Stratification by class
is a TODO; for now it's a deterministic random split (seeded, reproducible).

Run:  python data/scripts/split.py
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Any

from config import Paths, load_config, resolve_paths

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(interim_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return (image, matching-label-or-None) pairs from a flat interim layout."""
    images_dir, labels_dir = interim_dir / "images", interim_dir / "labels"
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(images_dir.iterdir()) if images_dir.exists() else []:
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        pairs.append((img, label if label.exists() else None))
    return pairs


def split_pairs(pairs: list, ratios: tuple[float, float, float], seed: int) -> dict[str, list]:
    """Shuffle deterministically and partition into train/val/test by ratio."""
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def run(cfg: dict[str, Any], paths: Paths) -> Path:
    """Split ``interim_dir`` into ``processed_dir``; return the processed dir."""
    s = cfg["split"]
    ratios = (s["train"], s["val"], s["test"])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    if s.get("stratify"):
        logger.warning("stratify=true requested but not yet implemented; using random split.")

    pairs = find_pairs(paths.interim_dir)
    if not pairs:
        raise RuntimeError(f"No image/label pairs under {paths.interim_dir}/images. Run clean first.")

    splits = split_pairs(pairs, ratios, s["seed"])
    for name, items in splits.items():
        img_out = paths.processed_dir / name / "images"
        lbl_out = paths.processed_dir / name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy2(img, img_out / img.name)
            if label is not None:
                shutil.copy2(label, lbl_out / label.name)
        logger.info("%s: %d images", name, len(items))
    return paths.processed_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    out = run(cfg, paths)
    logger.info("Split written to %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
