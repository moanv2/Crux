"""Label QA — counts instances per class and flags suspicious boxes.

Walks every YOLO label file under a dataset root and reports:
  - per-class instance counts (and any class index outside the locked list),
  - empty label files (images with no boxes),
  - boxes whose coords fall outside [0, 1],
  - oversized boxes (area > ``sanity.max_box_area_frac``),
  - degenerate boxes (a side < ``sanity.min_box_side_frac``).

Exits non-zero if any problems are found, so it can gate the pipeline.

Run:  python data/scripts/sanity_check.py                 # checks processed_dir
      python data/scripts/sanity_check.py --root data/interim
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from config import Paths, load_config, resolve_paths
from labels import iter_label_files, parse_boxes

logger = logging.getLogger(__name__)


def check(root: Path, classes: list[str], sanity: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problem strings (empty == all good)."""
    problems: list[str] = []
    counts: Counter = Counter()
    n_files = 0
    max_area = sanity["max_box_area_frac"]
    min_side = sanity["min_box_side_frac"]

    for label_file in iter_label_files(root):
        n_files += 1
        boxes = parse_boxes(label_file)
        if not boxes:
            problems.append(f"empty label file: {label_file}")
            continue
        for i, b in enumerate(boxes):
            counts[b.cls] += 1
            if b.cls < 0 or b.cls >= len(classes):
                problems.append(f"{label_file}:{i} class {b.cls} outside class list")
            if not all(0.0 <= v <= 1.0 for v in (b.cx, b.cy, b.w, b.h)):
                problems.append(f"{label_file}:{i} coords out of [0,1]: {b}")
            if b.area > max_area:
                problems.append(f"{label_file}:{i} box area {b.area:.2f} > {max_area}")
            if b.w < min_side or b.h < min_side:
                problems.append(f"{label_file}:{i} degenerate box (w={b.w:.4f}, h={b.h:.4f})")

    logger.info("Scanned %d label files, %d boxes total.", n_files, sum(counts.values()))
    for idx, name in enumerate(classes):
        logger.info("  class %d %-8s: %d", idx, name, counts.get(idx, 0))
    return problems


def run(cfg: dict[str, Any], paths: Paths, root: Path | None = None) -> int:
    """Run the QA check; return the number of problems found."""
    root = root or paths.processed_dir
    if not root.exists():
        logger.warning("Nothing to check at %s yet.", root)
        return 0
    problems = check(root, cfg["classes"], cfg["sanity"])
    if problems:
        logger.warning("Found %d problem(s):", len(problems))
        for p in problems:
            logger.warning("  - %s", p)
    else:
        logger.info("No problems found. ✅")
    return len(problems)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    ap.add_argument("--root", default=None, help="Dataset dir to check (default: processed_dir).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    root = (paths.root / args.root).resolve() if args.root else None
    n_problems = run(cfg, paths, root)
    return 1 if n_problems else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
