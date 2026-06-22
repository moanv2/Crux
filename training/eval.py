"""Evaluate the fine-tuned hold detector — metrics + figures (Coder 1).

Runs validation on models/best.pt → mAP50, mAP50-95. Ultralytics also writes a PR
curve, confusion matrix, and sample-prediction grids into the run dir. For the
graded notebook, also compare against the COCO-pretrained baseline (before/after)
to make the transfer-learning gain explicit.

Run:  python training/eval.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))
from config import Paths, load_config, resolve_paths  # noqa: E402

logger = logging.getLogger(__name__)


def run(cfg: dict, paths: Paths):
    """Validate models/best.pt on the dataset and log key metrics."""
    if not paths.weights.exists():
        raise RuntimeError(f"{paths.weights} not found — run training/train.py first.")
    if not paths.data_yaml.exists():
        raise RuntimeError(f"{paths.data_yaml} not found — run data/scripts/run_pipeline.py first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics not installed — pip install ultralytics") from exc

    metrics = YOLO(str(paths.weights)).val(data=str(paths.data_yaml), imgsz=cfg["image_size"])
    logger.info("mAP50=%.4f  mAP50-95=%.4f", metrics.box.map50, metrics.box.map)
    logger.info("PR curve / confusion matrix / sample grids saved under: %s", metrics.save_dir)
    # TODO (notebook): also val() a COCO-pretrained yolo26n.pt for the before/after story.
    return metrics


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
