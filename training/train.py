"""Fine-tune YOLO26 on the hold dataset — Coder 1's graded transfer-learning core.

Reads model / epochs / batch / lr from config.yaml `training:` and the image size
from `image_size`. Trains on `paths.data_yaml`, then copies the best checkpoint to
`paths.weights` (models/best.pt) — the one agreed location Coder 2 consumes.

Run (Colab/GPU):  python training/train.py
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Reuse Diego's config loader (data/scripts/config.py finds the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))
from config import Paths, load_config, resolve_paths  # noqa: E402

logger = logging.getLogger(__name__)


def _device() -> str:
    """Pick a device that works on Colab (CUDA), Apple Silicon (MPS), or CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_train_kwargs(cfg: dict, paths: Paths) -> dict:
    """Translate config.yaml into Ultralytics ``model.train()`` keyword arguments.

    Pure + torch-free (bar the device probe) so it is unit-testable without a GPU. Diego
    sets the augmentation recipe in ``config.yaml augmentation:`` and we apply it here (the
    Roboflow export has no baked-in augmentation). Ultralytics has no literal "brightness"
    knob — HSV-value (``hsv_v``) is the standard proxy.
    """
    t = cfg["training"]
    aug = cfg.get("augmentation", {})
    return {
        "data": str(paths.data_yaml),
        "imgsz": cfg["image_size"],
        "epochs": t.get("epochs", 80),
        "batch": t.get("batch", 16),
        "lr0": t.get("lr0", 0.01),
        "patience": t.get("patience", 20),
        "seed": t.get("seed", 42),
        "deterministic": True,
        "project": str(paths.root / "training" / "runs"),
        "name": "finetune",
        "device": _device(),
        "fliplr": aug.get("fliplr", 0.5),
        "flipud": aug.get("flipud", 0.0),
        "hsv_v": aug.get("brightness", 0.2),   # brightness ≈ HSV value
        "degrees": aug.get("rotation_deg", 10),
    }


def run(cfg: dict, paths: Paths):
    """Fine-tune and freeze weights to models/best.pt."""
    t = cfg["training"]
    if not paths.data_yaml.exists():
        raise RuntimeError(f"{paths.data_yaml} not found — run data/scripts/run_pipeline.py first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics not installed — pip install ultralytics") from exc

    kwargs = build_train_kwargs(cfg, paths)
    model = YOLO(t.get("model", "yolo26n.pt"))
    results = model.train(**kwargs)

    best = Path(results.save_dir) / "weights" / "best.pt"
    paths.weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, paths.weights)
    logger.info("Training done. Frozen weights -> %s", paths.weights)
    return paths.weights


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
