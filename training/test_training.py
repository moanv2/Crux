"""Torch-free unit tests for the Model role (Jan): config→kwargs mapping + metrics formatting.

ultralytics/torch are imported lazily inside the run paths, so these tests pass anywhere —
Colab pre-install or a laptop with no GPU.   Run:  pytest training/test_training.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "training"))

from config import resolve_paths  # noqa: E402
import train  # noqa: E402
import eval as eval_mod  # noqa: E402  (training/ is on sys.path)

CFG = {
    "image_size": 640,
    "paths": {
        "raw_dir": "data/raw", "custom_dir": "data/custom", "interim_dir": "data/interim",
        "processed_dir": "data/processed", "data_yaml": "data/data.yaml",
        "data_card": "data/data_card.md", "weights": "models/best.pt",
    },
    "training": {"model": "yolo26n.pt", "epochs": 80, "batch": 16, "lr0": 0.01,
                 "patience": 20, "seed": 42},
    "augmentation": {"fliplr": 0.5, "flipud": 0.0, "brightness": 0.2, "rotation_deg": 10},
}


def test_build_train_kwargs_maps_augmentation_and_core():
    paths = resolve_paths(CFG, root=ROOT)
    k = train.build_train_kwargs(CFG, paths)
    assert k["fliplr"] == 0.5
    assert k["flipud"] == 0.0
    assert k["hsv_v"] == 0.2        # brightness proxy
    assert k["degrees"] == 10
    assert k["imgsz"] == 640
    assert k["epochs"] == 80
    assert k["batch"] == 16
    assert k["lr0"] == 0.01
    assert k["seed"] == 42
    assert k["deterministic"] is True
    assert k["name"] == "finetune"
    assert k["data"].endswith("data.yaml")
    assert k["device"] in {"cpu", "mps", "0"}


def test_build_train_kwargs_defaults_when_augmentation_missing():
    cfg = {k: v for k, v in CFG.items() if k != "augmentation"}
    paths = resolve_paths(cfg, root=ROOT)
    k = train.build_train_kwargs(cfg, paths)
    assert k["fliplr"] == 0.5 and k["flipud"] == 0.0
    assert k["hsv_v"] == 0.2 and k["degrees"] == 10


def test_summary_delta_is_finetuned_minus_baseline():
    s = eval_mod._summary(0.0, 0.0, 0.85, 0.61)
    assert s["delta"]["map50"] == 0.85
    assert round(s["delta"]["map"], 2) == 0.61


def test_metrics_to_markdown_has_table_and_gain_row():
    s = eval_mod._summary(0.01, 0.00, 0.85, 0.61)
    md = eval_mod.metrics_to_markdown(s)
    assert "| Model | mAP50 | mAP50-95 |" in md
    assert "Fine-tuned (best.pt)" in md
    assert "0.8500" in md
    assert "Δ (gain)" in md
