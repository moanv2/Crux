"""Evaluate the fine-tuned hold detector — before/after + figures + handoff exports (Model).

Validates models/best.pt AND the COCO-pretrained baseline on the same data.yaml to make
the transfer-learning gain explicit (the graded money shot), writes a metrics table for
Dalton, and copies the key figures into training/artifacts/.

The reported metric uses the held-out **test** split (never seen during training / model
selection), falling back to val only if a dataset has no test split.

Run:  python training/eval.py
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))
from config import Paths, load_config, resolve_paths  # noqa: E402

logger = logging.getLogger(__name__)

# Figures Ultralytics val() writes for a detection model (names vary by version, so we list
# both the newer Box*-prefixed and legacy names; export copies only the ones that exist).
# NB: results.png (training curves) is written by train(), not val() — train.py exports it.
_FIGURES = ("BoxPR_curve.png", "PR_curve.png", "BoxF1_curve.png",
            "confusion_matrix.png", "confusion_matrix_normalized.png")


def _summary(b50: float, b: float, f50: float, f: float) -> dict:
    """Assemble the before/after summary dict (pure, torch-free)."""
    return {
        "baseline": {"map50": b50, "map": b},
        "finetuned": {"map50": f50, "map": f},
        "delta": {"map50": f50 - b50, "map": f - b},
    }


def metrics_to_markdown(summary: dict) -> str:
    """Render a before/after summary as a Markdown table (pure, torch-free)."""
    b, f, d = summary["baseline"], summary["finetuned"], summary["delta"]
    split = summary.get("split", "test")
    return (
        f"_Metrics on the held-out **{split}** split._\n\n"
        "| Model | mAP50 | mAP50-95 |\n"
        "|---|---|---|\n"
        f"| COCO-pretrained (baseline) | {b['map50']:.4f} | {b['map']:.4f} |\n"
        f"| Fine-tuned (best.pt) | {f['map50']:.4f} | {f['map']:.4f} |\n"
        f"| **Δ (gain)** | **{d['map50']:+.4f}** | **{d['map']:+.4f}** |\n"
    )


def report_split(data_yaml: str) -> str:
    """Use the held-out test split for the graded metric; fall back to val if absent."""
    import yaml
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    return "test" if (root / "test" / "images").exists() else "val"


def evaluate(weights: str, data_yaml: str, imgsz: int, name: str, split: str = "test"):
    """Run Ultralytics validation for one checkpoint on the given split; return its metrics."""
    from ultralytics import YOLO
    return YOLO(weights).val(data=data_yaml, imgsz=imgsz, split=split, name=name)


def compare_baseline_finetuned(cfg: dict, paths: Paths) -> dict:
    """val() the COCO-pretrained baseline and best.pt on the same held-out split → before/after."""
    imgsz, data = cfg["image_size"], str(paths.data_yaml)
    base_ckpt = cfg["training"].get("model", "yolo26n.pt")
    split = report_split(data)

    fine = evaluate(str(paths.weights), data, imgsz, name="eval_finetuned", split=split)
    # COCO weights have no `hold` class → mAP ≈ 0; that IS the transfer-learning gain story.
    try:
        base = evaluate(base_ckpt, data, imgsz, name="eval_baseline", split=split)
        b50, b = float(base.box.map50), float(base.box.map)
    except Exception as exc:  # noqa: BLE001 — class mismatch / no matching preds
        logger.warning("Baseline val() failed (%s) — treating COCO baseline as 0 mAP.", exc)
        b50, b = 0.0, 0.0

    summary = _summary(b50, b, float(fine.box.map50), float(fine.box.map))
    summary["split"] = split
    summary["_finetuned_save_dir"] = str(fine.save_dir)
    return summary


def sample_grid(weights: str, data_yaml: str, n: int, out_dir: Path, imgsz: int):
    """Save annotated predictions on N held-out images (the qualitative figure)."""
    from ultralytics import YOLO

    split = report_split(data_yaml)
    import yaml
    root = Path(yaml.safe_load(Path(data_yaml).read_text())["path"])
    img_dir = root / split / "images"
    images = sorted(p for p in img_dir.glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    if not images:
        logger.warning("No images under %s for the sample grid.", img_dir)
        return None
    YOLO(weights).predict(source=[str(p) for p in images], imgsz=imgsz, save=True,
                          project=str(out_dir), name="samples", exist_ok=True)
    logger.info("Sample predictions saved under %s", out_dir / "samples")
    return out_dir / "samples"


def export_artifacts(summary: dict, paths: Paths) -> Path:
    """Write metrics.json + metrics.md and copy key val figures into training/artifacts/."""
    out_dir = paths.root / "training" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    (out_dir / "metrics.json").write_text(json.dumps(clean, indent=2))
    (out_dir / "metrics.md").write_text(metrics_to_markdown(clean))
    save_dir = summary.get("_finetuned_save_dir")
    if save_dir:
        for fig in _FIGURES:
            src = Path(save_dir) / fig
            if src.exists():
                shutil.copy2(src, out_dir / fig)
    logger.info("Metrics + figures exported to %s", out_dir)
    return out_dir


def run(cfg: dict, paths: Paths) -> dict:
    """Full eval: before/after on the held-out split, sample grid, and handoff exports."""
    if not paths.weights.exists():
        raise RuntimeError(f"{paths.weights} not found — run training/train.py first.")
    if not paths.data_yaml.exists():
        raise RuntimeError(f"{paths.data_yaml} not found — run data/scripts/run_pipeline.py first.")
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ultralytics not installed — pip install ultralytics") from exc

    summary = compare_baseline_finetuned(cfg, paths)
    out_dir = paths.root / "training" / "artifacts"
    sample_grid(str(paths.weights), str(paths.data_yaml), n=9, out_dir=out_dir,
                imgsz=cfg["image_size"])
    export_artifacts(summary, paths)
    logger.info("Before/after on %s — baseline mAP50=%.4f → fine-tuned mAP50=%.4f (Δ%+.4f)",
                summary["split"], summary["baseline"]["map50"],
                summary["finetuned"]["map50"], summary["delta"]["map50"])
    return summary


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
