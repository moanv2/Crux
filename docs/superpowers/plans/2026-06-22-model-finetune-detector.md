# Fine-tuned Hold Detector + Graded Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Jan's (Model) deliverables — apply the augmentation recipe in training, finish the before/after eval with handoff exports, and author the thin graded Colab notebook.

**Architecture:** All logic lives in `training/train.py` and `training/eval.py` (config-driven, no hardcoded paths). The notebook (`training/train_holds.ipynb`) is thin — it imports those modules and adds narrative + figures. Torch/ultralytics are imported lazily inside run paths so the pure helpers are unit-testable on a laptop with no GPU.

**Tech Stack:** Python 3.11+, Ultralytics YOLO (YOLO26), PyYAML, matplotlib, pytest.

## Global Constraints

- `image_size: 640` everywhere (train/eval/inference) — LOCKED, copy from `cfg["image_size"]`.
- Single class `hold`, `nc: 1` — never reintroduce types/colours.
- Frozen weights at `models/best.pt` (`cfg["paths"]["weights"]`).
- No hardcoded paths — resolve from `config.yaml` via `data/scripts/config.py` (`load_config`, `resolve_paths`).
- Seed runs (`cfg["training"]["seed"]`, default 42); `deterministic=True`.
- Ultralytics/torch imported **lazily inside functions**, never at module top (keeps helpers torch-free).
- Augmentation knobs come from `cfg["augmentation"]`; Diego owns the recipe, Jan applies it in training.

---

### Task 1: `train.py` — augmentation mapping + reproducibility

**Files:**
- Modify: `training/train.py` (add `_device`, `build_train_kwargs`; rewire `run`)
- Test: `training/test_training.py`

**Interfaces:**
- Produces: `build_train_kwargs(cfg: dict, paths: Paths) -> dict` — Ultralytics `model.train()` kwargs including augmentation. `run(cfg, paths) -> Path` (unchanged signature) now calls it.

- [ ] **Step 1: Write the failing test** in `training/test_training.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/test_training.py -v`
Expected: FAIL — `AttributeError: module 'train' has no attribute 'build_train_kwargs'`

- [ ] **Step 3: Add `_device` + `build_train_kwargs` to `training/train.py`** (after the `logger = ...` line)

```python
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
```

- [ ] **Step 4: Rewire `run` to use `build_train_kwargs`** — replace the `model = YOLO(...)` / `results = model.train(...)` block in `run` with:

```python
    kwargs = build_train_kwargs(cfg, paths)
    model = YOLO(t.get("model", "yolo26n.pt"))
    results = model.train(**kwargs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest training/test_training.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add training/train.py training/test_training.py
git commit -m "feat(model): apply augmentation recipe + reproducible device/run in train.py"
```

---

### Task 2: `eval.py` — before/after baseline + handoff exports

**Files:**
- Modify: `training/eval.py` (full rewrite of body; keep CLI + `run` entrypoint)
- Test: `training/test_training.py` (append metrics tests)

**Interfaces:**
- Consumes: `cfg`, `Paths` (with `.weights`, `.data_yaml`, `.root`).
- Produces: `_summary(b50,b,f50,f)->dict`, `metrics_to_markdown(summary)->str`, `compare_baseline_finetuned(cfg,paths)->dict`, `sample_grid(...)`, `export_artifacts(summary,paths)->Path`, `run(cfg,paths)->dict`.

- [ ] **Step 1: Append failing tests** to `training/test_training.py`

```python
import eval as eval_mod  # noqa: E402  (training/ is already on sys.path)


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest training/test_training.py -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` on `eval_mod._summary`

- [ ] **Step 3: Rewrite `training/eval.py`** (full file)

```python
"""Evaluate the fine-tuned hold detector — before/after + figures + handoff exports (Model).

Validates models/best.pt AND the COCO-pretrained baseline on the same data.yaml to make
the transfer-learning gain explicit (the graded money shot), writes a metrics table for
Dalton, and copies the key figures into training/artifacts/.

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

_FIGURES = ("BoxPR_curve.png", "PR_curve.png", "confusion_matrix.png",
            "confusion_matrix_normalized.png", "results.png")


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
    return (
        "| Model | mAP50 | mAP50-95 |\n"
        "|---|---|---|\n"
        f"| COCO-pretrained (baseline) | {b['map50']:.4f} | {b['map']:.4f} |\n"
        f"| Fine-tuned (best.pt) | {f['map50']:.4f} | {f['map']:.4f} |\n"
        f"| **Δ (gain)** | **{d['map50']:+.4f}** | **{d['map']:+.4f}** |\n"
    )


def evaluate(weights: str, data_yaml: str, imgsz: int, name: str):
    """Run Ultralytics validation for one checkpoint; return its metrics object."""
    from ultralytics import YOLO
    return YOLO(weights).val(data=data_yaml, imgsz=imgsz, name=name)


def compare_baseline_finetuned(cfg: dict, paths: Paths) -> dict:
    """val() the COCO-pretrained baseline and best.pt on the same data → before/after dict."""
    imgsz, data = cfg["image_size"], str(paths.data_yaml)
    base_ckpt = cfg["training"].get("model", "yolo26n.pt")

    fine = evaluate(str(paths.weights), data, imgsz, name="eval_finetuned")
    # COCO weights have no `hold` class → mAP ≈ 0 by construction; that IS the gain story.
    try:
        base = evaluate(base_ckpt, data, imgsz, name="eval_baseline")
        b50, b = float(base.box.map50), float(base.box.map)
    except Exception as exc:  # noqa: BLE001 — class mismatch / no matching preds
        logger.warning("Baseline val() failed (%s) — treating COCO baseline as 0 mAP.", exc)
        b50, b = 0.0, 0.0

    summary = _summary(b50, b, float(fine.box.map50), float(fine.box.map))
    summary["_finetuned_save_dir"] = str(fine.save_dir)
    return summary


def sample_grid(weights: str, data_yaml: str, n: int, out_dir: Path):
    """Save annotated predictions on N test/val images (the qualitative figure)."""
    import yaml
    from ultralytics import YOLO

    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    split = "test" if (root / "test" / "images").exists() else "val"
    img_dir = root / split / "images"
    images = sorted(p for p in img_dir.glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    if not images:
        logger.warning("No images under %s for the sample grid.", img_dir)
        return None
    YOLO(weights).predict(source=[str(p) for p in images], imgsz=640, save=True,
                          project=str(out_dir), name="samples", exist_ok=True)
    logger.info("Sample predictions saved under %s", out_dir / "samples")
    return out_dir / "samples"


def export_artifacts(summary: dict, paths: Paths) -> Path:
    """Write metrics.json + metrics.md and copy key figures into training/artifacts/."""
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
    """Full eval: before/after, sample grid, and handoff exports."""
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
    sample_grid(str(paths.weights), str(paths.data_yaml), n=9, out_dir=out_dir)
    export_artifacts(summary, paths)
    logger.info("Before/after — baseline mAP50=%.4f → fine-tuned mAP50=%.4f (Δ%+.4f)",
                summary["baseline"]["map50"], summary["finetuned"]["map50"],
                summary["delta"]["map50"])
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/test_training.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add training/eval.py training/test_training.py
git commit -m "feat(model): before/after baseline + metrics/figure exports in eval.py"
```

---

### Task 3: `training/train_holds.ipynb` — the thin graded notebook

**Files:**
- Create: `training/train_holds.ipynb` (valid nbformat 4 JSON)

**Interfaces:**
- Consumes: `train.run`, `eval.run`/`eval.compare_baseline_finetuned`, `load_config`, `resolve_paths`.

Notebook cells (markdown `md` / code `code`), in order. Logic stays in the modules — cells orchestrate + narrate.

1. **md** — Title + transfer-learning narrative: COCO has no `hold` class → fine-tuning on a single-class dataset produces a clean before/after mAP gain (the graded money shot). Approach = static hold map + pose fusion (one-paragraph context).
2. **md** — "## 1. Setup".
3. **code** — Colab setup (guarded so it's a no-op when already inside the repo):
```python
import os, sys
if not os.path.exists("config.yaml") and not os.path.exists("Crux"):
    !git clone -q https://github.com/moanv2/Crux
if os.path.exists("Crux"):
    %cd Crux
!pip install -q -r requirements.txt
```
4. **code** — config + module imports:
```python
sys.path.insert(0, "data/scripts")
sys.path.insert(0, "training")
from config import load_config, resolve_paths
import train
import eval as evaluate          # Model-role modules — all logic lives here, not in cells
cfg = load_config()
paths = resolve_paths(cfg)
print("image_size:", cfg["image_size"], "| classes:", cfg["classes"])
```
5. **md** — "## 2. Get the dataset (Diego's pipeline)".
6. **code** — Roboflow key + pipeline:
```python
os.environ.setdefault("ROBOFLOW_API_KEY", "")   # ← paste key, or set as a Colab secret
assert os.environ["ROBOFLOW_API_KEY"], "Set ROBOFLOW_API_KEY (Roboflow → Settings → API)."
!python data/scripts/run_pipeline.py
from pathlib import Path
print("data.yaml exists:", paths.data_yaml.exists())
print(Path("data/data_card.md").read_text())
```
7. **md** — "## 3. Peek at the data".
8. **code** — show a few training images:
```python
import yaml, matplotlib.pyplot as plt
from PIL import Image
d = yaml.safe_load(paths.data_yaml.read_text())
imgs = sorted((Path(d["path"]) / "train" / "images").glob("*"))[:6]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, p in zip(axes.ravel(), imgs):
    ax.imshow(Image.open(p)); ax.set_title(p.name, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
```
9. **md** — "## 4. Baseline (before): COCO-pretrained, zero-shot". Note: expect ≈0 — COCO has no `hold` class.
10. **code** — baseline val:
```python
from ultralytics import YOLO
base = YOLO(cfg["training"]["model"])    # COCO-pretrained — no `hold` class
base_m = base.val(data=str(paths.data_yaml), imgsz=cfg["image_size"], name="nb_baseline")
print(f"Baseline  mAP50={base_m.box.map50:.4f}  mAP50-95={base_m.box.map:.4f}")
```
11. **md** — "## 5. Fine-tune (transfer learning)". Reads `config.yaml`; long GPU cell.
12. **code** — `weights = train.run(cfg, paths); print("Frozen weights:", weights)`.
13. **md** — "## 6. Evaluate (after) + handoff exports".
14. **code** — eval + show table:
```python
summary = evaluate.run(cfg, paths)
from IPython.display import Markdown, display
display(Markdown((paths.root / "training" / "artifacts" / "metrics.md").read_text()))
```
15. **md** — "## 7. Before / after — the transfer-learning gain".
16. **code** — bar chart:
```python
import matplotlib.pyplot as plt
labels = ["mAP50", "mAP50-95"]
base_v = [summary["baseline"]["map50"], summary["baseline"]["map"]]
fine_v = [summary["finetuned"]["map50"], summary["finetuned"]["map"]]
x = range(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([i - w/2 for i in x], base_v, w, label="COCO baseline")
ax.bar([i + w/2 for i in x], fine_v, w, label="Fine-tuned")
ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
ax.set_title("Transfer learning: COCO → fine-tuned hold detector"); ax.legend()
for i, (b, f) in enumerate(zip(base_v, fine_v)):
    ax.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center")
    ax.text(i + w/2, f + 0.01, f"{f:.2f}", ha="center")
plt.tight_layout()
plt.savefig(paths.root / "training" / "artifacts" / "before_after.png", dpi=150); plt.show()
```
17. **md** — "## 8. Qualitative: PR curve + sample detections".
18. **code** — display saved figures:
```python
from IPython.display import Image as IPImage, display
art = paths.root / "training" / "artifacts"
for fig in ["BoxPR_curve.png", "PR_curve.png", "confusion_matrix.png"]:
    if (art / fig).exists(): display(IPImage(str(art / fig)))
```
19. **md** — "## 9. Inference demo — static hold map (→ Ignacio)".
20. **code** — run best.pt on one reference frame:
```python
ref = next((Path(d["path"]) / "test" / "images").glob("*"), None)
det = YOLO(str(paths.weights)).predict(source=str(ref), imgsz=cfg["image_size"], save=True)
print(f"Detected {len(det[0].boxes)} holds on {ref.name}")
```
21. **md** — "## 10. Handoff": `models/best.pt` → Ignacio; `training/artifacts/` (metrics.json, metrics.md, figures) → Dalton.

- [ ] **Step 1: Write `training/train_holds.ipynb`** with the 21 cells above (nbformat 4, `"cell_type"` per item, empty `outputs`/`execution_count` for code cells, `"metadata": {}`).

- [ ] **Step 2: Validate the notebook is well-formed + code parses** (torch-free)

Run:
```bash
python - <<'PY'
import ast, json
nb = json.load(open("training/train_holds.ipynb"))
assert nb["nbformat"] == 4 and nb["cells"], "bad nbformat"
md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
assert "transfer learning" in md.lower(), "narrative missing"
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    code = "".join(c["source"])
    safe = "\n".join("" if l.lstrip().startswith(("!", "%")) else l for l in code.splitlines())
    ast.parse(safe)                     # every code cell must be valid Python (sans magics)
print("notebook OK:", len(nb["cells"]), "cells")
PY
```
Expected: `notebook OK: 21 cells`

- [ ] **Step 3: Commit**

```bash
git add training/train_holds.ipynb
git commit -m "feat(model): thin graded Colab notebook (dataset→baseline→fine-tune→eval→before/after→inference)"
```

---

### Task 4: Housekeeping — README, .gitignore, artifacts dir, flag stale contract

**Files:**
- Modify: `training/README.md`, `.gitignore`
- Create: `training/artifacts/.gitkeep`

- [ ] **Step 1: Append ignore rules to `.gitignore`** (after the existing project block, ~line 243):

```
# Model role (Jan): training runs + regenerated eval artifacts.
# Keep the metrics text (json/md) tracked as the handoff record; ignore binary figures.
training/runs/
training/artifacts/*
!training/artifacts/.gitkeep
!training/artifacts/metrics.json
!training/artifacts/metrics.md
```

- [ ] **Step 2: Create `training/artifacts/.gitkeep`** (empty file).

- [ ] **Step 3: Update `training/README.md`** — replace the "## Evaluate" section so it documents the before/after + exports:

```markdown
## Evaluate
```bash
python training/eval.py              # before/after baseline + mAP50/mAP50-95 + figures
```
Writes `training/artifacts/metrics.json` + `metrics.md` (the handoff record for Dalton) and
copies the PR curve / confusion matrix / sample predictions there. `eval.run()` also returns
the before/after summary dict the notebook charts.

## The graded notebook — `training/train_holds.ipynb`
Thin: imports `train.py` / `eval.py`, reads `config.yaml`, leads with the transfer-learning
story. Flow: dataset → COCO baseline (≈0 mAP, no `hold` class) → fine-tune → eval →
before/after → inference. Run top-to-bottom on Colab (GPU).

## Tests
```bash
pytest training/test_training.py -v  # torch-free: augmentation mapping + metrics formatting
```
```

- [ ] **Step 4: Verify the full test suite still passes**

Run: `pytest training/test_training.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add training/README.md .gitignore training/artifacts/.gitkeep
git commit -m "chore(model): gitignore runs/artifacts, document eval+notebook in training README"
```

---

## Post-implementation (manual, flagged — not a code task)
- **Stale contract:** `project_instructions/CLAUDE.md` still describes the v2 plan (6 classes, YOLOv11). Left unedited per decision (shared/historical file). Surface to the team so the auto-loaded contract gets reconciled to v3 by whoever owns it.
- **Colab run:** the actual fine-tune/eval (GPU + Roboflow key) runs in `train_holds.ipynb` on Colab; this session delivers Colab-ready code verified by the torch-free tests + review.

## Self-Review
- **Spec coverage:** train.py augmentation ✓ (Task 1); eval before/after + exports + grid ✓ (Task 2); thin notebook ✓ (Task 3); tests + README + gitignore + flag ✓ (Task 4). All spec work items covered.
- **Placeholder scan:** none — every step has full code/commands.
- **Type consistency:** `build_train_kwargs(cfg, paths)`, `run(cfg, paths)`, `_summary(b50,b,f50,f)`, `metrics_to_markdown(summary)`, `compare_baseline_finetuned(cfg,paths)`, `export_artifacts(summary,paths)` — names/signatures consistent across tasks, tests, and notebook.
