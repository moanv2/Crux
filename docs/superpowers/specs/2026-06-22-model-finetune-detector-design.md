# Design — Model role (Jan): fine-tuned hold detector + graded notebook

**Date:** 2026-06-22
**Author:** Jan (Model), with Claude Code
**Status:** Approved (design), pending implementation

## Context

Crux is a computer-vision "climbing coach". The **graded core** is a fine-tuned
**YOLO26** detector for a single class `hold` (transfer learning), evaluated with a
**before/after** mAP comparison against the COCO-pretrained baseline. This is Jan's
(Model) deliverable. Diego (Data) hands off `data.yaml` + `data_card.md`; Jan hands
off `models/best.pt` → Ignacio (Integration) and metrics/figures → Dalton (Docs).

The project pivoted from the original v2 plan (6 hold-type classes, YOLOv11) to v3
(**single class `hold`, YOLO26**). Authoritative sources: `jan-claude-context.md`,
`config.yaml`, root `README.md`. `project_instructions/CLAUDE.md` is **stale** (still
describes v2) — flagged, not edited (it is a shared/historical file outside Model scope).

## Constraints / environment

- This laptop has **no GPU, no `torch`/`ultralytics`/`roboflow` installed, no
  `data/data.yaml`, no Roboflow key**. Real training/eval runs on **Colab GPU**.
- Therefore this session produces **Colab-ready code + the graded notebook**, not an
  executed training run. Verification here is limited to **torch-free unit tests**.
- Everything is **config-driven** (`config.yaml`); no hardcoded paths. `image_size: 640`
  and single class `hold` are LOCKED. Seed runs for reproducibility.
- The notebook is **thin**: it imports logic from `train.py`/`eval.py` (per `CLAUDE.md`
  "don't bury logic in one giant cell") — single source of truth.

## Work items

### 1. `training/train.py` — apply the augmentation recipe + reproducibility
Currently `run()` ignores `config.yaml`'s `augmentation:` block. Add a **pure,
torch-free** function:

```
build_train_kwargs(cfg, paths) -> dict
```

Maps Diego's recipe → Ultralytics `model.train()` knobs:
- `augmentation.fliplr` (0.5) → `fliplr`
- `augmentation.flipud` (0.0) → `flipud`
- `augmentation.brightness` (0.2) → `hsv_v`  *(Ultralytics has no literal "brightness";
  HSV-value is the standard proxy — documented inline)*
- `augmentation.rotation_deg` (10) → `degrees`

Plus core kwargs (from existing config): `data`, `imgsz=image_size`, `epochs`, `batch`,
`lr0`, `patience`, `seed`, and new: `deterministic=True`, stable
`project=<repo>/training/runs`, `name="finetune"`, `device` auto-detect (works on Colab
CUDA and laptop CPU/MPS). `run(cfg, paths)` keeps its signature and the `best.pt` copy;
it just calls `build_train_kwargs`. **Caller interface unchanged.**

### 2. `training/eval.py` — before/after baseline + handoff exports
Replace the `# TODO` baseline with small functions:
- `evaluate(weights, data_yaml, imgsz, name) -> metrics` — one `YOLO(weights).val(...)` wrapper.
- `compare_baseline_finetuned(cfg, paths) -> dict` — `val()` the **COCO-pretrained
  `training.model` (`yolo26n.pt`)** and `best.pt` on the same `data.yaml`; return
  `{baseline:{map50,map}, finetuned:{map50,map}, delta:{map50,map}}`.
- `sample_grid(weights, data_yaml, n, out)` — annotated prediction montage on N test images.
- `export_artifacts(summary, figures, out_dir)` — write `metrics.json` + `metrics.md`
  table; copy PR curve / confusion matrix / results.png / sample grid into
  `training/artifacts/`.
- `run(cfg, paths)` — orchestrates the above; logs the before/after summary. CLI unchanged.

**Honest baseline caveat:** COCO has no `hold` class, so baseline mAP is ≈0 — that IS the
transfer-learning point. COCO index 0 is `person` and Ultralytics matches by index, so a
stray detection could alias onto a hold GT and yield a tiny non-zero number. The baseline
`val()` is wrapped defensively; the notebook states plainly that pretrained COCO weights
have no hold concept (≈0 AP) and fine-tuning teaches it. No inflated delta.

### 3. `training/train_holds.ipynb` — the graded notebook (thin)
Leads with the transfer-learning narrative; imports `train`/`eval`; reads `config.yaml`.
Sections, top-to-bottom:
1. Narrative intro (markdown — the "why")
2. Setup (`pip install -r requirements.txt`, import config loader)
3. Get data (Roboflow key → `data/scripts/run_pipeline.py` → `data.yaml`; show data_card stats)
4. Peek at data (a few images with GT boxes)
5. **Baseline (before)** — COCO `yolo26n.pt` `.val()` → ≈0 mAP
6. **Fine-tune** — `train.run(cfg, paths)` → `best.pt` (long Colab-GPU cell)
7. **Eval (after)** — `eval` → mAP50/mAP50-95 + PR curve + confusion matrix + sample grid
8. **Before/after** — comparison table + bar chart (centerpiece figure)
9. **Inference demo** — `best.pt` on an empty-wall frame → static hold map (ties to Ignacio)
10. **Handoff** — where `best.pt` and `artifacts/` land

### 4. Tests + housekeeping
- `training/test_training.py` (pytest), **torch-free**: covers `build_train_kwargs`
  mapping and the `metrics.md`/`metrics.json` formatting. Both modules import
  `ultralytics` *inside* functions, so these tests run with no GPU/torch (pass this session).
- Update `training/README.md` to match the enriched eval + notebook.
- `.gitignore`: ignore `training/runs/` and binary figures under `training/artifacts/`;
  keep `training/artifacts/*.json` + `*.md` tracked (the committed metrics record).
- **Flag only** on `project_instructions/CLAUDE.md` (no edit).

## Module boundaries
- `train.py`: `build_train_kwargs` (pure) | `run` (side effects) | `main` (CLI). Unchanged signature for `run`.
- `eval.py`: `evaluate` | `compare_baseline_finetuned` | `sample_grid` | `export_artifacts` | `run` | `main`.
- `train_holds.ipynb`: orchestration + narrative only; no business logic.
- `test_training.py`: torch-free unit tests for the pure functions.

## Out of scope (this session)
- Executing real training/eval (no GPU/dataset/key) — runs on Colab.
- Any change to `data/`, `pipeline/`, or other roles' files.
- Editing `project_instructions/CLAUDE.md` (flagged only).

## Success criteria
- `build_train_kwargs` emits the documented augmentation mapping; unit test passes here.
- `eval.py` produces a before/after dict + `metrics.json`/`metrics.md` + figures (verified by
  torch-free formatting test + code review; full run validated later on Colab).
- Notebook reads top-to-bottom as dataset → baseline → fine-tune → eval → before/after →
  inference, importing the modules, leading with the transfer-learning story.
- No hardcoded paths; `image_size 640` and single class `hold` preserved.
