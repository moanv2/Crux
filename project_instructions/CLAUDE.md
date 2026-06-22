# CLAUDE.md — Climbing CV Project

This file is the team's behavioral contract for Claude Code. It is loaded automatically at the start of every Claude Code session in this repo. Keep it specific and concise: every line should change how the agent acts.

**What we're building:** a vision system that detects and classifies climbing holds on a wall (fine-tuned YOLO, the graded core), tracks the climber with pose estimation, and infers which holds are in use plus the reach to the next hold.

**Deadlines:** notebook due July 1 (23:59 Madrid); live demo + presentation July 2.

---

## Tech stack
- Python 3.11.
- Ultralytics YOLO (YOLO26), Roboflow SDK, OpenCV, NumPy.
- Training in Google Colab (GPU). Inference and the demo on a laptop.

## Shared conventions (apply to everyone, no exceptions)
- **Image size is 640** for training, eval, and inference. Never change it.
- **Class list (LOCKED):** `jug, pocket, pinch, sloper, crimp, volume` (indices 0–5) — use the exact same names *and order* everywhere. Manually annotated by hold type; see `data/annotation_guide.md`.
- **Dataset format:** YOLO, described by `data/data.yaml`.
- **Frozen detector weights live at `models/best.pt`** — one agreed location, fixed filename.
- The detector and the pose model both run on the same 640 frame. Keep a single pixel space; do not rescale between them.
- **No hardcoded paths.** Put data and weights paths in `config.yaml` and read from there, so code runs unchanged in Colab and locally.
- **Modular code:** importable `.py` modules with functions, docstrings, and type hints. Do not bury logic in one giant notebook cell.
- Small, descriptive commits. One branch per role.

## Working with Claude Code
- Follow the Shared Conventions above in everything you generate.
- Keep changes scoped to the current role's directory.
- Add a quick sanity check or test for each script you write.
- This file is shared; do not add personal or machine-specific paths here (use `config.yaml` or a local, uncommitted file).

## Repo layout
```
data/        scripts/ (download, clean, ingest_custom, stats, split, export), data.yaml, data_card.md
training/    training notebook, train.py, eval.py, configs
models/      best.pt  (the one frozen detector)
pipeline/    holds_map.py, pose.py, associate.py, reach.py, overlay.py, run.py
demo/        footage references, rendered videos
config.yaml  paths + thresholds
README.md, CLAUDE.md
```

---

## Roles

### Diego — Data
Goal: a clean, versioned, documented YOLO dataset, built as a reproducible script pipeline.
Build each as a standalone, re-runnable script under `data/scripts/`:
- `download.py` — pull the chosen Roboflow dataset via the SDK.
- `clean.py` — normalize class names, resize to 640, fix EXIF/rotation, dedupe, drop corrupt or empty.
- `ingest_custom.py` — add the team's phone photos (annotated in Roboflow), merge.
- `stats.py` — class balance + counts, written to `data_card.md`.
- `split.py` — train/val/test, no leakage, stratify by class where feasible.
- `export.py` — YOLO format + `data.yaml`.
Conventions: lock the class list first and post it in Shared Conventions above. The whole pipeline runs end to end with one command. Record the Roboflow dataset version ID in `data_card.md`.
Hand off: `data.yaml` + `data_card.md` → Coder 1.
Ask Claude Code to: write a sanity check that counts labels per class and flags empty or oversized boxes.

### Coder 1 — Model
Goal: the fine-tuned hold detector + the graded notebook core.
Build:
- `training notebook` / `train.py` — load pretrained YOLO26n/s, fine-tune on `data.yaml`.
- `eval.py` — mAP50, mAP50-95, confusion matrix, per-class metrics, sample-prediction grid; save figures.
- export frozen weights to `models/best.pt`.
Conventions: same 640, same class list and order; seed runs for reproducibility. The notebook leads with the transfer learning story; the combined logic is a clearly labelled section below.
Inputs: `data.yaml` + `data_card.md` (Diego). Hand off: `models/best.pt` → Coder 2; figures/metrics → Presenter 2.
Ask Claude Code to: parameterize epochs/img/batch via `config.yaml` and auto-generate the confusion matrix + sample grid.

### Coder 2 — Integration
Goal: the combined pipeline + the demo.
Build under `pipeline/`:
- `holds_map.py` — run the detector once on an empty-wall frame, persist the hold map (boxes + classes as JSON).
- `pose.py` — per-frame pose (pretrained `yolo*-pose`) → 17 keypoints.
- `associate.py` — wrist/ankle inside a hold box (+ margin, + N-frame smoothing) → holds in use.
- `reach.py` — nearest unused hold above the higher hand; normalize by shoulder→wrist length.
- `overlay.py` + `run.py` — draw everything, process a clip end to end, export the rendered demo + fallback.
Also: capture the demo footage (empty-wall frame + climbing clip, fixed tripod).
Conventions: consume `models/best.pt` + the shared class list; keep the 640 pixel space; graceful degradation — still render holds + skeleton + reach if association is noisy.
Inputs: `models/best.pt` (Coder 1). Hand off: rendered demo + fallback → presenters.
Ask Claude Code to: load both models once and loop frames; make association thresholds config-driven; include a `--fallback` render path.

### Presenter 1 — Narrative
Mostly slides, not code. Use Claude (pptx) for the deck and speaker notes.
Deliver: intro/motivation/approach slides + the live-demo script. Coordinate demo timing with Coder 2.

### Presenter 2 — Results / Docs
Deliver: results slides from Coder 1's figures; the repo `README.md`; the assembled final deck (with Presenter 1); the demo video dropped in.
Inputs: figures/metrics (Coder 1), rendered demo (Coder 2). Use Claude (pptx) for slides and Claude Code for the README.

---

## Fill these in before coding
- [x] Final class list (names + order) — `jug, pocket, pinch, sloper, crimp, volume` (Diego)
- [x] Chosen Roboflow dataset — fork of GDSCMoonless rock-climbing-hold-detection, re-typed to the 6 classes + custom photos
- [ ] Repo host + branch naming
