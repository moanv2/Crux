# Crux

Final project for Computer Vision @ IE School of Science and Technology.

**What we're building:** a vision system that scans a climbing wall, detects and
classifies holds (fine-tuned YOLO, the graded core), tracks the climber with pose
estimation, and infers which holds are in use plus the reach to the next hold.

The graded core artifact is the fine-tuned YOLO **hold detector** (transfer learning).
The pose and combined logic is an applied extension and the demo wow factor.

**Deadlines:** notebook due July 1 (23:59 Madrid); live demo + presentation July 2.

## Tech stack
- Python 3.11.
- Ultralytics YOLO (YOLOv26), Roboflow SDK, OpenCV, NumPy.
- Training in Google Colab (GPU). Inference and the demo on a laptop.
- Capture on phones, with a tripod for the demo clip.

## Approach: static hold map
Detect holds **once on an empty-wall reference frame** to build a fixed hold map,
then run **only pose per frame** and associate the climber against that static map.

Why: the climber's body occludes holds exactly when a hand or foot is on them, so
per-frame hold detection drops the holds you most care about. Scanning the empty
wall first removes the occlusion problem entirely. It also gives a clean demo arc:
scan the wall, climber enters, watch the contacts light up. It requires a static
camera (phone or webcam on a tripod) and a fixed wall.

```
empty-wall frame --> Hold detector (custom, run ONCE) --> static hold map -+
                                                                           +-> association -> reach -> overlay
each frame       --> Pose model (pretrained, per frame) -> 17 keypoints ---+
```
Both models run on the same 640 image, so outputs already share pixel space.
No coordinate alignment needed.

## Scope
- **In:** multi-class hold detection (fine-tuned), pose overlay (pretrained),
  hold-in-use association, next-hold reach.
- **Out:** V-grade prediction, move classification, overhang-optimised pose,
  multi-camera, any edge or hardware deployment.
- **Stretch (optional):** instance-segmentation masks instead of boxes; a near
  real-time laptop webcam demo instead of a pre-rendered clip.

## Repo layout
```
data/        scripts/ (download, clean, ingest_custom, stats, split, export), data.yaml, data_card.md
training/    training notebook, train.py, eval.py, configs
models/      best.pt  (the one frozen detector)
pipeline/    holds_map.py, pose.py, associate.py, reach.py, overlay.py, run.py
demo/        footage references, rendered videos
config.yaml  paths + thresholds
project_instructions/  plan, work split, CLAUDE.md contract
README.md
```

## Shared conventions
- **Image size is 640** for training, eval, and inference. Never change it.
- **Class list** is locked by Diego; use the exact same names *and order* everywhere.
- **Dataset format:** YOLO, described by `data/data.yaml`.
- **Frozen detector weights live at `models/best.pt`**, one agreed location, fixed filename.
- **No hardcoded paths.** Data and weights paths live in `config.yaml`, so code runs
  unchanged in Colab and locally.
- Modular, importable `.py` modules with docstrings and type hints.
- Small, descriptive commits. One branch per role.

## Team & roles

| Person | Role | Owns | Primary deliverable |
|---|---|---|---|
| **Diego** | Data | Dataset | Clean, versioned YOLO dataset + data card |
| Coder 1 | Model | Fine-tuned detector | Frozen weights + the graded notebook core |
| Coder 2 | Integration | Combined pipeline + demo | Working pipeline + rendered demo + fallback |
| Presenter 1 | Narrative | Story half of deck | Intro/motivation slides + demo script |
| Presenter 2 | Results / Docs | Results half + notebook docs | Results slides + clean README + assembled deck |

### Diego: Data
Clean, versioned, documented YOLO dataset built as a reproducible script pipeline
under `data/scripts/`: `download` then `clean` then `ingest_custom` then `stats`
then `split` then `export`, plus a `sanity_check`. Locks the class list; records
the Roboflow version in `data_card.md`. **Hands off** `data.yaml` + `data_card.md` to Coder 1.

### Coder 1: Model
Fine-tunes pretrained YOLOv11 on `data.yaml` (`train.py` / notebook), evaluates
(mAP50, mAP50-95, confusion matrix, per-class metrics, sample grid via `eval.py`),
and exports frozen weights to `models/best.pt`. The notebook leads with the
transfer-learning story. **Hands off** `best.pt` to Coder 2; figures/metrics to Presenter 2.

### Coder 2: Integration
Builds the combined pipeline under `pipeline/`: static hold map (`holds_map.py`),
per-frame pose (`pose.py`), wrist/ankle-in-box association (`associate.py`), reach
to the next hold (`reach.py`), and the OpenCV overlay + runner (`overlay.py`,
`run.py`). Captures the demo footage. Degrades gracefully: still shows hold map +
skeleton + reach if association gets noisy. **Hands off** the rendered demo + fallback to the presenters.

### Presenter 1: Narrative
Problem framing and high-level method narrative; intro/motivation/approach slides
and the live-demo script. Coordinates demo timing with Coder 2.

### Presenter 2: Results / Docs
Turns Coder 1's metrics into results slides, writes the limitations/future-work
section, polishes the notebook README, and assembles the final deck with the demo video.

## Timeline

**Week 1 (Jun 17 to 23) — Foundations**
- Diego: lock dataset + class list, build the scripted data pipeline, export YOLO, coordinate photo capture.
- Coder 1: baseline fine-tune running end to end in Colab (rough mAP).
- Coder 2: pose model running on a sample clip; overlay scaffold; capture demo footage.
- Presenter 1: draft problem framing + slide skeleton.
- Presenter 2: stand up notebook README + eval reporting template.

**Week 2 (Jun 24 to 30) — Integration & polish**
- Diego: add custom photos, augmentation, finalise split + data card.
- Coder 1: tune detector, full eval, freeze weights.
- Coder 2: static hold map + association + reach; integrate end to end; render demo + fallback.
- All: end to end demo dry run.
- Presenter 1 + 2: build deck, write up results, rehearse.

**Jul 1** notebook frozen and submitted. **Jul 2** live demo + presentation, recorded fallback ready.

## Handoffs
| From | To | What | When |
|---|---|---|---|
| Diego | Coder 1 | Dataset export + data card | End of week 1 |
| Coder 1 | Coder 2 | Frozen weights (`best.pt`) | Early week 2 |
| Coder 1 | Presenter 2 | Metrics + plots | Early/mid week 2 |
| Coder 2 | Presenters | Rendered demo + fallback video | Mid/late week 2 |
| Presenter 1 | Presenter 2 | Narrative slides for assembly | Late week 2 |
