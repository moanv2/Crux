# Crux

Final project for Computer Vision @ IE School of Science and Technology.

**What we're building:** a computer-vision **climbing coach**. We fine-tune a YOLO
detector to find every hold on a wall (the graded core), track the climber's body
with pretrained pose estimation, and fuse the two into a **quantified climb
debrief** — which holds were used, where the crux is (time-per-move), and feet
cuts — feedback you can't get just watching yourself.

The graded core artifact is the fine-tuned YOLO **hold detector** (single-class,
transfer learning). The pose + analysis layer is the applied extension and the
demo wow factor.

**Deadlines:** notebook due July 1 (23:59 Madrid); live demo + presentation July 2.

## Tech stack
- Python 3.11.
- Ultralytics YOLO (YOLO26), Roboflow SDK, OpenCV, NumPy.
- Training in Google Colab (GPU). Inference and the demo on a laptop.
- Capture on phones, tripod for the demo clip. Demo on a **vertical / slab** wall —
  pose estimation is unreliable on steep overhangs.

## Approach: static hold map + pose fusion
Detect holds **once on an empty-wall reference frame** to build a fixed hold map,
then run **only pose per frame** and analyze the climber against that static map.

Why: the body occludes holds exactly when a hand or foot is on them, so per-frame
hold detection drops the holds you most care about. Scanning the empty wall first
removes the occlusion problem and gives a clean demo arc — scan the wall, climber
enters, watch the contacts light up and the debrief build.

```
empty-wall frame --> Hold detector (fine-tuned YOLO26, run ONCE) --> static hold map -+
                                                                                       +-> analysis --> debrief
each frame       --> Pose model (pretrained, per frame) --> 17 keypoints --------------+   (contacts, crux /
                                                                                            time-per-move, feet cuts)
```
Both models run on the same 640 image, so outputs share pixel space — no alignment.

## Scope
- **In:** single-class hold detection (fine-tuned, graded core); pretrained pose
  overlay; **climb analysis** = contacts (holds used + order), crux via
  time-per-move, and feet cuts.
- **Stretch (only if the core is solid):** reach / extension per move (normalized by
  shoulder→wrist span); hold-**type** classification → route difficulty profile;
  **color** → route finder.
- **Out:** V-grade prediction, hips-off-wall efficiency (needs a side camera),
  move classification, multi-camera, any edge or hardware deployment.

## Repo layout
```
data/        scripts/ (download, clean, stats, split, export, sanity_check, run_pipeline), data.yaml, data_card.md
training/    training notebook, train.py, eval.py
models/      best.pt  (the one frozen detector)
pipeline/    holds_map.py, pose.py, analyze.py, overlay.py, run.py
demo/        footage references, rendered videos
config.yaml  paths + thresholds
project_instructions/  plan, work split, CLAUDE.md contract
README.md
```

## Shared conventions
- **Image size is 640** for training, eval, and inference. Never change it.
- **Single class: `hold`** — use that exact name everywhere. (Hold *types* are a future stretch.)
- **Dataset format:** YOLO, described by `data/data.yaml`.
- **Frozen detector weights live at `models/best.pt`** — one agreed location, fixed filename.
- **No hardcoded paths.** Data and weights paths live in `config.yaml`, so code runs unchanged in Colab and locally.
- Modular, importable `.py` modules with docstrings and type hints. Small commits, one branch per role.

## Team & roles

| Person | Role | Owns | Primary deliverable |
|---|---|---|---|
| **Diego** | Data | Dataset | Clean single-class YOLO dataset + data card |
| Jan | Model | Fine-tuned detector | Frozen weights + the graded notebook core |
| Ignacio | Integration | Climbing-coach pipeline + demo | Working pipeline + rendered debrief demo + fallback |
| Claudia | Narrative | Story half of deck | Intro/motivation slides + demo script |
| Dalton | Results / Docs | Results half + notebook docs | Results slides + clean README + assembled deck |

### Diego: Data
Reproducible YOLO dataset under `data/scripts/`: `download` → `clean` (resize 640,
**remap all boxes to the single class `hold`**, dedupe, drop corrupt) → `stats` →
`split` → `export`, plus `sanity_check`. ~425 images from the Climbing Replica Test fork (single-class).
Records the Roboflow version in `data_card.md`. **Hands off** `data.yaml` + `data_card.md` to Jan.

### Jan: Model
Fine-tunes pretrained **YOLO26** on `data.yaml` (single-class `hold`) in
`train.py` / the notebook. Eval (`eval.py`): mAP50, mAP50-95, PR curve, a
sample-prediction grid, and a **before/after** (COCO-pretrained baseline vs
fine-tuned) to show the transfer-learning gain. Exports frozen weights to
`models/best.pt`. The notebook leads with the transfer-learning story. **Hands off**
`best.pt` to Ignacio; figures/metrics to Dalton.

### Ignacio: Integration (the climbing coach)
Builds the pipeline under `pipeline/`: static hold map (`holds_map.py`), per-frame
pose (`pose.py`), and the **analysis layer** (`analyze.py`):
- **contacts** — wrist/ankle inside a hold box (+ margin, + N-frame smoothing) → holds used + order;
- **crux** — body-motion magnitude per frame → segment moves → longest pause = crux / time-per-move;
- **feet cuts** — both ankles off all holds → count + timestamps.

Renders the debrief overlay + card (`overlay.py`, `run.py`) → demo video + recorded
fallback. Captures the footage (empty-wall frame + climbing clip, tripod, vertical
wall). Degrades gracefully: still shows hold map + skeleton if analysis is noisy.
**Hands off** the rendered demo + fallback to the presenters.

### Claudia: Narrative
Problem framing ("a debrief you can't see yourself") and high-level method
narrative (transfer learning + pose fusion); intro/motivation/approach slides and
the live-demo script. Coordinates demo timing with Ignacio.

### Dalton: Results / Docs
Turns Jan's metrics into results slides (mAP, PR curve, before/after, sample
detections) and the debrief-output slides; writes limitations/future (2D-pose
limits, overhang, hips-needs-side-camera, hold types as future work); polishes the
README; assembles the final deck with the demo video.

## Timeline (~10 days)

**Days 1–4 — Spine (each works in parallel, nobody waits)**
- Diego: ingest the ~425-image single-class dataset, clean/split/export, hand off `data.yaml`.
- Jan: baseline + first fine-tune running end to end in Colab (rough mAP, before/after).
- Ignacio: pose running on a sample clip; capture the empty-wall frame + climbing clip; overlay scaffold.
- Claudia: problem framing + slide skeleton. Dalton: notebook README + eval template.

**Days 5–8 — Integration & polish**
- Jan: tune detector, full eval, **freeze `best.pt`**.
- Ignacio: static hold map + `analyze.py` (contacts, crux, feet cuts) + render the debrief demo + fallback.
- All: end-to-end dry run.
- Claudia + Dalton: build deck, write up results, rehearse.

**Jul 1** notebook frozen and submitted. **Jul 2** live demo + presentation, recorded fallback ready.

## Handoffs
| From | To | What | When |
|---|---|---|---|
| Diego | Jan | `data.yaml` + `data_card.md` | Day 3–4 |
| Jan | Ignacio | Frozen weights (`best.pt`) | Day 5–6 |
| Jan | Dalton | Metrics + plots | Day 5–6 |
| Ignacio | Presenters | Rendered debrief demo + fallback | Day 7–8 |
| Claudia | Dalton | Narrative slides for assembly | Day 8–9 |
