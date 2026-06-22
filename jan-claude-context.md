# Jan — Project Context & Your Job (Model)

_Full briefing for Jan: what we're building, **why** the dataset looks the way it does, and **exactly** what you need to do. ~5-minute read. Written by Diego (Data). Ping Diego with anything unclear._

---

## TL;DR
- **Project:** a computer-vision **"climbing coach"** — detect the holds on a wall, track the climber's body, and turn a climb video into a **debrief** (which holds were used, where the crux is, feet cuts).
- **Graded core = your part:** fine-tune **YOLO26** to detect holds (a single class, `hold`) and evaluate it. That's the transfer-learning centerpiece.
- **Dataset is ready:** Roboflow `diego-alfaro-s-workspace / climbing-replica-test-bdtpi`, **version 3** (~425 images, single class `hold`, 640×640). Already wired into `config.yaml`.
- **Your output:** `models/best.pt` + eval metrics/figures → hand to Ignacio (Integration) and Dalton (results slides).
- **Deadlines:** notebook **Jul 1** (23:59 Madrid), live demo **Jul 2**.

**Team:** Diego = Data, **Jan = Model (you)**, Ignacio = Integration, Claudia = Narrative, Dalton = Results/Docs.

---

## 1. What we're building — the "climbing coach"
The system has two models running on the **same 640px frame**:
- **Hold detector (yours, fine-tuned):** run **once** on an empty-wall photo → a static map of where every hold is.
- **Pose model (pretrained, Ignacio's):** per frame → 17 body keypoints.

Fuse them → a **debrief**: which holds the hands/feet are on (*contacts*), where the climber struggled (*crux* = longest pause / time-per-move), and *feet cuts*. The value is that **inference layer** — feedback a climber can't get just watching themselves.

Two deliverables:
1. **The graded notebook = the fine-tuned detector + eval (transfer learning). ← YOUR job.**
2. The live demo = pose fusion + debrief overlay (Ignacio).

---

## 2. Why single class `hold`? (and why not jug/crimp/sloper, or colours)
Everyone asks this, so here's the honest reasoning — please don't reintroduce types/colours:

- **We tried multi-class by hold *type* first, and it died on data.** There is **no pre-typed climbing-hold dataset at scale**. We checked the big ones — GDSCMoonless (~4k imgs) and Climbing Replica Test (3.1k imgs / 221k boxes) — and **every box is just `hold`**. The only *typed* sets are tiny (≤100 imgs) and messy (they mix `foothold`/`edge`/`rail`, i.e. function + overlapping classes). Labeling types ourselves = thousands of boxes by hand, it's **subjective** (climbers disagree on jug-vs-crimp), and the label noise would **cap the detector's accuracy**.
- **We considered colour → route, and spray-wall route suggestion.** Colour is redundant on colour-coded walls and meaningless on spray walls; route-suggestion has no ground truth, so it **can't be evaluated** for the graded notebook.
- **The real insight:** a *per-hold label* (hold / type / colour) is never the value — a climber can already see those. **Value comes from inference** over the detections + the body (the debrief). So the detector only needs to find *where* the holds are.
- **Therefore: single class `hold`.** It's robust, the labels are free and clean, and it keeps the transfer-learning story crisp — COCO has no "hold" class, so fine-tuning produces a clear **before/after mAP gain** (your notebook's money shot). Types/colours are an explicit **future stretch**, not the core.

---

## 3. The dataset — what it is
- Roboflow project **`climbing-replica-test-bdtpi`** (a fork of "Climbing Replica Test"), **version 3**: ~425 images, single class `hold`, preprocessing = **Auto-Orient + Resize 640×640**, **no augmentation** (we augment during training, not in the dataset).
- After our cleaning pipeline runs it's **~417 images / 28,704 `hold` boxes**, split 70/20/10 (≈291 / 83 / 43).
- **It is NOT in git** — images are large and live in Roboflow. Git only holds the scripts + `config.yaml` + the data card (`data/data_card.md`). You pull the images yourself → see §5.

---

## 4. Your exact job (Model)
1. **Get the dataset** (§5) → you'll have `data/data.yaml` (`nc: 1, names: ['hold']`).
2. **Fine-tune:** `python training/train.py` — reads `config.yaml` (`training.model = yolo26n.pt`, `image_size = 640`, 80 epochs, batch 16, lr0 0.01) and copies the best checkpoint to **`models/best.pt`**.
3. **Evaluate:** `python training/eval.py` → mAP50, mAP50-95 (+ PR curve, sample-prediction grid). **Also run `model.val()` on the plain COCO-pretrained `yolo26n.pt` for a before/after** — that's the transfer-learning result the grade hinges on.
4. **Build the graded Colab notebook** around `train.py` + `eval.py`: dataset → pretrained → fine-tune → eval → inference, leading with the transfer-learning narrative.
5. **Hand off:** `models/best.pt` → Ignacio; metrics + figures → Dalton.

More detail in **`training/README.md`**.

---

## 5. How to get the dataset
### 🅰 Roboflow — recommended (reproducible, already wired)
```bash
git clone https://github.com/moanv2/Crux && cd Crux
pip install -r requirements.txt
# get a FREE Roboflow key: roboflow.com → Settings → API → Private API Key
export ROBOFLOW_API_KEY="your_key"          # Colab: set it as an env var / secret
python data/scripts/run_pipeline.py         # pulls v3 → data/data.yaml + data/processed/
```
- The project is public, so your own key can pull it. **If you get an access error, ask Diego to add you to the Roboflow workspace.**
- This regenerates `data/data.yaml` with **your** machine's absolute path — correct for wherever you run (so don't copy someone else's `data.yaml`).

### 🅱 Google Drive — backup (no Roboflow key needed)
- Diego drops **`dataset_v3.zip`** (~35 MB = `data/processed/` + `data.yaml`) in the team Drive.
- Unzip it, then edit `data.yaml`'s `path:` to point at your unzipped folder.

Either way, you train on `data/data.yaml`.

---

## 6. Conventions & gotchas (don't trip on these)
- **Image size 640 everywhere** (`config.yaml image_size`). Never change it.
- **Single class `hold`, `nc: 1`.** The confusion matrix is trivial with one class — lean on **mAP50 / mAP50-95 + PR curve + the before/after baseline + qualitative sample predictions** for the eval section.
- **Frozen weights live at `models/best.pt`** — one agreed location (gitignored; share via Drive for Ignacio).
- **`data.yaml` uses an absolute `path:` on purpose** — Ultralytics resolves a *relative* path against its own `datasets_dir`, which breaks training. It's gitignored and regenerated per machine, so **run the pipeline locally** instead of reusing someone's `data.yaml`.
- **No augmentation in the Roboflow version** — apply it in training (Ultralytics augments online; knobs are in `config.yaml augmentation:`).
- Everything is **config-driven** (`config.yaml`) — no hardcoded paths, so the same code runs in Colab and locally.

---

## 7. Quick map of the repo
| Path | What |
|---|---|
| `config.yaml` | Single source of truth — Roboflow coords, image size, classes, training knobs |
| `data/scripts/run_pipeline.py` | Get the dataset (download → clean → split → export) |
| `data/data_card.md` | Dataset stats (counts, source, version) |
| `training/train.py` · `eval.py` · `README.md` | **Your work** |
| `models/best.pt` | **Your output** (→ Ignacio) |
| `pipeline/` | Ignacio's integration (consumes `best.pt`) |
| `project_instructions/` | Original plan + work split |

**Deadlines:** notebook **Jul 1**, demo **Jul 2**. Questions → **Diego (Data)**.
