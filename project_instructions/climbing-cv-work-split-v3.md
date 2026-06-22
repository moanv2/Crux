# Climbing CV Work Split v3 (5 people)
Companion to **Plan v3**.
*Change from v2: Diego moves to Data (data engineering); Integration moves to Coder 2.*

- 3 coders (all on Claude Max), 2 presenters.
- Diego owns Data. Coder 1 owns Model. Coder 2 owns Integration.

---

## Hardware: none required
The project runs entirely on Colab (training) and a laptop (demo). Dataset photos and demo footage are captured on phones, with a tripod for the demo clip. No Raspberry Pi or accelerator in scope.

---

## Roster

| # | Person | Role | Owns | Primary deliverable |
|---|---|---|---|---|
| 1 | **Diego** | Data | Dataset | Clean, versioned YOLO dataset + data card |
| 2 | Coder 1 | Model | Fine-tuned detector | Frozen weights + the graded notebook core |
| 3 | Coder 2 | Integration | Combined pipeline + demo | Working pipeline + rendered demo + fallback |
| 4 | Presenter 1 | Narrative | Story half of deck | Intro/motivation slides + demo script |
| 5 | Presenter 2 | Results/Docs | Results half + notebook docs | Results slides + clean README + assembled deck |

---

## Diego — Data
**Owns:** the dataset end to end, built as a reproducible script pipeline (not just clicking around the Roboflow UI). The detector's accuracy is capped by this, so it sits upstream of the whole team.
- Pick and import the chosen Roboflow hold dataset via the SDK.
- Define the hold-type class taxonomy and lock the exact class list for everyone.
- Profile the data: image counts and class balance (e.g. catching 600 jugs vs 15 pockets).
- Clean: consistent class names, resize to 640, fix EXIF/rotation, dedupe, drop corrupt or junk images, fix obvious bad labels.
- Ingest the team's phone photos of the wall and holds, annotate in Roboflow, and merge so training data matches the demo.
- Split into train/val/test with no leakage; stratify by class where feasible.
- Set augmentations (flips, brightness, slight rotation) for gym-lighting robustness.
- Version in Roboflow and write a short data card (counts, classes, how it was built).
- Export in YOLO format with a `data.yaml`.

**Skills this builds:** ingestion, cleaning and normalization, class-imbalance handling, annotation and label QA, train/test splitting and leakage, augmentation pipelines, dataset versioning and documentation.

**Claude Max leverage:** Claude Code to write each pipeline step as a standalone, re-runnable script, plus a sanity check that counts labels per class and flags empty or oversized boxes.

**Done when:** exported dataset + `data.yaml` + data card handed to Coder 1.

**Note:** the core dataset can be done by the end of week 1. Use the freed time to deepen it (tighter label QA, more custom photos, augmentation experiments, proper versioning) rather than drifting into other roles.

## Coder 1 — Model
**Owns:** the fine-tuned hold detector and the graded notebook core.
- Build the Colab notebook: load pretrained YOLOv11, fine-tune on `data.yaml`.
- Run hyperparameter passes (epochs, image size, batch, lr, augmentation).
- Evaluate: mAP50, mAP50-95, confusion matrix, per-class metrics, sample predictions.
- Freeze and export weights (`best.pt`) for Coder 2.
- Write the transfer learning narrative in the notebook. **This is the graded centerpiece, so it leads the notebook; the combined logic sits below it as a labelled extension.**

**Claude Max leverage:** Claude Code to scaffold the Ultralytics training script, generate eval/plotting code, draft the notebook markdown, and debug training.

**Done when:** frozen weights + a documented notebook section (dataset, fine-tune, eval, inference) + metrics handed to Presenter 2.

## Coder 2 — Integration
**Owns:** the combined pipeline, overall technical integration, and the demo.
- Integrate the pretrained pose model; per-frame inference on clips.
- Capture the demo footage: an empty-wall frame + a short climbing clip, fixed camera on a tripod.
- **Static hold map:** run the detector once on the empty-wall frame and persist the map.
- Build association logic (wrist/ankle in hold box + temporal smoothing) and reach logic.
- Assemble one clean end to end pipeline with the OpenCV overlay.
- Render the combined demo and the recorded fallback video.

**Not blocked early:** pose needs no training, and the overlay scaffold + footage don't need final weights. Build all of that in week 1, then swap in Coder 1's frozen weights in week 2. With no edge track, Coder 2 carries extra margin and is the natural buffer for whichever task runs tightest.

**Claude Max leverage:** Claude Code for the fusion/association/reach code and the OpenCV overlay, and for debugging the pose + detector integration.

**Done when:** working combined pipeline + rendered demo + recorded fallback.

## Presenter 1 — Narrative
**Owns:** the story half of the deck and live-demo narration.
- Problem framing: why CV for climbing, the "which holds / reach" idea, real-world hook (route reading, training boards, coaching).
- High-level method narrative (transfer learning, two-model fusion, static hold map) for a general audience.
- Intro, motivation, and approach-overview slides.
- Script the live-demo narration and coordinate timing with Coder 2.

**Claude leverage:** draft and refine the narrative, generate speaker notes; Claude can build the deck (pptx).

**Done when:** narrative slides + demo script + speaker notes.

## Presenter 2 — Results / Docs
**Owns:** the results half of the deck and the notebook documentation.
- Turn Coder 1's metrics into slides: mAP, confusion matrix, sample detections, combined-pipeline screenshots.
- Limitations and future work (overhang pose, grading, move classification).
- Notebook README / documentation pass so the submitted notebook reads cleanly.
- Assemble the final deck with Presenter 1 and drop in the demo/fallback video.

**Claude leverage:** write results commentary, structure limitations, polish the README; Claude can build slides (pptx).

**Done when:** results slides + clean notebook README + assembled final deck.

---

## Handoffs
| From | To | What | When |
|---|---|---|---|
| Diego | Coder 1 | Dataset export + data card | End of week 1 |
| Coder 1 | Coder 2 | Frozen weights (`best.pt`) | Early week 2 |
| Coder 1 | Presenter 2 | Metrics + plots | Early/mid week 2 |
| Coder 2 | Presenters | Rendered demo + fallback video | Mid/late week 2 |
| Presenter 1 | Presenter 2 | Narrative slides for assembly | Late week 2 |

## Everyone starts day 1 in parallel (nobody waits)
- **Diego:** dataset setup + the scripted data pipeline.
- **Coder 1:** training scaffold on the raw Roboflow set before custom photos land.
- **Coder 2:** pose pipeline + overlay scaffold + capture the demo footage.
- **Presenters:** problem framing, deck skeleton, README template.

## Shared conventions
- One repo or Drive folder; agree the exact class names up front.
- Fixed image size (640) across training, eval, and inference.
- A single agreed location for the frozen weights so everyone pulls the same file.
