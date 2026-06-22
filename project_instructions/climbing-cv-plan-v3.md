# Climbing CV Plan v3
**Combined hold detection + climber pose**
*Change from v2: role labels updated. Diego on Data, Coder 1 on Model, Coder 2 on Integration.*

- **Notebook due:** July 1, 23:59 Madrid · **Live demo + presentation:** July 2
- **Stack:** Ultralytics YOLO, Google Colab, Roboflow Universe
- **Compute:** Colab for training, a laptop for the demo · **Capture:** phones (tripod for the demo clip)
- **Team:** Diego (data), Coder 1 (model), Coder 2 (integration), 2 presenters

---

## Goal
A vision system that scans a climbing wall, detects and classifies holds, tracks the climber's body, and infers **which holds are in use** plus the **reach to the next hold**.

> Graded core artifact = the fine-tuned YOLO **hold detector** (transfer learning). The pose + combined logic is an applied extension and the demo wow factor.

## Scope
**In:** multi-class hold detection (fine-tuned), pose overlay (pretrained), hold-in-use association, next-hold reach.
**Out:** V-grade prediction, move classification, overhang-optimised pose, multi-camera, any edge or hardware deployment.
**Stretch (optional):** instance segmentation masks instead of boxes; a near real time laptop webcam demo instead of a pre-rendered clip.

---

## Key design decision: static hold map
Detect holds **once on an empty-wall reference frame** to build a fixed hold map. Then run **only pose per frame** and associate the climber against that static map.

- **Why:** the climber's body occludes holds exactly when a hand or foot is on them, so per-frame hold detection drops the holds you most care about. Scanning the empty wall first removes the occlusion problem entirely.
- **Bonus:** clean demo arc. Scan the wall, climber enters, watch the contacts light up.
- **Requirement:** a static camera (phone or webcam on a tripod) and a fixed wall/board.

## Architecture
```
empty-wall frame ──► Hold detector (custom, run ONCE) ──► static hold map ─┐
                                                                            ├─► association ─► reach ─► overlay
each frame       ──► Pose model (pretrained, per frame) ─► 17 keypoints ───┘
```
Both models run on the same image, so outputs already share pixel space. No coordinate alignment needed.

## Data
- **Holds:** start with **one** dataset to avoid label-merge pain. Candidates: *Climbing Holds and Volumes* (type labels: jug/crimp/sloper/pocket + volumes) or *GDSCMoonless Rock-Climbing-Hold-Detection* (~4.3k images). Add ~50 to 150 of your own photos (phones), annotate in Roboflow, export in YOLO format.
- **Pose:** none needed (pretrained COCO 17-keypoint).
- **Demo footage:** a short climbing clip of a teammate plus a matching empty-wall frame, same fixed camera. A built board is optional; a real gym wall works too.

## Models
- **Hold detector:** YOLOv11n or s, **detection (boxes)**, multi-class by hold *type*. Segmentation = optional polish.
- **Pose:** yolov11n-pose / yolov8n-pose, pretrained.
- **Notebook centerpiece (graded):** dataset → pretrained YOLO → fine-tune → eval (mAP50, mAP50-95, confusion matrix, sample predictions) → inference. Combined logic sits below as a clearly labelled extension.

## Integration logic (the "combined" part)
- **Association:** a hold is "in use (left/right hand)" if the matching wrist keypoint sits inside the hold box (expanded by a small margin) for ≥ N consecutive frames. Same for ankles → feet. Colour contacts (hands green, feet blue).
- **Reach:** from the higher hand, find the nearest *unused* hold above it, draw a line, label the distance. Normalise by the climber's shoulder→wrist pixel length for a rough body-scale estimate.
- **Graceful degradation:** if association gets noisy, still show hold map + skeleton + reach line independently. Never let the fancy layer break the basic visual.

---

## Timeline & owners

### Week 1 (Jun 17–23) — Foundations
| Owner | Task |
|---|---|
| Diego (data) | Lock dataset + class list, build the scripted data pipeline, export YOLO. Coordinate dataset photo capture (phones). |
| Coder 1 (model) | Baseline fine-tune running in Colab (rough mAP, end to end). |
| Coder 2 (integration) | Pose model running on a sample clip. Build the overlay scaffold. Capture the demo footage (empty-wall frame + climbing clip on a tripod). |
| Presenter 1 | Draft problem framing + slide skeleton. |
| Presenter 2 | Stand up notebook README + eval reporting template. |

### Week 2 (Jun 24–30) — Integration & polish
| Owner | Task |
|---|---|
| Diego | Add custom photos, augmentation, finalise split + data card. |
| Coder 1 | Tune detector, full eval (mAP, confusion matrix), **freeze weights**. |
| Coder 2 | Build static hold map + association + reach; integrate the end to end pipeline; render the combined demo + recorded fallback. |
| All | End to end demo dry run. |
| Presenter 1 + 2 | Build deck, write up results, rehearse. |

- **Jul 1** — Notebook frozen and submitted (23:59 Madrid).
- **Jul 2** — Live demo + presentation. Recorded fallback ready.

---

## Risk register
| Risk | Mitigation |
|---|---|
| Combined integration overruns | Spine (detector) passes on its own; layer combined on top; degrade gracefully. Coder 2 has more margin now the edge track is gone. |
| In-use holds occluded by the body | Static hold map (scan empty wall once). |
| Pose fails on extreme / overhang poses | Demo on slab or vertical; note overhang as a stated limitation. |
| Live demo flakiness | Run on pre-rendered clips + keep the recorded fallback. |
| Label-scheme mismatch when merging datasets | Start with a single dataset. |

## Demo script
1. **Empty-wall frame** → detector scans → hold map (the transfer learning result, on the laptop).
2. **Climbing clip** → skeleton overlay.
3. **Combined** → contacts light up (hands/feet), next-hold reach line + distance.
4. **Fallback** → pre-rendered video of steps 1 to 3.

## Deliverables checklist
- [ ] Colab notebook: fine-tuned hold detector + eval + inference *(graded core)*
- [ ] Combined pipeline: hold map + pose + association + reach
- [ ] Custom dataset (Roboflow) + own photos
- [ ] Trained weights
- [ ] Demo footage (empty frame + climbing clip)
- [ ] PowerPoint
- [ ] Recorded fallback demo video

## Open decisions (tell me to lock any)
1. **Hold dataset:** single (which?) vs merge.
2. **Detector output:** detection boxes vs segmentation masks.
3. **Demo footage:** a real gym clip vs a simple low home board.
