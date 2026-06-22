# Pipeline — Ignacio (Integration / the climbing coach)

Fuse the hold detector + pose into a **climb debrief**: which holds are in use,
the **crux** (time-per-move), and **feet cuts**.

## Modules (build in this order; each is stubbed with signatures + config wiring)
1. `holds_map.py` — run `models/best.pt` **once** on the empty-wall frame → static hold map (boxes) → JSON.
2. `pose.py` — per-frame pretrained pose (`yolo11n-pose`) → 17 keypoints per frame.
3. `analyze.py` — the coach logic, from (hold map + keypoints), thresholds from config `pipeline:`:
   - `contacts(...)` — wrist/ankle inside a hold box (+margin, +N frames) → holds used + order.
   - `crux(...)` — body-motion → segment moves → longest pause = crux / time-per-move.
   - `feet_cuts(...)` — both ankles off all holds → count + timestamps.
4. `overlay.py` + `run.py` — draw the debrief (skeleton + contacts + card) and render the demo + fallback.

## Inputs / outputs
- **In:** `models/best.pt` (Jan); demo footage in `demo/` (empty-wall frame + climbing clip, tripod, **vertical/slab** wall so pose is reliable).
- **Out:** rendered demo video + recorded fallback → presenters.

## Run
```bash
python pipeline/run.py --empty demo/empty.jpg --clip demo/climb.mp4
```

## Conventions (do not break)
- Detector and pose both run on the **same 640 pixel space** — no rescaling between them.
- Read all thresholds from `config.yaml` (`pipeline.association_margin_px`, `min_contact_frames`).
- **Graceful degradation:** if `analyze` gets noisy, still render hold map + skeleton so the demo never breaks.
