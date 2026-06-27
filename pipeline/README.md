# Pipeline — Ignacio (Integration)

Fuse the hold detector + pose: detect which holds are **in use** and the **reach
to the next hold**, then render the demo overlay.

## Modules (build in this order; each wired to `config.yaml`)
1. `holds_map.py` — run `models/best.pt` **once** on the empty-wall frame → static hold map (boxes) → JSON.
2. `pose.py` — per-frame pretrained pose (`yolo11n-pose`) → 17 keypoints per frame.
3. `associate.py` — `associate(...)`: wrist/ankle inside a hold box (+margin, +N-frame debounce) → holds in use per frame.
4. `reach.py` — `reach_per_frame(...)`: nearest *unused* hold above the higher hand, with distance normalized by shoulder→wrist length.
5. `overlay.py` + `run.py` — draw holds + skeleton + contacts + reach line, and render the demo + fallback.

## Inputs / outputs
- **In:** `models/best.pt` (Coder 1); demo footage in `demo/` (empty-wall frame + climbing clip, tripod, **vertical/slab** wall so pose is reliable).
- **Out:** rendered demo video + recorded fallback → presenters.

## Run
```bash
python pipeline/run.py --empty demo/empty.jpg --clip demo/climb.mp4 --out demo/out.mp4
```

## Conventions (do not break)
- Detector and pose both run on the **same pixel space** — the empty-wall frame and
  the clip come from the **same fixed camera (same resolution)**, so boxes and
  keypoints align with no rescaling between them.
- Read all thresholds from `config.yaml` (`pipeline.association_margin_px`, `min_contact_frames`).
- **Graceful degradation:** every overlay layer is optional, so the demo still draws
  hold map + skeleton even when association or the reach line is missing.
