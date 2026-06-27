"""Static hold map — run the detector ONCE on the empty-wall frame (Coder 2).

The climber's body occludes holds when in use, so we detect on the empty wall once
and persist the map (boxes + confidence) as JSON. Every later step associates the
climber against this fixed map.

Run:  python pipeline/holds_map.py --empty demo/empty.jpg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))
from config import Paths, load_config, resolve_paths  # noqa: E402

logger = logging.getLogger(__name__)


def build_hold_map(empty_frame: Path, weights: Path, imgsz: int, conf: float = 0.25) -> list[dict]:
    """Detect holds on the empty-wall frame → ``[{"xyxy", "conf", "cls"}, ...]``.

    Runs the frozen detector ONCE at ``imgsz``. Boxes come back in the frame's own
    pixel space, which is the same space pose runs in (same fixed camera), so no
    rescaling is needed downstream. Holds are sorted top→bottom, left→right so the
    map indices are stable across runs.
    """
    from ultralytics import YOLO  # heavy; imported lazily so the module stays importable

    if not empty_frame.is_file():
        raise FileNotFoundError(f"Empty-wall frame not found: {empty_frame}")
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")

    result = YOLO(str(weights)).predict(str(empty_frame), imgsz=imgsz, conf=conf, verbose=False)[0]
    holds = [
        {
            "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            "conf": round(float(box.conf[0]), 4),
            "cls": int(box.cls[0]),
        }
        for box in result.boxes
    ]
    holds.sort(key=lambda h: (h["xyxy"][1], h["xyxy"][0]))  # stable top→bottom, left→right
    return holds


def run(cfg: dict, paths: Paths, empty_frame: Path, out: Path) -> Path:
    holds = build_hold_map(empty_frame, paths.weights, cfg["image_size"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(holds, indent=2), encoding="utf-8")
    logger.info("Wrote %d holds -> %s", len(holds), out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--empty", required=True, help="Path to the empty-wall reference frame.")
    ap.add_argument("--out", default="demo/hold_map.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run(cfg, paths, Path(args.empty), paths.root / args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
