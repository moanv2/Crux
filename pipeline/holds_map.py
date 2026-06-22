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


def build_hold_map(empty_frame: Path, weights: Path, imgsz: int) -> list[dict]:
    """Detect holds on the empty-wall frame → [{"xyxy": [...], "conf": float}, ...]."""
    # TODO: from ultralytics import YOLO; r = YOLO(str(weights))(str(empty_frame), imgsz=imgsz)[0]
    #       return [{"xyxy": b.xyxy[0].tolist(), "conf": float(b.conf)} for b in r.boxes]
    raise NotImplementedError


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
