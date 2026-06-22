"""End-to-end climbing-coach demo (Coder 2).

Ties it all together: static hold map (once) + per-frame pose -> analysis
(contacts / crux / feet cuts) -> overlay -> rendered demo video. Always also keep a
recorded fallback so the live demo can't fail.

Run:  python pipeline/run.py --empty demo/empty.jpg --clip demo/climb.mp4 --out demo/out.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))
from config import Paths, load_config, resolve_paths  # noqa: E402

import analyze  # noqa: E402
import holds_map  # noqa: E402
import pose  # noqa: E402

logger = logging.getLogger(__name__)


def run(cfg: dict, paths: Paths, empty: Path, clip: Path, out: Path) -> Path:
    """Build the hold map, run pose per frame, analyze, overlay, and render to `out`."""
    p = cfg["pipeline"]
    holds = holds_map.build_hold_map(empty, paths.weights, cfg["image_size"])
    keypoints = list(pose.pose_per_frame(clip, imgsz=cfg["image_size"]))
    used = analyze.contacts(holds, keypoints, p["association_margin_px"], p["min_contact_frames"])
    crux = analyze.crux(keypoints)
    cuts = analyze.feet_cuts(holds, keypoints)
    logger.info("contacts=%s crux=%s feet_cuts=%s", used, crux, cuts)
    # TODO: per-frame overlay.draw_frame(...) + write the video (OpenCV VideoWriter) to `out`.
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--empty", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="demo/out.mp4")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run(cfg, paths, Path(args.empty), Path(args.clip), paths.root / args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
