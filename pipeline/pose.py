"""Per-frame body pose — pretrained, no training (Coder 2).

Runs a pretrained pose model (yolo11n-pose / yolo8n-pose) on each frame of the
climbing clip → 17 COCO keypoints per frame, in the same 640 pixel space as the
hold map. Keypoint indices we care about: 9/10 wrists, 15/16 ankles, 5/6 shoulders, 11/12 hips.

Run:  python pipeline/pose.py --clip demo/climb.mp4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

POSE_MODEL = "yolo11n-pose.pt"   # pretrained COCO 17-keypoint


def pose_per_frame(clip: Path, imgsz: int = 640, model: str = POSE_MODEL) -> list:
    """Return a list (one per frame) of 17 (x, y, conf) keypoints."""
    # TODO: from ultralytics import YOLO; for r in YOLO(model)(str(clip), stream=True, imgsz=imgsz):
    #       yield r.keypoints.data[0].tolist() if r.keypoints is not None else None
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="Path to the climbing clip.")
    args = ap.parse_args(argv)
    frames = pose_per_frame(Path(args.clip))
    logger.info("Pose keypoints for %d frames", len(frames))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
