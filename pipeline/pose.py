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


def pose_per_frame(clip: Path, imgsz: int = 640, model: str = POSE_MODEL, conf: float = 0.25):
    """Yield, per frame, the climber's 17 ``[x, y, conf]`` keypoints (or ``None``).

    Streams the clip so memory stays flat on long videos. When several people are
    detected we keep the highest-confidence one (the climber); when none are, we
    yield ``None`` so downstream steps can degrade gracefully on that frame.
    Keypoints are in the frame's own pixel space — the same space as the hold map.
    """
    from ultralytics import YOLO  # heavy; imported lazily so the module stays importable

    if not clip.is_file():
        raise FileNotFoundError(f"Clip not found: {clip}")

    for result in YOLO(model).predict(str(clip), stream=True, imgsz=imgsz, conf=conf, verbose=False):
        kpts = result.keypoints
        if kpts is None or kpts.data is None or len(kpts.data) == 0:
            yield None
            continue
        # pick the most confident person if the detector found more than one
        if result.boxes is not None and len(result.boxes) > 1:
            best = int(result.boxes.conf.argmax())
        else:
            best = 0
        yield kpts.data[best].tolist()  # 17 x [x, y, conf]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="Path to the climbing clip.")
    args = ap.parse_args(argv)
    frames = list(pose_per_frame(Path(args.clip)))
    detected = sum(1 for f in frames if f is not None)
    logger.info("Pose: %d/%d frames with a climber detected", detected, len(frames))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
