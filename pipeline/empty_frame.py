"""Derive the empty-wall reference (and a stabilized clip) from the climbing clip (Coder 2).

Our footage never includes a clean empty wall, so we synthesize one: sample frames
across the clip, register them to a reference (``Stabilizer``), and take the
per-pixel median — the moving climber is outvoted and cancels out, leaving the bare
wall for the detector to map ONCE. We also emit a stabilized copy of the clip so the
single static hold map stays aligned for every frame of the render.

Run:  python pipeline/empty_frame.py --clip demo/climb.mp4 --out-dir demo/_work
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from stabilize import Stabilizer

logger = logging.getLogger(__name__)


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    return frame if ok else None


def reference_frame(clip: Path) -> np.ndarray:
    """The clip's middle frame — used as the fixed coordinate space everything maps into."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open clip: {clip}")
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = _read_frame(cap, nf // 2)
    cap.release()
    if frame is None:
        raise FileNotFoundError(f"Could not read a frame from: {clip}")
    return frame


def build_empty_frame(clip: Path, stab: Stabilizer, n_samples: int = 80) -> np.ndarray:
    """Stabilized temporal-median empty wall: sample, warp to reference, median over time."""
    cap = cv2.VideoCapture(str(clip))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, nf - 1, min(n_samples, nf)).astype(int)
    warped = [stab.warp(f) for f in (_read_frame(cap, i) for i in idxs) if f is not None]
    cap.release()
    if not warped:
        raise ValueError(f"No frames sampled from: {clip}")
    return np.median(np.stack(warped), axis=0).astype(np.uint8)


def write_stabilized_clip(clip: Path, stab: Stabilizer, out: Path) -> Path:
    """Warp every frame into reference space and write the stabilized clip to ``out``."""
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, stab.size)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(stab.warp(frame))
        n += 1
    cap.release()
    writer.release()
    logger.info("Stabilized %d frames -> %s", n, out)
    return out


def prepare(clip: Path, work_dir: Path, n_samples: int = 80, max_features: int = 2000) -> tuple[Path, Path]:
    """Build (empty_frame.jpg, stabilized_clip.mp4) from a raw clip; returns their paths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stab = Stabilizer(reference_frame(clip), max_features=max_features)
    empty = build_empty_frame(clip, stab, n_samples)
    empty_path = work_dir / f"{clip.stem}_empty.jpg"
    cv2.imwrite(str(empty_path), empty)
    logger.info("Empty wall -> %s", empty_path)
    stab_clip = write_stabilized_clip(clip, stab, work_dir / f"{clip.stem}_stabilized.mp4")
    return empty_path, stab_clip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out-dir", default="demo/_work")
    ap.add_argument("--samples", type=int, default=80)
    args = ap.parse_args(argv)
    empty_path, stab_clip = prepare(Path(args.clip), Path(args.out_dir), args.samples)
    logger.info("Done: %s | %s", empty_path, stab_clip)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
