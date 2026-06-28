"""End-to-end climbing demo (Coder 2).

Ties it all together: static hold map (once) + per-frame pose -> association
(holds in use) -> next-hold reach -> overlay -> rendered demo video. Always also
keep a recorded fallback so the live demo can't fail.

The clip's camera need not be perfectly static and need not contain an empty wall:
when ``--empty`` is omitted we synthesize the empty-wall reference from the clip
(stabilized temporal median) and render on a stabilized copy so the static hold map
stays aligned. Pass ``--empty`` to skip that and use a real empty-wall frame.

Run:  python pipeline/run.py --clip demo/climb.mp4 --out demo/out.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "scripts"))
from config import Paths, load_config, resolve_paths  # noqa: E402

import associate  # noqa: E402
import empty_frame  # noqa: E402
import holds_map  # noqa: E402
import overlay  # noqa: E402
import pose  # noqa: E402
import reach  # noqa: E402

logger = logging.getLogger(__name__)


def render(clip: Path, holds: list[dict], keypoints_per_frame: list,
           contacts_per_frame: list[dict], reaches: list, out: Path,
           fps: float | None = None) -> Path:
    """Draw the overlay onto each clip frame and write the demo video to ``out``.

    Reads the clip again for the raw frames and pairs them with the precomputed
    per-frame keypoints / contacts / reach (same order as pose streamed them).
    Stops at the shorter of (frames, analysis) so a length mismatch can't crash it.
    """
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open clip: {clip}")
    fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    i, n = 0, len(keypoints_per_frame)
    try:
        while i < n:
            ok, frame = cap.read()
            if not ok:
                break
            kp = keypoints_per_frame[i]
            contacts = contacts_per_frame[i] if i < len(contacts_per_frame) else None
            reach_i = reaches[i] if i < len(reaches) else None
            drawn = overlay.draw_frame(frame, holds, kp, contacts, reach_i)
            if writer is None:
                h, w = drawn.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(drawn)
            i += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    logger.info("Rendered %d frames -> %s", i, out)
    return out


def run(cfg: dict, paths: Paths, clip: Path, out: Path, empty: Path | None = None) -> Path:
    """Build the hold map, run pose per frame, associate + reach, overlay, render to ``out``.

    With ``empty=None`` the empty-wall reference is derived from ``clip`` and the
    render runs on a stabilized copy so the static hold map stays aligned.
    """
    p = cfg["pipeline"]
    if empty is None:
        empty, clip = empty_frame.prepare(
            clip, out.parent / "_work",
            n_samples=p.get("empty_frame_samples", 80),
            max_features=p.get("stabilize_max_features", 2000),
        )

    holds = holds_map.build_hold_map(empty, paths.weights, cfg["image_size"])
    keypoints = list(pose.pose_per_frame(clip, imgsz=cfg["image_size"]))
    contacts = associate.associate(holds, keypoints, p["association_margin_px"], p["min_contact_frames"])
    reaches = reach.reach_per_frame(holds, keypoints, contacts)
    detected = sum(1 for k in keypoints if k is not None)
    logger.info("holds=%d frames=%d (climber in %d)", len(holds), len(keypoints), detected)
    return render(clip, holds, keypoints, contacts, reaches, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--empty", default=None, help="Empty-wall frame; derived from the clip if omitted.")
    ap.add_argument("--out", default="demo/out.mp4")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run(cfg, paths, Path(args.clip), paths.root / args.out,
        empty=Path(args.empty) if args.empty else None)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
