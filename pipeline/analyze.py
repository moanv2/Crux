"""Climbing-coach analysis — the value layer (Coder 2).

All built from the static hold map + per-frame keypoints, in the shared 640 pixel
space. Thresholds come from config.yaml `pipeline:`. These three outputs are the
debrief: which holds were used, where the crux is, and feet cuts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Contact:
    """A hold in use by a limb."""
    hold_index: int
    limb: str          # "left_hand" | "right_hand" | "left_foot" | "right_foot"
    first_frame: int


def contacts(hold_map: list[dict], keypoints_per_frame: list, margin_px: int, min_frames: int) -> list[Contact]:
    """Which holds each hand/foot uses.

    A hold is "in use" by a limb when that limb's keypoint sits inside the hold box
    (expanded by ``margin_px``) for >= ``min_frames`` consecutive frames. Returns the
    contacts in the order they first occur (the executed beta).
    """
    # TODO: for each frame, test wrist(9,10)/ankle(15,16) keypoints against each
    #       (expanded) hold box; debounce with a per-(hold,limb) consecutive counter.
    raise NotImplementedError


def crux(keypoints_per_frame: list) -> dict:
    """Crux via motion. Returns {"time_per_move": [...], "crux_move": int}.

    Compute body-motion per frame (sum of keypoint displacement, or hip speed),
    segment into moves vs pauses, and flag the longest pause as the crux candidate.
    """
    # TODO: per-frame motion -> threshold into move/pause segments -> longest pause.
    raise NotImplementedError


def feet_cuts(hold_map: list[dict], keypoints_per_frame: list) -> dict:
    """Feet cuts. Returns {"count": int, "frames": [...]}.

    A foot cut = both ankles leave all holds (off-wall) for a sustained moment.
    """
    # TODO: per-frame "ankle on any hold?" booleans; count on->off (both) transitions.
    raise NotImplementedError
