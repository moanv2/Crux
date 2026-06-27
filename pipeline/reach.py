"""Next-hold reach — from the higher hand to the nearest unused hold above it (Coder 2).

Per frame: take the higher hand (the wrist with the smaller y, i.e. higher on the
wall), find the nearest hold whose center is above that hand and not currently in
use, and report the target + distance. The pixel distance is normalized by the
climber's shoulder→wrist length so it reads as a rough body-scale "how far to
reach". Geometric only — no config thresholds. Same 640 pixel space throughout.
"""
from __future__ import annotations

import math

# COCO-17 keypoint indices.
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
KPT_CONF_MIN = 0.30

# Distance normalization guards. A heavily foreshortened arm (e.g. on an overhang)
# gives a tiny shoulder->wrist length that blows the ratio up, so we only normalize
# when the arm is plausibly long and the result is sane; otherwise report None and
# let the overlay fall back to a raw-pixel label.
MIN_ARM_PX = 20.0
MAX_NORM = 4.0


def _point(keypoints, idx: int) -> tuple[float, float] | None:
    """(x, y) of a keypoint, or None when it's missing / low-confidence."""
    x, y, conf = keypoints[idx][0], keypoints[idx][1], keypoints[idx][2]
    return (x, y) if conf >= KPT_CONF_MIN else None


def _hold_centers(hold_map: list[dict]) -> list[tuple[float, float]]:
    return [((b["xyxy"][0] + b["xyxy"][2]) / 2, (b["xyxy"][1] + b["xyxy"][3]) / 2) for b in hold_map]


def reach_frame(hold_map: list[dict], keypoints, used_holds: set[int]) -> dict | None:
    """Reach from the higher hand to the nearest unused hold above it (one frame).

    Returns ``{"from_xy", "to_hold", "to_xy", "distance_px", "distance_norm"}`` or
    ``None`` when there's no detected hand or no candidate hold above it.
    """
    if keypoints is None or not hold_map:
        return None

    hands = []  # (wrist_xy, shoulder_idx)
    for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        p = _point(keypoints, wrist_idx)
        if p is not None:
            hands.append((p, shoulder_idx))
    if not hands:
        return None
    (hx, hy), shoulder_idx = min(hands, key=lambda t: t[0][1])  # smallest y == highest hand

    centers = _hold_centers(hold_map)
    best_i, best_d = None, None
    for i, (cx, cy) in enumerate(centers):
        if i in used_holds or cy >= hy:  # skip used holds and anything not above the hand
            continue
        d = math.hypot(cx - hx, cy - hy)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    if best_i is None:
        return None

    shoulder = _point(keypoints, shoulder_idx)
    arm = math.hypot(shoulder[0] - hx, shoulder[1] - hy) if shoulder is not None else None
    norm = best_d / arm if arm and arm >= MIN_ARM_PX else None
    if norm is not None and norm > MAX_NORM:
        norm = None  # implausible (foreshortened arm / pose jitter) — fall back to pixels

    cx, cy = centers[best_i]
    return {
        "from_xy": (round(hx, 1), round(hy, 1)),
        "to_hold": best_i,
        "to_xy": (round(cx, 1), round(cy, 1)),
        "distance_px": round(best_d, 1),
        "distance_norm": round(norm, 2) if norm is not None else None,
    }


def reach_per_frame(hold_map: list[dict], keypoints_per_frame: list, contacts_per_frame: list[dict]) -> list:
    """Run :func:`reach_frame` over the clip; "used" holds come from the contacts of each frame."""
    out = []
    for keypoints, contacts in zip(keypoints_per_frame, contacts_per_frame):
        used = {h for h in contacts.values() if h is not None}
        out.append(reach_frame(hold_map, keypoints, used))
    return out
