"""Hold-in-use association — which holds each hand/foot is on (Coder 2).

Built from the static hold map + per-frame keypoints, in the shared 640 pixel
space. A hold is "in use" by a limb when that limb's keypoint sits inside the
hold box (expanded by ``margin_px``) for >= ``min_frames`` consecutive frames;
the N-frame debounce suppresses single-frame keypoint jitter. Thresholds come
from ``config.yaml`` ``pipeline:`` (``association_margin_px``, ``min_contact_frames``).
"""
from __future__ import annotations

# COCO-17 keypoint indices for the limbs we track.
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16
LIMB_KPT: dict[str, int] = {
    "left_hand": L_WRIST,
    "right_hand": R_WRIST,
    "left_foot": L_ANKLE,
    "right_foot": R_ANKLE,
}
KPT_CONF_MIN = 0.30  # ignore keypoints the pose model is unsure about


def _point_in_box(x: float, y: float, box: list[float], margin: int) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _hold_for_point(x: float, y: float, hold_map: list[dict], margin: int) -> int | None:
    """Index of the hold whose (expanded) box contains the point; nearest center wins ties."""
    best, best_d = None, None
    for i, hold in enumerate(hold_map):
        box = hold["xyxy"]
        if _point_in_box(x, y, box, margin):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            d = (x - cx) ** 2 + (y - cy) ** 2
            if best_d is None or d < best_d:
                best, best_d = i, d
    return best


def associate_frame(hold_map: list[dict], keypoints, margin_px: int) -> dict[str, int | None]:
    """Raw (un-debounced) ``limb -> hold index`` for ONE frame.

    Returns every limb mapped to ``None`` when the frame has no detected climber or
    the keypoint is low-confidence / on no hold.
    """
    out: dict[str, int | None] = {limb: None for limb in LIMB_KPT}
    if keypoints is None:
        return out
    for limb, idx in LIMB_KPT.items():
        x, y, conf = keypoints[idx][0], keypoints[idx][1], keypoints[idx][2]
        if conf < KPT_CONF_MIN:
            continue
        out[limb] = _hold_for_point(x, y, hold_map, margin_px)
    return out


def associate(hold_map: list[dict], keypoints_per_frame: list, margin_px: int, min_frames: int) -> list[dict]:
    """Per-frame CONFIRMED contacts with an N-frame debounce.

    Returns a list (one entry per frame) of ``{limb: hold_index | None}``. A limb's
    hold is only reported once the limb has sat in that same hold for >=
    ``min_frames`` consecutive frames; it drops back to ``None`` the moment the limb
    leaves all holds. While a limb is moving to a new hold but not yet confirmed, it
    reports ``None`` (no phantom contact on the hold being left).
    """
    confirmed: list[dict] = []
    streak_hold: dict[str, int | None] = {limb: None for limb in LIMB_KPT}
    streak_len: dict[str, int] = {limb: 0 for limb in LIMB_KPT}
    active: dict[str, int | None] = {limb: None for limb in LIMB_KPT}

    for keypoints in keypoints_per_frame:
        raw = associate_frame(hold_map, keypoints, margin_px)
        for limb in LIMB_KPT:
            h = raw[limb]
            if h is None:
                streak_hold[limb], streak_len[limb], active[limb] = None, 0, None
            elif h == streak_hold[limb]:
                streak_len[limb] += 1
                if streak_len[limb] >= min_frames:
                    active[limb] = h
            else:  # moved to a different hold — restart the streak
                streak_hold[limb], streak_len[limb] = h, 1
                active[limb] = h if min_frames <= 1 else None
        confirmed.append(dict(active))
    return confirmed
