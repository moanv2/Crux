"""Draw the climb overlay (Coder 2).

Renders, per frame: the static hold map (boxes), the pose skeleton, contacts lit
up (hands green / feet blue), and the next-hold reach line + normalized distance.
Graceful degradation: each layer is optional, so a frame still draws holds +
skeleton even when contacts or the reach line are missing. All inputs are in the
shared pixel space (see holds_map / pose).
"""
from __future__ import annotations

import cv2

# BGR colors.
HOLD_COLOR = (0, 165, 255)     # orange outline for every hold
HAND_COLOR = (0, 200, 0)       # green — hold held by a hand
FOOT_COLOR = (255, 150, 0)     # blue — hold under a foot
SKELETON_COLOR = (240, 240, 240)
REACH_COLOR = (0, 255, 255)    # yellow reach line
HAND_LIMBS = {"left_hand", "right_hand"}

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]  # COCO 17-kpt edges
KPT_CONF_MIN = 0.30


def _draw_holds(img, hold_map, contacts) -> None:
    used: dict[int, set[str]] = {}
    if contacts:
        for limb, h in contacts.items():
            if h is not None:
                used.setdefault(h, set()).add(limb)
    for i, hold in enumerate(hold_map):
        x1, y1, x2, y2 = (int(v) for v in hold["xyxy"])
        limbs = used.get(i)
        if limbs:
            color = HAND_COLOR if any(l in HAND_LIMBS for l in limbs) else FOOT_COLOR
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)  # translucent fill
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), HOLD_COLOR, 1)


def _draw_skeleton(img, keypoints) -> None:
    if keypoints is None:
        return
    for a, b in SKELETON:
        if keypoints[a][2] >= KPT_CONF_MIN and keypoints[b][2] >= KPT_CONF_MIN:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(img, pa, pb, SKELETON_COLOR, 2)
    for x, y, conf in keypoints:
        if conf >= KPT_CONF_MIN:
            cv2.circle(img, (int(x), int(y)), 3, SKELETON_COLOR, -1)


def _draw_reach(img, reach) -> None:
    if not reach:
        return
    fx, fy = (int(v) for v in reach["from_xy"])
    tx, ty = (int(v) for v in reach["to_xy"])
    cv2.arrowedLine(img, (fx, fy), (tx, ty), REACH_COLOR, 2, tipLength=0.15)
    norm = reach.get("distance_norm")
    label = f"reach {norm:.2f} arm" if norm is not None else f"reach {reach['distance_px']:.0f}px"
    cv2.putText(img, label, ((fx + tx) // 2, (fy + ty) // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, REACH_COLOR, 2, cv2.LINE_AA)


def draw_frame(frame, hold_map, keypoints, contacts=None, reach=None):
    """Return ``frame`` with holds + skeleton (+ contacts + reach line when available).

    Draws on a copy so the caller's frame is left untouched. Every layer is guarded,
    so missing keypoints / contacts / reach never break the basic visual.
    """
    img = frame.copy()
    _draw_holds(img, hold_map, contacts)
    _draw_skeleton(img, keypoints)
    _draw_reach(img, reach)
    return img
