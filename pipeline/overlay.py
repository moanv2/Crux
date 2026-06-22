"""Draw the debrief overlay (Coder 2).

Renders, per frame: the static hold map (boxes), the pose skeleton, contacts lit up
(hands green / feet blue), and a debrief card (time-per-move bar, feet-cut count,
crux marker). Graceful degradation: if analysis is missing, still draw holds + skeleton.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]  # COCO 17-kpt edges


def draw_frame(frame, hold_map, keypoints, contacts=None, debrief=None):
    """Return ``frame`` with holds + skeleton (+ contacts + debrief card if available)."""
    # TODO (OpenCV): cv2.rectangle for each hold; draw SKELETON edges; tint contact
    #       holds; render the debrief card. Keep it robust to missing keypoints/analysis.
    raise NotImplementedError
