"""Camera stabilization — register frames to a reference to cancel handheld drift (Coder 2).

Phone clips drift a few pixels even when "static", which smears a temporal-median
empty wall and slowly misaligns the static hold map over a clip. We estimate a
per-frame similarity transform (ORB features + RANSAC) onto one reference frame and
warp each frame into that fixed reference space, so the hold map and the pose
keypoints share one coordinate system for the whole clip.
"""
from __future__ import annotations

import cv2
import numpy as np

MATCHES_KEPT = 200  # best matches fed to the RANSAC transform estimate


class Stabilizer:
    """Aligns frames to a fixed reference frame via an ORB-based similarity transform."""

    def __init__(self, reference_bgr: np.ndarray, max_features: int = 2000, min_matches: int = 12):
        self.size = (reference_bgr.shape[1], reference_bgr.shape[0])  # (w, h) for warpAffine
        self.min_matches = min_matches
        self._orb = cv2.ORB_create(max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self._kref, self._dref = self._orb.detectAndCompute(gray, None)
        if self._dref is None:
            raise ValueError("Reference frame has no detectable features to stabilize against.")

    def affine(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """2x3 similarity transform mapping ``frame_bgr`` -> reference, or None if unreliable.

        RANSAC rejects the climber's motion as outliers, so the fit locks onto the
        static wall even though the reference frame contains the climber.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < self.min_matches:
            return None
        matches = self._bf.match(desc, self._dref)
        if len(matches) < self.min_matches:
            return None
        matches = sorted(matches, key=lambda m: m.distance)[:MATCHES_KEPT]
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self._kref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        return M

    def warp(self, frame_bgr: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        """Warp a frame into reference space. Estimates ``M`` if not given; on failure
        returns the frame untouched (graceful — one unalignable frame won't crash a run)."""
        if M is None:
            M = self.affine(frame_bgr)
        if M is None:
            return frame_bgr
        return cv2.warpAffine(frame_bgr, M, self.size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
