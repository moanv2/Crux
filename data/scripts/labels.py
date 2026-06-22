"""Tiny helpers for reading YOLO label files (shared by stats.py and sanity_check.py).

A YOLO label file holds one box per line: ``class_id cx cy w h`` with all
coordinates normalized to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Box:
    """One YOLO box (normalized coordinates)."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


def iter_label_files(root: Path) -> Iterator[Path]:
    """Yield every YOLO ``.txt`` label file under any ``labels/`` dir below root.

    Falls back to all ``*.txt`` under root if there are no ``labels/`` dirs, so
    it works on both Roboflow exports (train/labels, ...) and flat layouts.
    """
    labelled = sorted(p for p in root.rglob("*.txt") if "labels" in p.parts)
    if labelled:
        yield from labelled
    else:
        yield from sorted(root.rglob("*.txt"))


def parse_boxes(path: Path) -> list[Box]:
    """Parse one label file into a list of :class:`Box` (skips blank lines)."""
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = parts[:5]
        boxes.append(Box(int(float(cls)), float(cx), float(cy), float(w), float(h)))
    return boxes
