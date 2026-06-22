"""Step 2 — clean & normalize the raw YOLO dataset into data/interim/ (flat layout).

For the single-class 'hold' detector we:
  - collapse every box to class 0 ("hold") — idempotent; polygon/segmentation labels
    are converted to their bounding box rather than silently mangled;
  - drop corrupt / unreadable images and (optionally) empty-label images;
  - optionally resize to 640 (a stretch-resize leaves normalized YOLO coords unchanged);
  - optionally drop perceptual-hash **near**-duplicates (Hamming distance), before the
    split, so duplicates can't leak across train/val/test.

Reads from cfg.paths.raw_dir (any train/valid/test or flat YOLO layout) and writes a
flat data/interim/images + data/interim/labels with globally-unique filenames.
Optional deps (Pillow, imagehash) degrade gracefully: if missing, resize / corrupt-check
/ dedupe are skipped with a warning so the pipeline still runs.

Run:  python data/scripts/clean.py
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from config import Paths, load_config, resolve_paths

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Optional deps — degrade gracefully if absent.
try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False
try:
    import imagehash
    _HAS_IMAGEHASH = True
except Exception:  # pragma: no cover
    _HAS_IMAGEHASH = False


def find_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs from a YOLO export (train/valid/test or flat)."""
    pairs: list[tuple[Path, Path]] = []
    for label in sorted(raw_dir.rglob("*.txt")):
        if label.name.lower() in {"readme.txt", "requirements.txt"} or label.name.startswith("_"):
            continue
        img_dir = label.parent.parent / "images" if label.parent.name == "labels" else label.parent
        image = next((img_dir / f"{label.stem}{e}" for e in IMAGE_EXTS
                      if (img_dir / f"{label.stem}{e}").exists()), None)
        if image is None:  # fallback: image alongside the label
            image = next((label.parent / f"{label.stem}{e}" for e in IMAGE_EXTS
                          if (label.parent / f"{label.stem}{e}").exists()), None)
        if image is not None:
            pairs.append((image, label))
    return pairs


def _coords_to_bbox(coords: list[str]) -> list[str] | None:
    """Convert a YOLO line's coords (after the class id) to 'cx cy w h'.

    Accepts a 4-value bbox (passthrough) or a polygon (>=6 even values) which is
    reduced to its axis-aligned bounding box. Returns None for malformed input.
    """
    try:
        if len(coords) == 4:
            return [f"{float(v):.6f}" for v in coords]
        if len(coords) >= 6 and len(coords) % 2 == 0:
            xs = [float(coords[i]) for i in range(0, len(coords), 2)]
            ys = [float(coords[i]) for i in range(1, len(coords), 2)]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            return [f"{(xmin + xmax) / 2:.6f}", f"{(ymin + ymax) / 2:.6f}",
                    f"{xmax - xmin:.6f}", f"{ymax - ymin:.6f}"]
    except ValueError:
        return None
    return None


def remap_to_single_class(label_text: str) -> str:
    """Set every box's class id to 0; convert polygons to bboxes; drop malformed. Idempotent."""
    out: list[str] = []
    for line in label_text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        box = _coords_to_bbox(p[1:])
        if box is not None:
            out.append("0 " + " ".join(box))
    return ("\n".join(out) + "\n") if out else ""


def is_corrupt(image: Path) -> bool:
    """True if the image header can't be parsed (full-decode failures are caught at save)."""
    if _HAS_PIL:
        try:
            with Image.open(image) as im:
                im.verify()
            return False
        except Exception:
            return True
    return image.stat().st_size == 0


def save_image(src: Path, dst: Path, resize_to: int | None, fix_exif: bool) -> None:
    """Copy (and optionally EXIF-fix + stretch-resize) an image to dst. May raise on bad pixels."""
    if _HAS_PIL and (resize_to or fix_exif):
        with Image.open(src) as im:
            if fix_exif:
                im = ImageOps.exif_transpose(im)
            if resize_to:
                im = im.convert("RGB").resize((resize_to, resize_to))
            im.save(dst)
    else:
        shutil.copy2(src, dst)


def _unique_stem(label: Path, raw_dir: Path, used: set[str]) -> str:
    """Build a globally-unique interim stem from the label's path under raw_dir."""
    try:
        parts = [p for p in label.relative_to(raw_dir).parent.parts
                 if p not in ("images", "labels", ".", "")]
    except ValueError:
        parts = []
    base = ("_".join(parts) + "_" + label.stem) if parts else label.stem
    stem, n = base, 2
    while stem in used:
        stem, n = f"{base}_{n}", n + 1
    used.add(stem)
    return stem


def run(cfg: dict[str, Any], paths: Paths) -> Path:
    """Clean cfg.paths.raw_dir into cfg.paths.interim_dir; return interim dir."""
    c = cfg["clean"]
    if not paths.raw_dir.exists():
        raise RuntimeError(f"No raw data at {paths.raw_dir}. Run download first.")
    if not _HAS_PIL:
        logger.warning("Pillow not installed — resize/EXIF/corrupt-check are no-ops (pip install pillow).")
    do_dedupe = bool(c.get("dedupe")) and _HAS_IMAGEHASH and _HAS_PIL
    if c.get("dedupe") and not do_dedupe:
        logger.warning("dedupe needs Pillow + imagehash — skipping (pip install pillow imagehash).")

    img_out, lbl_out = paths.interim_dir / "images", paths.interim_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(paths.raw_dir)
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {paths.raw_dir}.")

    kept = dropped_corrupt = dropped_empty = dropped_dupe = 0
    kept_hashes: list[Any] = []
    used_stems: set[str] = set()
    resize_to = c.get("resize_to") if _HAS_PIL else None
    max_dist = c.get("dedupe_max_distance", 5)

    for image, label in pairs:
        if c.get("drop_corrupt") and is_corrupt(image):
            dropped_corrupt += 1
            continue
        text = (remap_to_single_class(label.read_text(encoding="utf-8"))
                if c.get("remap_to_single_class", True) else label.read_text(encoding="utf-8"))
        if not text.strip() and c.get("drop_empty_labels"):
            dropped_empty += 1
            continue

        if do_dedupe:
            try:
                with Image.open(image) as im:
                    h = imagehash.phash(im, hash_size=c.get("dedupe_hash_size", 16))
                if any((h - k) <= max_dist for k in kept_hashes):
                    dropped_dupe += 1
                    continue
                kept_hashes.append(h)
            except Exception:
                pass  # hashing failed → keep the image; a truly broken file is caught at save

        stem = _unique_stem(label, paths.raw_dir, used_stems)
        try:
            save_image(image, img_out / f"{stem}{image.suffix.lower()}", resize_to, bool(c.get("fix_exif")))
        except Exception:
            dropped_corrupt += 1
            used_stems.discard(stem)
            logger.warning("dropping unreadable image %s", image)
            continue
        (lbl_out / f"{stem}.txt").write_text(text, encoding="utf-8")
        kept += 1

    logger.info("clean: kept %d | dropped %d corrupt, %d empty, %d dupes -> %s",
                kept, dropped_corrupt, dropped_empty, dropped_dupe, paths.interim_dir)
    return paths.interim_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    out = run(cfg, paths)
    logger.info("Cleaned dataset written to %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
