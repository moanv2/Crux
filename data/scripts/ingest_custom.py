"""Step 3 — ingest the team's phone photos and merge them into ``data/interim/``.

Teammates annotate their wall/hold photos in Roboflow and export YOLO; drop the
export under ``ingest.incoming_dir``. This merges them into the cleaned set so
training data matches the demo footage. Class names must match the locked list.

Run:  python data/scripts/ingest_custom.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from config import Paths, load_config, resolve_paths

logger = logging.getLogger(__name__)


def validate_classes(label_dir: Path, classes: list[str]) -> None:
    """Fail loudly if the custom export uses classes outside the locked list."""
    # TODO: parse each .txt, check every class index is < len(classes).
    raise NotImplementedError


def run(cfg: dict[str, Any], paths: Paths) -> Path:
    """Merge custom photos from ``ingest.incoming_dir`` into ``paths.interim_dir``."""
    incoming = (paths.root / cfg["ingest"]["incoming_dir"]).resolve()
    classes = cfg["classes"]
    if not incoming.exists():
        logger.warning("No custom photos at %s — skipping ingest.", incoming)
        return paths.interim_dir
    logger.info("Ingesting custom photos from %s (classes=%s)", incoming, classes)
    # TODO: validate_classes -> copy/resize images + labels into interim_dir,
    #       prefixing filenames (e.g. "custom_") to avoid collisions.
    raise NotImplementedError("TODO: implement custom photo merge.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    out = run(cfg, paths)
    logger.info("Custom photos merged into %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
