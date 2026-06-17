"""Step 1 — download the team's annotated Roboflow project via the SDK into ``data/raw/``.

This pulls a specific *version* of YOUR forked + re-typed project (boxes seeded
from GDSCMoonless, re-typed to our 6 classes, plus custom photos). Workspace /
project / version come from ``config.yaml``; the API key comes from the
``ROBOFLOW_API_KEY`` environment variable (never commit it).

Run:  python data/scripts/download.py
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from config import Paths, load_config, resolve_paths

logger = logging.getLogger(__name__)


def run(cfg: dict[str, Any], paths: Paths) -> Path:
    """Download the configured Roboflow version into ``paths.raw_dir``."""
    rf = cfg["roboflow"]
    if str(rf["workspace"]).startswith("TODO") or str(rf["project"]).startswith("TODO"):
        raise RuntimeError(
            "Set roboflow.workspace/project/version in config.yaml to your forked project "
            "(grab them from the dataset's 'Download Dataset → show download code')."
        )
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY in your environment (e.g. a .env file).")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("roboflow not installed — run: pip install -r requirements.txt") from exc

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    project = Roboflow(api_key=api_key).workspace(rf["workspace"]).project(rf["project"])
    version = project.version(int(rf["version"]))
    dataset = version.download(rf["format"], location=str(paths.raw_dir), overwrite=True)

    location = Path(dataset.location)
    logger.info("Downloaded %s/%s v%s (%s) -> %s",
                rf["workspace"], rf["project"], rf["version"], rf["format"], location)
    logger.info("Record this version id (%s) in %s.", rf["version"], paths.data_card.name)
    return location


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run(cfg, paths)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
