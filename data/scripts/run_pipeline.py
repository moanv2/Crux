"""Run the whole data pipeline end to end with one command.

    python data/scripts/run_pipeline.py

Order: download -> clean -> ingest_custom -> stats -> split -> export -> sanity_check.
By default (non-strict) the run keeps going so the implemented tail can run once
inputs exist: stub steps raise NotImplementedError and are skipped, and `download`
is soft-skipped when it isn't configured yet (TODO Roboflow coords or no
ROBOFLOW_API_KEY) so you can run on data already in data/raw. Any *other* step that
errors is a hard stop. Pass --strict to stop at the first failing/stub step.
Use --only / --skip to run a subset.
"""
from __future__ import annotations

import argparse
import logging

from config import load_config, resolve_paths

import clean
import download
import export
import ingest_custom
import sanity_check
import split
import stats

logger = logging.getLogger(__name__)

# (name, module) in execution order.
STEPS = [
    ("download", download),
    ("clean", clean),
    ("ingest_custom", ingest_custom),
    ("stats", stats),
    ("split", split),
    ("export", export),
    ("sanity_check", sanity_check),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: repo root).")
    ap.add_argument("--strict", action="store_true", help="Stop at the first failing/stub step.")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="Run only these steps.")
    ap.add_argument("--skip", nargs="+", metavar="STEP", default=[], help="Skip these steps.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    steps = [(n, m) for n, m in STEPS if (not args.only or n in args.only) and n not in args.skip]
    failures = 0
    for name, module in steps:
        logger.info("=== %s ===", name)
        try:
            module.run(cfg, paths)
        except NotImplementedError as exc:
            failures += 1
            logger.warning("%s is not implemented yet: %s", name, exc)
            if args.strict:
                return 1
        except Exception as exc:  # noqa: BLE001
            # download depends on external setup (Roboflow coords + ROBOFLOW_API_KEY);
            # treat its "not ready" failure as a soft skip so the implemented tail still runs.
            if name == "download" and not args.strict:
                failures += 1
                logger.warning("download skipped (not configured/ready): %s", exc)
                continue
            logger.exception("%s failed: %s", name, exc)
            return 1
    logger.info("Pipeline finished (%d step(s) skipped as stubs).", failures)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
