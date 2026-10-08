#!/usr/bin/env python3
"""Generate synthetic fallback data for one or more dataset keys.

Last resort of the acquisition ladder (Section 5.6 step 4). Every generated
table is tagged ``data_source="synthetic"`` downstream.

Usage::

    python scripts/make_synthetic_data.py sentiment_mh osmi_2014
    python scripts/make_synthetic_data.py --all-missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindsense.data.download import FETCHERS, DatasetResult  # noqa: E402
from mindsense.data.synthetic import make_synthetic  # noqa: E402
from mindsense.utils.logging import get_logger  # noqa: E402

log = get_logger("mindsense.synthetic_script")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keys", nargs="*", help="dataset keys to synthesise")
    parser.add_argument("--all-missing", action="store_true",
                        help="synthesise every configured dataset with no raw files")
    args = parser.parse_args(argv)

    keys = list(args.keys)
    if args.all_missing:
        from mindsense.utils.io import load_config, repo_path

        for key in load_config()["datasets"]:
            raw_dir = repo_path("data", "raw", key)
            if not raw_dir.exists() or not any(raw_dir.rglob("*")):
                keys.append(key)
    if not keys:
        if args.all_missing:
            log.info("all datasets already have raw files; nothing to synthesise")
            return 0
        log.error("nothing to do: pass keys or --all-missing")
        return 2

    failures = 0
    for key in keys:
        if key not in FETCHERS:
            log.error("unknown key %r", key)
            failures += 1
            continue
        res = make_synthetic(key, prior=DatasetResult(key, "failed", notes="forced"))
        log.info("%s -> %s (%s)", key, res.status, res.notes)
        failures += int(res.status != "synthetic")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
