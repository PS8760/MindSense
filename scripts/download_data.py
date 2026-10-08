#!/usr/bin/env python
"""Acquire every configured dataset (Section 5.6 fallback ladder) + inventory.

Usage
-----
    python scripts/download_data.py                 # all datasets
    python scripts/download_data.py --only dass42   # one dataset
    python scripts/download_data.py --no-synthetic  # never fabricate data

Writes ``reports/tables/data_inventory.csv``: key, rank, tier, source URL,
licence, population, experiments, access status, ``real|synthetic``, primary
file, SHA-256, rows/cols, access date (prompt.md Rule 8 / Section 5.6 step 6).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mindsense.data.download import FETCHERS, DatasetResult, fetch_all, relabel_if_synthetic
from mindsense.utils.io import ensure_dir, file_sha256, load_config, repo_path
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.download_data")

INVENTORY_PATH = repo_path("reports", "tables", "data_inventory.csv")

#: Primary file (relative to data/raw/<key>/) used for hashing + shape.
_PRIMARY = {
    "student_depression": "Student Depression Dataset.csv",
    "student_depression_small": "Depression Student Dataset.csv",
    "sentiment_mh": "Combined Data.csv",
    "dass42": "DASS_data_21.02.19/data.csv",
    "osmi_2014": "osmi_2014.csv",
    "osmi_2017_21": "OSMI Mental Health in Tech Survey 2017.csv",
    "drug_reviews": "drugsComTrain_raw.csv",
    "dreaddit": "train-00000-of-00001.parquet",
    "fer2013": None,  # directory of images — hashed via manifest of paths
    "druglib": None,
}


def _primary_path(key: str) -> Path | None:
    name = _PRIMARY.get(key)
    if name:
        path = repo_path("data", "raw", key, name)
        return path if path.exists() else None
    raw = repo_path("data", "raw", key)
    files = (
        sorted(p for p in raw.rglob("*") if p.is_file() and p.name != ".mindsense_synthetic")
        if raw.exists()
        else []
    )
    return files[0] if files else None


def _shape(path: Path | None, key: str) -> tuple[object, object]:
    if path is None:
        return "", ""
    try:
        if path.suffix == ".jpg" or key == "fer2013":
            n = len(list(repo_path("data", "raw", key).rglob("*.jpg")))
            return n, 1
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep="\t" if path.name == "data.csv" else ",")
        return len(df), len(df.columns)
    except Exception as exc:  # noqa: BLE001 - inventory must not fail the run
        log.warning("could not read %s: %s", path, exc)
        return "", ""


def _digest(path: Path | None, key: str) -> str:
    if path is None:
        return ""
    if key == "fer2013" or path.suffix == ".jpg":
        names = "\n".join(
            str(p.relative_to(repo_path()))
            for p in sorted(repo_path("data", "raw", key).rglob("*.jpg"))
        )
        import hashlib

        return hashlib.sha256(names.encode()).hexdigest()
    return file_sha256(path)


def build_inventory(results: list[DatasetResult]) -> pd.DataFrame:
    cfg = load_config()
    today = date.today().isoformat()
    rows = []
    for res in results:
        meta = cfg["datasets"][res.key]
        primary = _primary_path(res.key)
        n_rows, n_cols = _shape(primary, res.key)
        synthetic = res.status == "synthetic"
        rows.append(
            {
                "key": res.key,
                "rank": meta.get("rank"),
                "tier": meta.get("tier"),
                "source_url": meta.get("source_url", ""),
                "licence": meta.get("licence", ""),
                "population": meta.get("population", ""),
                "experiments": " ".join(map(str, meta.get("experiments", []))),
                "access_status": res.status,
                "data_source": "synthetic" if synthetic else "real",
                "primary_file": primary.relative_to(repo_path()) if primary else "",
                "sha256": _digest(primary, res.key),
                "rows": n_rows,
                "cols": n_cols,
                "accessed_on": today,
                "notes": res.notes or res.source,
            }
        )
    # Upsert: ``--only`` runs must not wipe rows for other datasets.
    inv = pd.DataFrame(rows)
    if INVENTORY_PATH.exists():
        old = pd.read_csv(INVENTORY_PATH)
        old = old[~old["key"].isin(inv["key"])]
        inv = pd.concat([old, inv], ignore_index=True)
    inv = inv.sort_values("rank").reset_index(drop=True)
    ensure_dir(INVENTORY_PATH.parent)
    inv.to_csv(INVENTORY_PATH, index=False)
    log.info("data inventory -> %s (%d datasets)", INVENTORY_PATH, len(inv))
    return inv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY",
        help="restrict to one dataset key (repeatable)",
    )
    parser.add_argument(
        "--no-synthetic", action="store_true", help="never fall back to labelled synthetic data"
    )
    args = parser.parse_args(argv)

    if args.only:
        unknown = [k for k in args.only if k not in FETCHERS]
        if unknown:
            parser.error(f"unknown dataset key(s): {unknown}; valid: {sorted(FETCHERS)}")
        from mindsense.data.synthetic import make_synthetic

        results = []
        for key in args.only:
            res = relabel_if_synthetic(FETCHERS[key]())
            if not res.ok and not args.no_synthetic:
                res = make_synthetic(key, prior=res)
            results.append(res)
    else:
        results = fetch_all(allow_synthetic=not args.no_synthetic)

    build_inventory(results)
    failed = [r for r in results if not r.ok]
    for r in results:
        log.info("  %-24s %s", r.key, r.status)
    if failed:
        log.warning("not OK: %s", ", ".join(f"{r.key} ({r.notes[:60]})" for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
