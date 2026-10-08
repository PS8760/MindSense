"""Path handling and artifact IO (JSON/CSV/parquet) with repo-root discovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(*parts: str) -> Path:
    """Return an absolute path inside the repository."""
    return REPO_ROOT.joinpath(*parts)


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (parents included) and return it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def load_config() -> dict[str, Any]:
    """Load ``config/config.yaml``."""
    with open(repo_path("config", "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_helplines() -> dict[str, Any]:
    """Load ``config/helplines.yaml``."""
    with open(repo_path("config", "helplines.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_json(obj: Any, path: Path | str, *, indent: int = 2) -> Path:
    """Serialise ``obj`` to JSON (path created on demand)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str)
    return path


def load_json(path: Path | str) -> Any:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_table(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    """Read csv/tsv/parquet based on file suffix."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if path.suffix in {".gz"} and path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to parquet, creating parent dirs."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def file_sha256(path: Path | str, *, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 of a file (used for the data inventory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
