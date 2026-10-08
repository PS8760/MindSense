"""Dataset acquisition with the Section-5.6 fallback ladder.

For every dataset we try, in order:

1. **Automated download** (Kaggle anonymous/API, OpenML, direct URL, HF parquet).
2. **Manual placement** – print exact instructions and accept pre-placed files.
3. **Alternate source** – the mirror listed in the ranked inventory.
4. **Synthetic fallback** (last resort, ``scripts/make_synthetic_data.py``) –
   clearly tagged ``data_source="synthetic"`` everywhere.

Nothing here writes to the repository history: raw files live in
``data/raw/<key>/`` which is git-ignored (Section 5.7).
"""

from __future__ import annotations

import io
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from mindsense.utils.io import ensure_dir, load_config, repo_path
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.download")
UA = {"User-Agent": "MindSense-academic/1.0 (research; contact via repo README)"}
TIMEOUT = 120


@dataclass
class DatasetResult:
    """Outcome of one dataset acquisition attempt."""

    key: str
    status: str  # downloaded | present | manual | synthetic | failed
    source: str = ""
    files: list[Path] = field(default_factory=list)
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"downloaded", "present", "synthetic"}


def _raw_dir(key: str) -> Path:
    return ensure_dir(repo_path("data", "raw", key))


def _existing_files(key: str, suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(
        p for p in _raw_dir(key).rglob("*") if p.is_file() and p.suffix in suffixes
    )


def _save_bytes(key: str, filename: str, content: bytes) -> Path:
    path = _raw_dir(key) / filename
    path.write_bytes(content)
    return path


def _unzip(key: str, blob: bytes) -> list[Path]:
    dest = _raw_dir(key)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest)
    return [dest / n for n in zf.namelist() if not n.endswith("/")]


def kaggle_download(ref: str, key: str) -> list[Path]:
    """Download a public Kaggle dataset.

    Tries the official ``kaggle`` CLI first when ``~/.kaggle/kaggle.json``
    exists, then falls back to the anonymous public API (works for datasets
    with an open licence).
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            import subprocess

            ensure_dir(_raw_dir(key))
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", ref, "-p", str(_raw_dir(key)), "--unzip"],
                check=True,
                capture_output=True,
                timeout=600,
            )
            return _existing_files(key, (".csv", ".tsv", ".txt", ".zip"))
        except Exception as exc:  # noqa: BLE001 - ladder continues to anonymous API
            log.warning("kaggle CLI failed for %s (%s); trying anonymous API", ref, exc)

    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    resp = requests.get(url, headers=UA, timeout=TIMEOUT, stream=True)
    resp.raise_for_status()
    blob = resp.content
    if blob[:2] == b"PK":
        return _unzip(key, blob)
    # Some datasets return a bare csv
    name = resp.headers.get("Content-Disposition", "").split("filename=")[-1].strip('"')
    return [_save_bytes(key, name or "dataset.csv", blob)]


def direct_zip(url: str, key: str) -> list[Path]:
    """Download and extract a zip from a direct URL."""
    resp = requests.get(url, headers=UA, timeout=TIMEOUT)
    resp.raise_for_status()
    return _unzip(key, resp.content)


def direct_file(url: str, key: str, filename: str) -> Path:
    """Download a single file from a direct URL."""
    resp = requests.get(url, headers=UA, timeout=TIMEOUT)
    resp.raise_for_status()
    return _save_bytes(key, filename, resp.content)


def openml_download(data_id: int, key: str, filename: str) -> Path:
    """Fetch an OpenML dataset by id and persist it as CSV."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=data_id, as_frame=True).frame
    path = _raw_dir(key) / filename
    df.to_csv(path, index=False)
    return [path]  # type: ignore[return-value]


def hf_parquet(repo: str, key: str) -> list[Path]:
    """Download the parquet shards of a Hugging Face dataset (no auth needed)."""
    api = f"https://huggingface.co/api/datasets/{repo}/tree/main/data"
    files = requests.get(api, headers=UA, timeout=TIMEOUT).json()
    out: list[Path] = []
    for meta in files:
        name = Path(meta["path"]).name
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/data/{name}"
        resp = requests.get(url, headers=UA, timeout=TIMEOUT)
        resp.raise_for_status()
        out.append(_save_bytes(key, name, resp.content))
    return out


def _manual(key: str, url: str, expected: str) -> DatasetResult:
    """Step 2 of the ladder: instruct the user where to place a manual file."""
    dest = _raw_dir(key)
    log.warning(
        "MANUAL STEP for %s:\n  1. Open %s\n  2. Download the file\n"
        "  3. Place it as: %s/%s",
        key,
        url,
        dest,
        expected,
    )
    return DatasetResult(key, "manual", source=url, notes=f"place file at {dest}/{expected}")


# --------------------------------------------------------------------------- #
# Per-dataset fetchers (each returns DatasetResult, never raises)
# --------------------------------------------------------------------------- #

def fetch_student_depression() -> DatasetResult:
    key = "student_depression"
    if files := _existing_files(key, (".csv",)):
        return DatasetResult(key, "present", "kaggle", files)
    try:
        files = kaggle_download("hopesb/student-depression-dataset", key)
        files = [f for f in files if f.suffix == ".csv"]
        return DatasetResult(key, "downloaded", "kaggle:hopesb/student-depression-dataset", files)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_sentiment_mh() -> DatasetResult:
    key = "sentiment_mh"
    if files := _existing_files(key, (".csv",)):
        return DatasetResult(key, "present", "kaggle", files)
    try:
        files = [
            f
            for f in kaggle_download("suchintikasarkar/sentiment-analysis-for-mental-health", key)
            if f.suffix == ".csv"
        ]
        return DatasetResult(key, "downloaded", "kaggle:suchintikasarkar/...", files)
    except Exception as exc:  # # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_dass42() -> DatasetResult:
    key = "dass42"
    if files := _existing_files(key, (".csv", ".txt")):
        return DatasetResult(key, "present", "openpsychometrics", files)
    try:
        files = direct_zip(
            "https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip", key
        )
        return DatasetResult(key, "downloaded", "openpsychometrics.org", files)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_osmi_2014() -> DatasetResult:
    key = "osmi_2014"
    if files := _existing_files(key, (".csv",)):
        return DatasetResult(key, "present", "openml", files)
    try:
        path = openml_download(43674, key, "osmi_2014.csv")
        return DatasetResult(key, "downloaded", "openml:43674", [path])  # type: ignore[list-item]
    except Exception:
        try:  # alternate id listed in the spec
            path = openml_download(43664, key, "osmi_2014.csv")
            return DatasetResult(key, "downloaded", "openml:43664", [path])  # type: ignore[list-item]
        except Exception as exc:  # noqa: BLE001
            return DatasetResult(key, "failed", notes=str(exc))


def fetch_drug_reviews() -> DatasetResult:
    """Drugs.com reviews: Kaggle mirror (UCI static zip URL is 404 as of 2026-10)."""
    key = "drug_reviews"
    if files := _existing_files(key, (".tsv", ".csv")):
        return DatasetResult(key, "present", "kaggle-mirror", files)
    try:
        files = [
            f
            for f in kaggle_download("jessicali9530/kuc-hackathon-winter-2018", key)
            if f.suffix in {".tsv", ".csv"}
        ]
        return DatasetResult(
            key, "downloaded", "kaggle:jessicali9530/kuc-hackathon-winter-2018", files
        )
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(
            key,
            "failed",
            notes=f"{exc} | manual: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018",
        )


def fetch_fer2013() -> DatasetResult:
    key = "fer2013"
    if list(_raw_dir(key).rglob("*.jpg")):
        return DatasetResult(key, "present", "kaggle", list(_raw_dir(key).rglob("*.jpg"))[:1])
    try:
        files = kaggle_download("msambare/fer2013", key)
        return DatasetResult(key, "downloaded", "kaggle:msambare/fer2013", files)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_dreaddit() -> DatasetResult:
    key = "dreaddit"
    if files := _existing_files(key, (".parquet", ".csv")):
        return DatasetResult(key, "present", "huggingface", files)
    try:
        files = hf_parquet("andreagasparini/dreaddit", key)
        return DatasetResult(key, "downloaded", "hf:andreagasparini/dreaddit", files)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_osmi_2017() -> DatasetResult:
    key = "osmi_2017_21"
    if files := _existing_files(key, (".csv",)):
        return DatasetResult(key, "present", "kaggle", files)
    try:
        files = [
            f
            for f in kaggle_download("osmihelp/osmi-mental-health-in-tech-survey-2017", key)
            if f.suffix == ".csv"
        ]
        return DatasetResult(key, "downloaded", "kaggle:osmihelp/...2017", files)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_student_small() -> DatasetResult:
    key = "student_depression_small"
    if files := _existing_files(key, (".csv",)):
        return DatasetResult(key, "present", "kaggle", files)
    try:
        files = [
            f
            for f in kaggle_download("ikynahidwin/depression-student-dataset", key)
            if f.suffix == ".csv"
        ]
        return DatasetResult(key, "downloaded", "kaggle:ikynahidwin/...", files)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(key, "failed", notes=str(exc))


def fetch_druglib() -> DatasetResult:
    """Tier-C optional dataset (UCI 461). Skipped cleanly if unreachable."""
    key = "druglib"
    if files := _existing_files(key, (".txt", ".tsv", ".csv")):
        return DatasetResult(key, "present", "uci", files)
    urls = [
        "https://archive.ics.uci.edu/static/public/461/druglib+review+dataset.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00461/DrugLibTest.zip",
    ]
    for url in urls:
        try:
            files = direct_zip(url, key)
            return DatasetResult(key, "downloaded", url, files)
        except Exception:  # noqa: BLE001
            continue
    return DatasetResult(
        key,
        "manual",
        source="https://archive.ics.uci.edu/dataset/461/druglib+review+dataset",
        notes="optional tier-C; download manually or skip",
    )


FETCHERS: dict[str, Callable[[], DatasetResult]] = {
    "student_depression": fetch_student_depression,
    "sentiment_mh": fetch_sentiment_mh,
    "dass42": fetch_dass42,
    "osmi_2014": fetch_osmi_2014,
    "drug_reviews": fetch_drug_reviews,
    "fer2013": fetch_fer2013,
    "dreaddit": fetch_dreaddit,
    "osmi_2017_21": fetch_osmi_2017,
    "student_depression_small": fetch_student_small,
    "druglib": fetch_druglib,
}


def fetch_all(*, allow_synthetic: bool = True) -> list[DatasetResult]:
    """Run the acquisition ladder for every configured dataset."""
    cfg = load_config()
    results: list[DatasetResult] = []
    for key in cfg["datasets"]:
        fetcher = FETCHERS.get(key)
        if fetcher is None:  # pragma: no cover - config drift guard
            results.append(DatasetResult(key, "failed", notes="no fetcher defined"))
            continue
        log.info("fetching %s ...", key)
        res = fetcher()
        if res.status == "manual" or (not res.ok and allow_synthetic):
            if allow_synthetic:
                from mindsense.data.synthetic import make_synthetic

                res = make_synthetic(key, prior=res)
        log.info("  -> %s (%s)", res.status, res.source or res.notes[:80])
        results.append(res)
    return results


def redownload(key: str) -> None:
    """Force a fresh download of one dataset (used by tests/scripts)."""
    shutil.rmtree(_raw_dir(key), ignore_errors=True)
    FETCHERS[key]()
