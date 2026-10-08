"""Zoo head comparison — a self-contained, torch-free subprocess.

The zoo heads (scikit-learn logreg/forest, xgboost) each bundle their own
OpenMP runtime. On macOS, fitting them in the same interpreter that has
already imported torch segfaults (or hangs) on a duplicate ``libomp`` —
reproduced on arm64 with torch 2.14.1 + xgboost 3.0.0 in every import order.
``mindsense.models.image.train_face_model`` therefore runs this module as a
child process that imports **no** torch (only numpy/sklearn/xgboost, which
coexist fine), and reads results back as JSON.

Contract
--------
``python -m mindsense.models.zoo <frames.npz> <out.json>``

``frames.npz`` must contain ``train_emb``, ``train_y``, ``val_emb``,
``val_y``, ``test_emb``, ``test_y`` (numpy arrays). ``out.json`` receives
one metrics block per zoo model, keyed by name. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mindsense.models.metrics import (
    classification_metrics,
    emotion_classes_from_config,
)
from mindsense.utils.logging import get_logger

ZOO = ("logreg", "forest", "xgboost")

log = get_logger("mindsense.zoo")


def _fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[Any, float]:
    """Fit one zoo model on embeddings; returns ``(estimator, fit seconds)``."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    t0 = time.perf_counter()
    if name == "logreg":
        model = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0, random_state=seed
        )
        model.fit(x_train, y_train)
    elif name == "forest":
        # Threads, not processes: loky multiprocessing workers are what made
        # earlier in-process runs fragile (orphanable worker processes + leaked
        # semaphores on macOS). Threads keep the child fully self-contained.
        from joblib import parallel_config

        model = RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"
        )
        with parallel_config(prefer="threads", n_jobs=-1):
            model.fit(x_train, y_train)
    elif name == "xgboost":
        model = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=seed,
        )
        model.fit(x_train, y_train)
    else:  # pragma: no cover - unreachable
        raise ValueError(f"unknown zoo model {name}")
    return model, time.perf_counter() - t0


def _predict_array(estimator: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.predict(x))


def run(frames_path: Path, out_path: Path, *, seed: int = 42) -> int:
    """Fit every zoo model on shared embeddings and score each on val + test."""
    with np.load(frames_path) as data:
        frames: dict[str, dict[str, np.ndarray]] = {}
        for split in ("train", "val", "test"):
            frames[split] = {
                "emb": np.asarray(data[f"{split}_emb"]),
                "y": np.asarray(data[f"{split}_y"]),
            }
    classes = emotion_classes_from_config()
    x_train, y_train = frames["train"]["emb"], frames["train"]["y"]
    comparison: dict[str, dict[str, Any]] = {}
    for name in ZOO:
        estimator, fit_s = _fit_model(name, x_train, y_train, seed)
        pred_val = _predict_array(estimator, frames["val"]["emb"])
        pred_test = _predict_array(estimator, frames["test"]["emb"])
        comparison[name] = {
            "fit_seconds": round(fit_s, 2),
            "val": classification_metrics(frames["val"]["y"], pred_val, classes),
            "test": classification_metrics(frames["test"]["y"], pred_test, classes),
        }
        log.info(
            "compare %-8s val acc %.3f macro-F1 %.3f | test acc %.3f macro-F1 %.3f (%.1fs)",
            name,
            comparison[name]["val"]["accuracy"],
            comparison[name]["val"]["macro_f1"],
            comparison[name]["test"]["accuracy"],
            comparison[name]["test"]["macro_f1"],
            fit_s,
        )
    out_path.write_text(json.dumps(comparison))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames_npz", type=Path)
    parser.add_argument("out_json", type=Path)
    args = parser.parse_args(argv)
    return run(args.frames_npz, args.out_json)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main(sys.argv[1:]))
