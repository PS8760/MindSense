"""Zoo subprocess isolation — regression guard for the macOS OpenMP segfault.

``mindsense.models.image`` (torch) and ``mindsense.models.zoo`` (sklearn /
xgboost) must never share an interpreter: both bundle OpenMP runtimes and
fitting xgboost after torch is imported segfaults/hangs on macOS.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def test_zoo_module_is_torch_free() -> None:
    before = set(sys.modules)
    import mindsense.models.zoo  # noqa: F401

    newly_added = set(sys.modules) - before
    assert not {"torch", "mindsense.models.image"} & newly_added


def test_zoo_cli_runs_in_child_process_and_reports_every_head() -> None:
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as td:
        frames = Path(td) / "frames.npz"
        out = Path(td) / "zoo.json"
        np.savez(
            frames,
            train_emb=rng.rand(120, 16).astype("float32"),
            train_y=rng.randint(0, 7, 120),
            val_emb=rng.rand(30, 16).astype("float32"),
            val_y=rng.randint(0, 7, 30),
            test_emb=rng.rand(30, 16).astype("float32"),
            test_y=rng.randint(0, 7, 30),
        )
        proc = subprocess.run(
            [sys.executable, "-m", "mindsense.models.zoo", str(frames), str(out)],
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert proc.returncode == 0, proc.stderr
        results = json.loads(out.read_text())
    assert set(results) == {"logreg", "forest", "xgboost"}
    for entry in results.values():
        for split in ("val", "test"):
            assert {"accuracy", "macro_f1", "support", "n"} <= set(entry[split])
    assert (
        "torch"
        not in subprocess.run(
            [
                sys.executable,
                "-c",
                "import mindsense.models.zoo, sys; print('torch' in sys.modules)",
            ],
            capture_output=True,
            text=True,
        ).stdout
    )
