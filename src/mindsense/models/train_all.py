"""``make train`` entry point — orchestrates every implemented trainer.

Exp 3 (image, MobileNetV2 → FER-2013), Exp 4 (tabular risk screen),
Exp 6 (text screening) and Exp 7 (DASS-42 prognosis) all compare several
model types and rank them on validation; the best model is scored on the
held-out test split. A cross-experiment leaderboard of winners is written to
``reports/metrics/leaderboard.json``.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

from mindsense import GLOBAL_SEED
from mindsense.utils.io import repo_path, save_json
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.train")


def _train_image(*, quick: bool, seed: int) -> dict[str, Any]:
    from mindsense.models.image import train_face_model

    models: tuple[str, ...] = ("mlp",) if quick else None
    return train_face_model(
        models=models,
        mlp_epochs=2 if quick else 20,
        mlp_batch_size=32 if quick else 256,
        batch_size=32 if quick else 64,
        num_workers=0 if quick else 4,
        limit_batches=1 if quick else None,
        pretrained=not quick,
        export=not quick,
        seed=seed,
    )


def _run_classifier(result: dict[str, Any]) -> str:
    test = result["test"]
    return (
        f"ok  acc={test['accuracy']:.3f} macroF1={test['macro_f1']:.3f} "
        f"[{result['best']}]"
    )


def _run_regressor(result: dict[str, Any]) -> str:
    test = result["test"]
    return f"ok  rmse={test['rmse_mean']:.3f} n={test['n']} [{result['best']}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MindSense model training (all experiments).")
    parser.add_argument("--quick", action="store_true", help="tiny smoke run (CI/local sanity).")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="GLOBAL_SEED (default 42).")
    args = parser.parse_args(argv)

    summary: dict[str, str] = {}
    leaders: dict[str, str] = {}
    started = time.perf_counter()

    try:
        result = _train_image(quick=args.quick, seed=args.seed)
        best = result["best_by_test_macro_f1"]
        leaders["exp03_image"] = best
        test = result["test"]
        summary["exp03_image"] = (
            f"ok  acc={test['accuracy']:.3f} macroF1={test['macro_f1']:.3f} [{best}]"
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        summary["exp03_image"] = f"skipped  ({exc})"

    try:
        from mindsense.models.tabular import train_tabular

        result = train_tabular(quick=args.quick, seed=args.seed)
        leaders["exp04_tabular"] = result["best"]
        summary["exp04_tabular"] = _run_classifier(result)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        summary["exp04_tabular"] = f"skipped  ({exc})"

    try:
        from mindsense.models.text import train_text

        result = train_text(quick=args.quick, seed=args.seed)
        leaders["exp06_text"] = result["best"]
        summary["exp06_text"] = _run_classifier(result)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        summary["exp06_text"] = f"skipped  ({exc})"

    try:
        from mindsense.models.prognosis import train_prognosis

        result = train_prognosis(quick=args.quick, seed=args.seed)
        leaders["exp07_prognosis"] = result["best"]
        summary["exp07_prognosis"] = _run_regressor(result)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        summary["exp07_prognosis"] = f"skipped  ({exc})"

    save_json(
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "best_model_per_experiment": leaders,
        },
        repo_path("reports", "metrics", "leaderboard.json"),
    )

    log.info("training finished in %.1fs", time.perf_counter() - started)
    for name, status in summary.items():
        log.info("  %-16s %s", name, status)
    log.info("leaderboard: %s", leaders)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())