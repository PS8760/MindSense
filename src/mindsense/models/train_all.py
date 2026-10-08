"""``make train`` entry point — orchestrates every implemented trainer.

Exp 3 (image, MobileNetV2 → FER-2013) is live here. Experiments 4/6/7 are
reported as not-built-yet so the CLI is honest about what actually ran;
they are implemented in later milestones of the same build order.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MindSense model training (all experiments).")
    parser.add_argument("--quick", action="store_true", help="tiny smoke run (CI/local sanity).")
    parser.add_argument("--seed", type=int, default=42, help="GLOBAL_SEED (default 42).")
    args = parser.parse_args(argv)

    summary: dict[str, str] = {}
    started = time.perf_counter()

    try:
        result = _train_image(quick=args.quick, seed=args.seed)
        summary["exp03_image"] = (
            f"ok  acc={result['test']['accuracy']:.3f} macroF1={result['test']['macro_f1']:.3f}"
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        summary["exp03_image"] = f"skipped  ({exc})"

    for exp in ("exp04_tabular", "exp06_text", "exp07_prognosis"):
        summary[exp] = "skipped  (not built yet — later milestone)"

    log.info("training finished in %.1fs", time.perf_counter() - started)
    for name, status in summary.items():
        log.info("  %-16s %s", name, status)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
