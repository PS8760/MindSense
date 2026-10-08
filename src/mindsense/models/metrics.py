"""Torch-free evaluation metrics (shared by the face trainer and zoo subprocess).

Kept import-light: no torch/transformers at module scope, so the zoo child
interpreter (``python -m mindsense.models.zoo``) never pulls in torch or its
OpenMP runtime. Class labels come from ``config.yaml`` via ``load_config``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mindsense.utils.io import load_config


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Accuracy, macro/micro precision, macro recall, F1 variants + per-class support."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    classes = list(classes)
    macro = {"average": "macro", "zero_division": 0}
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, **macro)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, **macro)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, **macro)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "per_class_recall": {
            cls: round(float(r), 4)
            for cls, r in zip(
                classes,
                recall_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0),
                strict=True,
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist(),
        "support": {cls: int((y_true == i).sum()) for i, cls in enumerate(classes)},
        "n": int(len(y_true)),
    }


def emotion_classes_from_config() -> list[str]:
    """Emotion class labels straight from config (no ``mindsense.models.image`` import)."""
    return list(load_config()["image_model"]["emotion_classes"])