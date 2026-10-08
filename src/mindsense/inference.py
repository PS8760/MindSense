"""Inference facade: the only module the Streamlit app may call for predictions.

Contract (also recorded in ``docs/DECISIONS.md``)
-------------------------------------------------
* Pure Python — **no** streamlit import here; every function is unit-testable.
* Model artifacts live in ``models/`` and are registered in
  ``models/metadata.json`` under ``artifacts``::

      {"artifacts": {"risk": "risk_tabular.joblib", "text": "text_classifier.joblib", ...},
       "model_versions": {"risk": "exp06-v1", ...},
       "risk_tiers": {"lower": 0.35, "elevated": 0.65},
       "prognosis": {"selected_items": [3, 5, 10, ...]}}

* Every ``predict_*`` / ``analyze_*`` call returns ``{"available": bool}``.
  When an artifact (or its metadata) is missing the result is
  ``available=False`` with a human-readable ``message`` — never an
  exception — so the app renders a helpful empty state instead of crashing.
* ``ValueError`` is raised only for *invalid user input*, with a message
  safe to show in the UI.
* Artifacts load lazily and are cached; :func:`reset_cache` clears them
  (used by tests).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from mindsense.utils.io import load_config, load_json, repo_path

__all__ = [
    "availability",
    "analyze_face",
    "analyze_text",
    "extract_entities",
    "predict_intervention",
    "predict_risk",
    "predict_severity",
    "prognosis_info",
    "reset_cache",
]

_MODELS_DIR = repo_path("models")

_DEFAULT_ARTIFACTS = {
    "risk": "risk_tabular.joblib",
    "intervention": "intervention.joblib",
    "prognosis": "prognosis.joblib",
    "text": "text_classifier.joblib",
    "face": "face_emotion.onnx",
}

_TRAIN_HINT = "Model not trained yet — run ``make train`` (Experiments 3–7)."

#: Feature order used when metadata does not specify one (matches
#: ``mindsense.data.harmonize.HARMONIZED_COLUMNS`` minus ids/labels).
_RISK_FEATURES = [
    "population",
    "age",
    "gender",
    "sleep_hours",
    "stress_pressure_score",
    "satisfaction_score",
    "work_study_hours",
    "financial_stress",
    "family_history",
    "support_available",
    "diet_quality",
]

_RISK_TIERS_DEFAULT = {"lower": 0.35, "elevated": 0.65}


def reset_cache() -> None:
    """Drop cached metadata/artifacts (tests, retraining within a session)."""
    _metadata.cache_clear()
    _load.cache_clear()


# --------------------------------------------------------------------------- #
# artifact loading
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _metadata() -> dict[str, Any]:
    path = _MODELS_DIR / "metadata.json"
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:  # pragma: no cover - corrupt metadata is treated as absent
        return {}


@functools.lru_cache(maxsize=8)
def _load(key: str) -> Any:
    """Return the loaded artifact for ``key`` or None."""
    meta = _metadata()
    filename = meta.get("artifacts", {}).get(key) or _DEFAULT_ARTIFACTS.get(key)
    if not filename:
        return None
    path = _MODELS_DIR / filename
    if not path.exists():
        return None
    if path.suffix == ".onnx":
        return str(path)  # onnxruntime session is built lazily in analyze_face
    import joblib

    try:
        return joblib.load(path)
    except Exception:  # pragma: no cover - unreadable artifact ≈ absent
        return None


def _unavailable(message: str = _TRAIN_HINT) -> dict[str, Any]:
    return {"available": False, "message": message}


def _model_version(key: str) -> str:
    return str(_metadata().get("model_versions", {}).get(key, "unversioned"))


def _artifact_path(key: str) -> Path | None:
    meta = _metadata()
    filename = meta.get("artifacts", {}).get(key) or _DEFAULT_ARTIFACTS.get(key)
    if not filename:
        return None
    path = _MODELS_DIR / filename
    return path if path.exists() else None


def availability() -> dict[str, dict[str, Any]]:
    """Status of every module, for the Home page status strip and Lab Results."""
    status: dict[str, dict[str, Any]] = {}
    for key, label in (
        ("risk", "Lifestyle risk model (Exp 6)"),
        ("intervention", "Intervention-likelihood (Exp 4)"),
        ("prognosis", "DASS severity prognosis (Exp 4)"),
        ("text", "Text mental-state classifier (Exp 7)"),
        ("face", "Face emotion cue (Exp 3)"),
    ):
        path = _artifact_path(key)
        status[key] = {
            "label": label,
            "available": path is not None,
            "detail": f"artifact: {path.name}" if path else _TRAIN_HINT,
            "version": _model_version(key),
        }
    try:
        import mindsense.nlp.entities  # noqa: F401

        entities_ok, entities_detail = True, "mindsense.nlp.entities loaded"
    except Exception as exc:  # pragma: no cover - depends on build order
        entities_ok, entities_detail = False, f"entity module not ready ({type(exc).__name__})"
    status["entities"] = {
        "label": "Clinical entity extraction (Exp 5)",
        "available": entities_ok,
        "detail": entities_detail,
        "version": _model_version("entities"),
    }
    return status


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _require(features: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in features or features[k] is None]
    if missing:
        raise ValueError(f"Missing input(s): {', '.join(missing)}")


def _validate_risk_features(features: Mapping[str, Any]) -> None:
    _require(features, ["population", "age", "gender"])
    if str(features["population"]) not in {"student", "professional"}:
        raise ValueError("population must be 'student' or 'professional'")
    if str(features["gender"]) not in {"male", "female", "other"}:
        raise ValueError("gender must be male, female or other")
    age = float(features["age"])
    if not 5 <= age <= 120:
        raise ValueError(f"age must be between 5 and 120, got {age}")
    ranges = {
        "sleep_hours": (2.0, 16.0),
        "stress_pressure_score": (0.0, 5.0),
        "satisfaction_score": (0.0, 5.0),
        "work_study_hours": (0.0, 24.0),
        "financial_stress": (0.0, 5.0),
        "family_history": (0.0, 1.0),
        "support_available": (0.0, 2.0),
        "diet_quality": (0.0, 2.0),
    }
    for field, (lo, hi) in ranges.items():
        value = features.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue  # unknown-for-population stays NaN (harmonize contract)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric, got {value!r}") from exc
        if not lo <= numeric <= hi:
            raise ValueError(f"{field} must be within [{lo}, {hi}], got {numeric}")


def _risk_row(features: Mapping[str, Any]) -> pd.DataFrame:
    order = list(_metadata().get("feature_order", _RISK_FEATURES))
    row = {k: features.get(k, float("nan")) for k in order}
    frame = pd.DataFrame([row])
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["population"] = frame["population"].astype(str)
    frame["gender"] = frame["gender"].astype(str)
    return frame


def _tier(probability: float) -> tuple[str, dict[str, float]]:
    tiers = {**_RISK_TIERS_DEFAULT, **_metadata().get("risk_tiers", {})}
    if probability >= tiers["elevated"]:
        label = "higher"
    elif probability >= tiers["lower"]:
        label = "elevated"
    else:
        label = "lower"
    return label, tiers


def _contributions(model: Any, row: pd.DataFrame) -> list[dict[str, Any]]:
    """Approximate per-feature contributions when the model exposes weights."""
    means = _metadata().get("feature_means")
    weights: Any = getattr(model, "coef_", None)
    if weights is None:
        weights = getattr(model, "feature_importances_", None)
    if weights is None and hasattr(model, "steps"):  # sklearn pipeline
        try:
            last = model.steps[-1][1]
            weights = getattr(last, "coef_", None)
            if weights is None:
                weights = getattr(last, "feature_importances_", None)
        except Exception:  # pragma: no cover
            weights = None
    if weights is None:
        return []
    flat = getattr(weights, "ravel", lambda: weights)()
    columns = list(row.columns)
    if len(flat) != len(columns):
        return []
    out: list[dict[str, Any]] = []
    for col, weight in zip(columns, flat, strict=True):
        value = row.iloc[0][col]
        if pd.isna(value):
            continue
        baseline = float(means.get(col, 0.0)) if means else 0.0
        try:
            contribution = float(weight) * (float(value) - baseline)
        except (TypeError, ValueError):
            continue
        out.append({"feature": col, "value": float(value), "contribution": contribution})
    out.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return out[:8]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def predict_risk(features: Mapping[str, Any]) -> dict[str, Any]:
    """Lifestyle → elevated-risk probability, tier and explanations (Exp 6).

    ``features`` uses the harmonized schema; unknown-for-population fields
    may be ``None``/NaN. Returns ``available=False`` until the artifact exists.
    """
    _validate_risk_features(features)
    model = _load("risk")
    if model is None:
        return _unavailable()
    row = _risk_row(features)
    try:
        proba = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(row)[0])
    except Exception as exc:
        return _unavailable(f"risk model failed to run ({exc})")
    proba = min(max(proba, 0.0), 1.0)
    tier, thresholds = _tier(proba)
    return {
        "available": True,
        "probability": proba,
        "tier": tier,
        "thresholds": thresholds,
        "contributions": _contributions(model, row),
        "model_version": _model_version("risk"),
        "input_echo": {k: (None if pd.isna(v) else v) for k, v in row.iloc[0].items()},
    }


def predict_intervention(features: Mapping[str, Any]) -> dict[str, Any]:
    """Lifestyle/workplace inputs → intervention-likelihood (Exp 4, associational)."""
    _validate_risk_features(features)
    model = _load("intervention")
    if model is None:
        return _unavailable()
    row = _risk_row(features)
    try:
        proba = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(row)[0])
    except Exception as exc:
        return _unavailable(f"intervention model failed to run ({exc})")
    return {
        "available": True,
        "likelihood": min(max(proba, 0.0), 1.0),
        "model_version": _model_version("intervention"),
        "warning": "Associational model — not evidence that changing a factor causes change.",
    }


def predict_severity(
    answers: Mapping[int, int] | Sequence[int],
    demographics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact DASS screener → depression-severity band with confidence (Exp 4)."""
    meta = _metadata()
    selected = list(meta.get("prognosis", {}).get("selected_items", []))
    budget = int(load_config()["prognosis"]["dass_item_budget"])
    if isinstance(answers, Mapping):
        if not selected:
            selected = sorted(int(k) for k in answers)
        values = [int(answers[i]) for i in selected]
    else:
        values = [int(v) for v in answers]
        if selected and len(values) != len(selected):
            raise ValueError(
                f"expected {len(selected)} answers (selected items), got {len(values)}"
            )
    if len(values) != budget and not selected:
        raise ValueError(f"expected {budget} answers, got {len(values)}")
    for i, v in enumerate(values, start=1):
        if not 0 <= v <= 3:
            raise ValueError(f"answer {i} must be 0–3, got {v}")

    model = _load("prognosis")
    if model is None:
        return _unavailable()
    payload = pd.DataFrame([{"items": values, **dict(demographics or {})}])
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(payload)[0]
            classes = [str(c) for c in getattr(model, "classes_", range(len(proba)))]
            idx = max(range(len(proba)), key=lambda i: proba[i])
            band, confidence = classes[idx], float(proba[idx])
        else:
            band, confidence = str(model.predict(payload)[0]), None
    except Exception as exc:
        return _unavailable(f"prognosis model failed to run ({exc})")
    return {
        "available": True,
        "band": band,
        "confidence": confidence,
        "selected_items": selected,
        "model_version": _model_version("prognosis"),
    }


def prognosis_info() -> dict[str, Any]:
    """Compact-screener metadata: selected items, budget, model availability.

    The item *selection* is Experiment 4's output, stored in
    ``models/metadata.json`` under ``prognosis.selected_items``.
    """
    meta = _metadata()
    budget = int(load_config()["prognosis"]["dass_item_budget"])
    selected = [int(i) for i in meta.get("prognosis", {}).get("selected_items", [])]
    return {
        "selected_items": selected,
        "budget": budget,
        "model_available": _artifact_path("prognosis") is not None,
    }


def analyze_text(text: str) -> dict[str, Any]:
    """Free text → class probabilities, influential words and crisis signal (Exp 7)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text to analyze.")
    if len(text) > 50_000:
        raise ValueError("Text is too long (50 000 characters max).")
    from mindsense.screening.crisis import text_crisis

    model = _load("text")
    if model is None:
        return {**_unavailable(), "crisis": text_crisis(text)}
    try:
        proba = model.predict_proba([text])[0]
        classes = [str(c) for c in model.classes_]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba, strict=True)}
        top_class = max(probabilities, key=probabilities.__getitem__)
    except Exception as exc:
        return {**_unavailable(f"text model failed to run ({exc})"), "crisis": text_crisis(text)}

    return {
        "available": True,
        "top_class": top_class,
        "probabilities": probabilities,
        "top_words": _top_words(model, text, top_class),
        "crisis": text_crisis(text, class_probabilities=probabilities),
        "model_version": _model_version("text"),
    }


def _top_words(model: Any, text: str, top_class: str, k: int = 8) -> list[dict[str, Any]]:
    """Words most responsible for ``top_class`` (coef × tf-idf weight)."""
    try:
        vectorizer = model.named_steps["vectorizer"]
        classifier = model.named_steps["classifier"]
        class_index = list(classifier.classes_).index(top_class)
        coef = classifier.coef_[class_index]
    except Exception:
        return []
    vector = vectorizer.transform([text])
    weights = vector.toarray()[0]
    nonzero = weights.nonzero()[0]
    reverse_vocab = {pos: term for term, pos in vectorizer.vocabulary_.items()}
    scored = [
        {"word": reverse_vocab[idx], "weight": float(coef[idx] * weights[idx])}
        for idx in nonzero
        if idx in reverse_vocab
    ]
    scored.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return scored[:k]


def extract_entities(text: str) -> dict[str, Any]:
    """Clinical-style entities from text (Exp 5, ``mindsense.nlp.entities``)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text to extract entities from.")
    try:
        from mindsense.nlp import entities as entity_module
    except Exception as exc:
        return _unavailable(f"entity extraction not ready ({type(exc).__name__})")

    extractor = getattr(entity_module, "extract_entities", None)
    try:
        if extractor is None:
            extractor_cls = getattr(entity_module, "EntityExtractor", None)
            if extractor_cls is None:
                return _unavailable("entity module exposes neither extract_entities nor EntityExtractor")
            extractor = extractor_cls().extract
        result = extractor(text)
    except Exception as exc:
        return _unavailable(f"entity extraction failed ({exc})")

    items = _normalise_entities(result)
    return {"available": True, "entities": items, "model_version": _model_version("entities")}


def _normalise_entities(result: Any) -> list[dict[str, Any]]:
    """Accept list[dict], list[tuple] or spacy-style spans and normalise them."""
    entities: list[dict[str, Any]] = []
    for item in result or []:
        if isinstance(item, Mapping):
            entities.append(
                {
                    "text": str(item.get("text", "")),
                    "label": str(item.get("label", item.get("type", "ENTITY"))),
                    "start": int(item.get("start", -1)),
                }
            )
        elif isinstance(item, tuple) and len(item) >= 2:
            entities.append({"text": str(item[0]), "label": str(item[1]), "start": -1})
        else:
            entities.append(
                {"text": str(getattr(item, "text", item)), "label": str(getattr(item, "label_", "ENTITY")),
                 "start": int(getattr(item, "start", -1))}
            )
    return entities


def analyze_face(image: Any) -> dict[str, Any]:
    """Face crop → emotion probabilities + mood cue (Exp 3, ONNX, optional).

    Requires ``ENABLE_FACE=1`` **and** the ``models/face_emotion.onnx``
    artifact; otherwise returns ``available=False`` with the exact reason.
    """
    if os.environ.get("ENABLE_FACE", "0") != "1":
        return _unavailable("Face module is switched off — start the app with ``ENABLE_FACE=1``.")
    path = _artifact_path("face")
    if path is None:
        return _unavailable(_TRAIN_HINT)
    try:
        import cv2
        import numpy as np  # noqa: F401
        import onnxruntime as ort  # noqa: F401
    except Exception as exc:
        return _unavailable(f"face runtime missing ({type(exc).__name__})")

    image_array = _to_bgr(image)
    if image_array is None:
        raise ValueError("Could not read the image — please upload a clear photo.")
    cfg = load_config()["image_model"]
    height, width = cfg["input_size"]
    resized = cv2.resize(image_array, (width, height))
    blob = resized.astype("float32") / 255.0
    blob = blob.transpose(2, 0, 1)[None]  # NCHW

    session = _face_session(str(path))
    input_name = session.get_inputs()[0].name
    try:
        logits = session.run(None, {input_name: blob})[0]
    except Exception as exc:
        return _unavailable(f"face model failed to run ({exc})")

    scores = _softmax(logits[0])
    classes = list(cfg["emotion_classes"])
    probabilities = {
        cls: round(float(score), 4) for cls, score in zip(classes, scores, strict=False)
    }
    top_emotion = max(probabilities, key=probabilities.__getitem__)
    cue = next(
        (group for group, members in cfg["mood_cue_groups"].items() if top_emotion in members),
        "neutral",
    )
    return {
        "available": True,
        "top_emotion": top_emotion,
        "probabilities": probabilities,
        "mood_cue": cue,
        "gradcam": None,
        "model_version": _model_version("face"),
        "disclaimer": "Screening aid only — not a diagnostic tool.",
    }


@functools.lru_cache(maxsize=2)
def _face_session(path: str) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _to_bgr(image: Any) -> Any | None:
    """Normalise upload/camera payloads to an HxWx3 uint8 BGR array."""
    import cv2
    import numpy as np

    if isinstance(image, (bytes, bytearray)):
        array = np.frombuffer(bytes(image), dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return array
    if hasattr(image, "read"):  # PIL image
        array = np.array(image.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return None


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = np.asarray(values, dtype="float64")
    shifted = shifted - shifted.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
