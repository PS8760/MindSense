"""Unit tests for the Exp 3 image model (``mindsense.models.image``).

Heavy frameworks are guarded with ``pytest.importorskip`` so the suite stays
runnable in a runtime-only environment. Tests never hit the network: the
model is built with ``pretrained=False`` and the FER-like fixture is written
to a temp directory (parquet manifest + real JPEGs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402  (guarded by importorskip)


def _fer_frame(root: Path, cls: str, *, n: int, offset: int) -> pd.DataFrame:
    """A tiny manifest slice: ``n`` solid-colour 48x48 JPEGs for one emotion."""
    import PIL.Image

    rows = []
    for i in range(n):
        img = PIL.Image.fromarray(np.full((48, 48, 3), offset + i, dtype=np.uint8))
        path = root / f"{cls}_{i}.jpg"
        img.save(path)
        rows.append({"path": str(path), "emotion": cls, "data_source": "real"})
    return pd.DataFrame(rows)


def _patch_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module at a tiny 48x48 image_model config."""

    monkeypatch.setattr(
        "mindsense.models.image.load_config",
        lambda: {
            "image_model": {
                "input_size": [48, 48],
                "emotion_classes": [
                    "angry",
                    "disgust",
                    "fear",
                    "happy",
                    "sad",
                    "surprise",
                    "neutral",
                ],
                "mood_cue_groups": {
                    "negative-affect": ["angry", "disgust", "fear", "sad"],
                    "neutral": ["neutral"],
                    "positive-affect": ["happy", "surprise"],
                },
                "onnx_max_mb": 20,
            }
        },
    )


def _make_tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Repo-shaped tmp tree and redirects ``mindsense.models.image`` IO there."""
    from mindsense.models import image

    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    images = tmp_path / "images_fer"
    images.mkdir()

    classes = image.emotion_classes()
    frames = []
    for i, cls in enumerate(classes):
        frame = _fer_frame(images, cls, n=12, offset=i * 40)
        frame["split"] = "test" if i == 0 else "train"
        frames.append(frame)
    manifest = pd.concat(frames)
    manifest.to_parquet(tmp_path / "data" / "processed" / "fer2013_manifest.parquet")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    from mindsense.utils import io

    monkeypatch.setattr(image, "repo_path", fake_repo_path)
    monkeypatch.setattr(image, "save_json", io.save_json)
    monkeypatch.setattr(image, "load_json", io.load_json)
    return tmp_path


def test_mood_cue_mapping(monkeypatch) -> None:
    from mindsense.models import image

    _patch_config(monkeypatch)
    assert image.mood_cue_of("sad") == "negative-affect"
    assert image.mood_cue_of("neutral") == "neutral"
    assert image.mood_cue_of("surprise") == "positive-affect"
    assert image.mood_cue_of("confused") == "neutral"  # unknown → neutral fallback


def test_model_forward_shape_and_internal_normalisation(monkeypatch) -> None:
    from mindsense.models.image import FaceEmotionNet

    _patch_config(monkeypatch)
    model = FaceEmotionNet(num_classes=7, pretrained=False).eval()
    out = model(torch.rand(2, 3, 48, 48))
    assert out.shape == (2, 7)
    gray = torch.full((1, 3, 48, 48), 0.5)
    with torch.no_grad():
        normalised = (gray - model.pixel_mean) / model.pixel_std
    expected = (0.5 - 0.485) / 0.229
    assert abs(float(normalised[0, 0, 0, 0]) - expected) < 1e-6


def test_inverse_frequency_weights(monkeypatch) -> None:
    from mindsense.models.image import _inverse_frequency_weights

    _patch_config(monkeypatch)
    w = _inverse_frequency_weights(np.array([7, 7, 1]))
    assert abs(float(sum(w.values())) - 3.0) < 1e-9  # normalised to mean 1
    assert w["2"] > w["0"]


def test_export_onnx_parity(monkeypatch, tmp_path: Path) -> None:
    from mindsense.models.image import ONNX_PARITY_TOL, FaceEmotionNet, export_onnx

    _patch_config(monkeypatch)
    model = FaceEmotionNet(num_classes=7, pretrained=False).eval()
    dest = tmp_path / "face.onnx"
    batch = torch.rand(4, 3, 48, 48)
    info = export_onnx(model, dest, input_size=(48, 48), parity_batch=batch)
    assert dest.exists()
    assert info["size_mb"] < 20
    assert info["parity_ok"]
    assert info["parity_max_abs"] <= ONNX_PARITY_TOL


def test_register_face_artifact(tmp_path: Path) -> None:
    from mindsense.models.image import register_face_artifact

    onnx = tmp_path / "face_emotion.onnx"
    onnx.write_bytes(b"fake")
    meta_path = register_face_artifact(
        onnx,
        version="exp03-v1",
        training={"datasets": {"fer2013": "real"}},
        models_dir=tmp_path / "models",
    )
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["artifacts"]["face"] == "face_emotion.onnx"
    assert meta["model_versions"]["face"] == "exp03-v1"
    assert meta["training"]["face"]["datasets"]["fer2013"] == "real"


def test_gradcam_heatmap_shape(monkeypatch) -> None:
    from mindsense.models.image import FaceEmotionNet, gradcam_heatmap

    _patch_config(monkeypatch)
    model = FaceEmotionNet(num_classes=7, pretrained=False).eval()
    batch = torch.rand(1, 3, 48, 48)
    heat = gradcam_heatmap(model, batch, device=torch.device("cpu"))
    assert heat.shape == (48, 48)
    assert np.isfinite(heat).all()
    assert float(heat.min()) >= 0.0
    assert float(heat.max()) <= 1.0 + 1e-6


def test_train_smoke_on_tiny_fixture(tmp_path: Path, monkeypatch) -> None:
    """End-to-end training path on the small fixture (mlp-only, ONNX)."""
    from mindsense.models import image

    _patch_config(monkeypatch)
    root = _make_tmp_repo(tmp_path, monkeypatch)
    result = image.train_face_model(
        models=("mlp",),
        mlp_epochs=2,
        mlp_batch_size=4,
        batch_size=4,
        num_workers=0,
        device="cpu",
        pretrained=False,
        limit_batches=1,
        seed=42,
    )
    assert len(result["comparison"]) == 1 and "mlp" in result["comparison"]
    assert len(result["mlp_history"]) == 2
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1"} <= set(result["test"])
    assert result["onnx"]["parity_ok"]
    assert (root / "models" / "face_emotion.onnx").exists()
    assert (root / "models" / "metadata.json").exists()
    assert (root / "reports" / "metrics" / "exp03_image.json").exists()
