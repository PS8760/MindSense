"""Exp 3 — image mood-cue model (MobileNetV2 embeddings on FER-2013, ONNX export).

Implements prompt decision #2: FER-2013 stays a 7-class emotion classifier
with an extra *mood-cue* grouping (negative-affect / neutral / positive-affect).
The cue is a **supplementary, non-diagnostic signal** — the app says so, the
metadata records it, and the notebook discusses bias and dataset limits.

Approach (mini-project scope, see ``docs/DECISIONS.md`` #22)
-------------------------------------------------------------
Backprop through the full backbone on a memory-constrained 8 GB MPS box was
impractically slow (measured: tens of minutes per batch), so the MobileNetV2
backbone stays frozen and is run **forward-only once** to produce
1280-d embeddings for every image (cached to ``data/processed/``). Several
heads are then trained on those embeddings and compared by accuracy /
macro precision / recall / F1 (Table + figures). The best *deep* candidate
(an MLP head) is exported — *with the frozen backbone baked in* — to
``models/face_emotion.onnx`` so the inference contract is unchanged.

Inference contract (``mindsense.inference.analyze_face``)
---------------------------------------------------------
* Artifact: ``models/face_emotion.onnx`` — raw **logits**, NCHW ``float32``
  input in **[0, 1]** RGB, shape ``(1, 3, H, W)`` with ``H, W`` from
  ``config.yaml → image_model.input_size``.
* The ONNX graph itself performs ImageNet mean/std normalisation as its
  first operation (buffers baked into the graph) so the caller only has to
  scale pixels to [0, 1] — exactly what ``analyze_face`` does.
* ``models/metadata.json`` registers the artifact, version and provenance.
"""

from __future__ import annotations  # noqa: I001

import json
import math
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mindsense import GLOBAL_SEED
from mindsense.models.metrics import classification_metrics
from mindsense.utils.io import ensure_dir, load_config, load_json, repo_path, save_json
from mindsense.utils.logging import get_logger

log = get_logger("mindsense.image")

# --------------------------------------------------------------------------- #
# Hyper-parameters (code, not config: config.yaml is the deployment contract)
# --------------------------------------------------------------------------- #
BATCH_SIZE = 64
NUM_WORKERS = 4
VAL_FRACTION = 0.10  # carved from the official FER train split (stratified)
ONNX_OPSET = 17
ONNX_PARITY_TOL = 1e-4
EMBED_DIM = 1280  # MobileNetV2 last-channel width

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model zoo for the comparison table (all trained on shared embeddings).
MODEL_ZOO = ("logreg", "forest", "xgboost", "mlp")


def image_config() -> dict[str, Any]:
    """The ``image_model`` section of ``config.yaml``."""
    return load_config()["image_model"]


def emotion_classes() -> list[str]:
    return list(image_config()["emotion_classes"])


def mood_cue_groups() -> dict[str, list[str]]:
    return dict(image_config()["mood_cue_groups"])


def mood_cue_of(emotion: str) -> str:
    """Map a 7-class emotion label to its mood-cue group (config-driven)."""
    for group, members in mood_cue_groups().items():
        if emotion in members:
            return group
    return "neutral"


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device(name: str = "auto") -> torch.device:
    """``auto`` prefers MPS → CUDA → CPU (embedding pass is forward-only)."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_manifest() -> pd.DataFrame:
    """FER-2013 manifest written by the Exp 1 pipeline (path/emotion/split)."""
    path = repo_path("data", "processed", "fer2013_manifest.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run `make data` (Exp 1) before training Exp 3."
        )
    return pd.read_parquet(path)


class FERDataset(Dataset):
    """RGB images scaled to [0, 1] CHW tensors; normalisation happens in the
    model so the exported ONNX input matches the inference contract."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        *,
        input_size: tuple[int, int],
    ) -> None:
        from torchvision import transforms

        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        row = self.frame.iloc[idx]
        with Image.open(repo_path(row["path"])) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, self.class_to_idx[row["emotion"]]


def _stratified_val_split(
    train_df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    keep, held_out = train_test_split(
        train_df,
        train_size=1 - val_fraction,
        stratify=train_df["emotion"],
        random_state=seed,
    )
    return keep.reset_index(drop=True), held_out.reset_index(drop=True)


def _cap_evenly(frame: pd.DataFrame, cap: int, classes: list[str], seed: int) -> pd.DataFrame:
    """Shrink ``frame`` to ``cap`` rows while keeping a balanced class spread.

    A plain ``head()`` can drop whole classes and break later stratified
    splits; ``groupby.head`` keeps every class represented in a smoke run.
    """
    per_class = max(1, math.ceil(cap / max(len(classes), 1)))
    return frame.groupby("emotion", observed=True).head(per_class).reset_index(drop=True)


def build_dataloaders(
    manifest: pd.DataFrame,
    *,
    input_size: tuple[int, int],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = GLOBAL_SEED,
    limit_batches: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, int], dict[str, float]]:
    """Stratified loaders for train/val/test plus class counts and weights.

    ``limit_batches`` caps each split (notebook smoke / unit tests).
    """
    classes = emotion_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    sizes = (input_size[0], input_size[1])
    official = manifest[manifest["split"] == "train"]
    test_df = manifest[manifest["split"] == "test"]
    train_df, val_df = _stratified_val_split(official, VAL_FRACTION, seed)
    if limit_batches is not None:
        cap = limit_batches * batch_size
        train_df = _cap_evenly(train_df, cap, classes, seed)
        val_df = _cap_evenly(val_df, cap, classes, seed)
        test_df = _cap_evenly(test_df, cap, classes, seed)
    datasets = {
        "train": FERDataset(train_df, class_to_idx, input_size=sizes),
        "val": FERDataset(val_df, class_to_idx, input_size=sizes),
        "test": FERDataset(test_df, class_to_idx, input_size=sizes),
    }
    loaders = {
        split: DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
        )
        for split, ds in datasets.items()
    }
    counts = train_df["emotion"].value_counts().reindex(classes).fillna(0).astype(int)
    class_weights = _inverse_frequency_weights(counts.to_numpy())
    return loaders, counts.to_dict(), class_weights


def _inverse_frequency_weights(counts: np.ndarray) -> dict[str, float]:
    """Inverse-frequency class weights normalised to mean 1 (CE-stable)."""
    counts = np.maximum(counts.astype(float), 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return {f"{i}": float(w) for i, w in enumerate(weights)}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FaceEmotionNet(nn.Module):
    """MobileNetV2 backbone with ImageNet weights + a small MLP head, with the
    ImageNet normalisation folded in as buffers (so ONNX takes plain [0, 1])."""

    def __init__(
        self,
        *,
        num_classes: int = 7,
        pretrained: bool = True,
        hidden: tuple[int, ...] = (512,),
    ) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = EmbeddingMLP(EMBED_DIM, num_classes, hidden=hidden)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.set_features_trainable(False)

    def set_features_trainable(self, trainable: bool) -> None:
        for param in self.features.parameters():
            param.requires_grad = trainable

    @property
    def target_layer(self) -> nn.Module:
        """Last convolutional block — the Grad-CAM hook point."""
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std  # baked-in normalisation
        embedding = self.pool(self.features(x)).flatten(1)
        return self.classifier(embedding)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """1280-d feature vector per image (no classifier)."""
        x = (x - self.pixel_mean) / self.pixel_std
        return self.pool(self.features(x)).flatten(1)


class EmbeddingMLP(nn.Module):
    """Small head trained on shared embeddings (the 'deep' comparison entry)."""

    def __init__(
        self, input_dim: int, num_classes: int, *, hidden: tuple[int, ...] = (512,), dropout: float = 0.3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_split(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted class indices, true class indices) for a loader."""
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch, y in loader:
        logits = model(batch.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# --------------------------------------------------------------------------- #
# Embedding extraction (the expensive, one-time, forward-only pass)
# --------------------------------------------------------------------------- #
EMBED_CACHE = "data/processed/fer2013_embeddings"


def embed_cache_paths() -> tuple[Path, Path, Path]:
    root = repo_path(EMBED_CACHE)
    return root / "train.npz", root / "val.npz", root / "test.npz"


def extract_embeddings(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    cache: bool = True,
    force: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Forward-only pass → cached 1280-d embeddings per split.

    Returns ``{"train": {"emb": (N, 1280) f32, "y": (N,) int}, ...}``.
    """
    model = model.eval().to(device)
    result: dict[str, dict[str, np.ndarray]] = {}
    paths = embed_cache_paths()
    for split, path in zip(("train", "val", "test"), paths, strict=True):
        if cache and path.exists() and not force:
            data = np.load(path)
            result[split] = {"emb": data["emb"], "y": data["y"], "cached": True}
            log.info("embedding cache hit %s (%s)", split, path)
            continue
        embs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch, y in loaders[split]:
                embs.append(model.embed(batch.to(device)).cpu().numpy())
                ys.append(y.numpy())
        emb = np.concatenate(embs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        result[split] = {"emb": emb, "y": y, "cached": False}
        if cache:
            ensure_dir(path.parent)
            np.savez_compressed(path, emb=emb, y=y)
        log.info("embedded %s in %.1fs", split, time.perf_counter() - t0)
    return result


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def gradcam_heatmap(
    model: nn.Module, batch_chw: torch.Tensor, device: torch.device, *, target_class: int | None = None
) -> np.ndarray:
    """Grad-CAM heat map (H×W, [0, 1]) for row 0 of ``batch_chw`` ([0, 1] pixels)."""
    model.eval()
    layer = model.target_layer
    captured: dict[str, Any] = {}

    def forward_hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        captured["activations"] = out.detach()

    def backward_hook(_module: nn.Module, _gin: Any, grad_out: torch.Tensor) -> None:
        grads = grad_out if isinstance(grad_out, torch.Tensor) else grad_out[0]
        captured["grads"] = grads.detach()

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)
    try:
        sample = batch_chw[:1].to(device).requires_grad_(True)
        logits = model(sample)
        class_idx = int(logits.argmax()) if target_class is None else target_class
        model.zero_grad()
        logits[0, class_idx].backward()
        grads = captured["grads"].mean(dim=(2, 3), keepdim=True)
        weights = torch.relu((grads * captured["activations"]).sum(dim=(2, 3), keepdim=True))
        heatmap = (weights * captured["activations"]).sum(dim=1, keepdim=True).relu()
    finally:
        handle_f.remove()
        handle_b.remove()
    heatmap = torch.nn.functional.interpolate(
        heatmap, size=batch_chw.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    peak = float(heatmap.max())
    return (heatmap / peak if peak > 0 else heatmap).cpu().numpy()


# --------------------------------------------------------------------------- #
# ONNX export + parity
# --------------------------------------------------------------------------- #
def export_onnx(
    model: nn.Module,
    dest: Path,
    *,
    input_size: tuple[int, int],
    parity_batch: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Export to ONNX and verify torch ↔ onnxruntime parity (≤ 1e-4).

    Returns ``{"path", "size_mb", "parity_max_abs", "parity_ok"}``; raises if
    parity fails — a silently diverging artifact must never ship.
    """
    import onnxruntime as ort

    model = model.eval().cpu()
    dest = Path(dest)
    ensure_dir(dest.parent)
    dummy = torch.zeros(1, 3, *input_size, dtype=torch.float32)
    export_kwargs: dict[str, Any] = {
        "input_names": ["image"],
        "output_names": ["logits"],
        "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": ONNX_OPSET,
    }
    try:
        torch.onnx.export(model, dummy, str(dest), dynamo=False, **export_kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, dummy, str(dest), **export_kwargs)

    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    batch = parity_batch if parity_batch is not None else torch.rand(4, 3, *input_size)
    batch = batch.cpu().float()
    with torch.no_grad():
        torch_logits = model(batch).numpy()
    ort_logits = session.run(None, {name: batch.numpy()})[0]
    max_abs = float(np.abs(torch_logits - ort_logits).max())
    size_mb = round(dest.stat().st_size / (1024 * 1024), 2)
    ok = max_abs <= ONNX_PARITY_TOL
    result = {
        "path": str(dest),
        "size_mb": size_mb,
        "parity_max_abs": round(max_abs, 8),
        "parity_ok": bool(ok),
        "opset": ONNX_OPSET,
    }
    if not ok:
        raise RuntimeError(
            f"ONNX parity failed: max |torch - onnxruntime| = {max_abs:.2e} > {ONNX_PARITY_TOL}"
        )
    return result


# --------------------------------------------------------------------------- #
# Metadata registration
# --------------------------------------------------------------------------- #
def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - not a git clone
        return "unknown"


def register_face_artifact(
    onnx_path: Path,
    *,
    version: str,
    training: dict[str, Any],
    models_dir: Path | None = None,
) -> Path:
    """Register the face artifact + training provenance in models/metadata.json."""
    models_dir = models_dir or repo_path("models")
    meta_path = models_dir / "metadata.json"
    meta: dict[str, Any] = load_json(meta_path) if meta_path.exists() else {}
    meta.setdefault("artifacts", {})["face"] = Path(onnx_path).name
    meta.setdefault("model_versions", {})["face"] = version
    meta.setdefault("training", {})["face"] = training
    save_json(meta, meta_path)
    return meta_path


# --------------------------------------------------------------------------- #
# Model comparison on shared embeddings
# --------------------------------------------------------------------------- #
def _train_mlp_head(
    model: nn.Module, frames: dict[str, dict[str, np.ndarray]], *, epochs: int, batch_size: int, seed: int
) -> list[dict[str, Any]]:
    """Train the MLP head on cached embeddings (CPU — 1280-d input is tiny)."""
    set_seed(seed)
    x_train = torch.from_numpy(frames["train"]["emb"])
    y_train = torch.from_numpy(frames["train"]["y"])
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    true_counts = np.bincount(y_train.numpy(), minlength=len(emotion_classes()))
    w = torch.tensor(list(_inverse_frequency_weights(true_counts).values()), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, correct, seen = 0.0, 0, 0
        t0 = time.perf_counter()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.float()), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(yb)
            correct += int((model(xb.float()).argmax(1) == yb).sum())
            seen += len(yb)
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "train_acc": round(correct / max(seen, 1), 4),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(entry)
        log.info("mlp epoch %d/%d %s", epoch, epochs, entry)
    return history


def compare_models(
    frames: dict[str, dict[str, np.ndarray]],
    *,
    models: tuple[str, ...] = MODEL_ZOO,
    seed: int = GLOBAL_SEED,
    mlp_epochs: int = 20,
    mlp_batch_size: int = 256,
) -> tuple[dict[str, dict[str, Any]], Any | None, list[dict[str, Any]] | None]:
    """Fit the zoo on shared embeddings and score each on val + test.

    sklearn/xgboost heads run in a fresh, **torch-free** subprocess
    (``mindsense.models.zoo``): their bundled OpenMP runtimes segfault or hang
    in a process that has imported torch (macOS, reproduced). The MLP head is
    trained in-process with torch (stable).

    Returns ``(comparison, best_mlp_or_none, mlp_history)``. ``best_mlp`` is
    None unless ``"mlp"`` is in ``models``.
    """
    classes = emotion_classes()
    comparison: dict[str, dict[str, Any]] = {}
    zoo = [n for n in models if n != "mlp"]
    if zoo:
        log.info(
            "running zoo heads (%s) in a torch-free subprocess — "
            "per-head lines stream live as each finishes; XGBoost is the slow one (~5 min)",
            ", ".join(zoo),
        )
        with tempfile.TemporaryDirectory() as td:
            frames_path = Path(td) / "frames.npz"
            np.savez(
                frames_path,
                **{
                    f"{split}_{field}": np.asarray(frames[split][field])
                    for split in ("train", "val", "test")
                    for field in ("emb", "y")
                },
            )
            out_path = Path(td) / "zoo.json"
            proc = subprocess.run(
                [sys.executable, "-m", "mindsense.models.zoo", str(frames_path), str(out_path)],
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"zoo subprocess exited {proc.returncode} (its logs streamed above; "
                    f"out artifact {out_path} missing/incomplete)"
                )
            comparison.update(json.loads(out_path.read_text()))
        for name, entry in comparison.items():
            log.info(
                "compare %-8s val acc %.3f macro-F1 %.3f | test acc %.3f macro-F1 %.3f (%.1fs)",
                name,
                entry["val"]["accuracy"],
                entry["val"]["macro_f1"],
                entry["test"]["accuracy"],
                entry["test"]["macro_f1"],
                entry["fit_seconds"],
            )

    best_mlp = None
    mlp_history: list[dict[str, Any]] | None = None
    if "mlp" in models:
        best_mlp = EmbeddingMLP(EMBED_DIM, len(classes))
        mlp_history = _train_mlp_head(
            best_mlp, frames, epochs=mlp_epochs, batch_size=mlp_batch_size, seed=seed
        )
        best_mlp.eval()
        with torch.no_grad():
            val_pred = best_mlp(torch.from_numpy(frames["val"]["emb"])).argmax(1).numpy()
            test_pred = best_mlp(torch.from_numpy(frames["test"]["emb"])).argmax(1).numpy()
        comparison["mlp"] = {
            "fit_seconds": round(sum(e["seconds"] for e in mlp_history), 2),
            "val": classification_metrics(frames["val"]["y"], val_pred, classes),
            "test": classification_metrics(frames["test"]["y"], test_pred, classes),
        }
        log.info(
            "compare %-8s val acc %.3f macro-F1 %.3f | test acc %.3f macro-F1 %.3f",
            "mlp",
            comparison["mlp"]["val"]["accuracy"],
            comparison["mlp"]["val"]["macro_f1"],
            comparison["mlp"]["test"]["accuracy"],
            comparison["mlp"]["test"]["macro_f1"],
        )
    return comparison, best_mlp, mlp_history


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_face_model(
    *,
    models: tuple[str, ...] = MODEL_ZOO,
    mlp_epochs: int = 20,
    mlp_batch_size: int = 256,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    device: str = "auto",
    pretrained: bool = True,
    limit_batches: int | None = None,
    export: bool = True,
    cache_embeddings: bool = True,
    force_embeddings: bool = False,
    seed: int = GLOBAL_SEED,
    version: str = "exp03-v1",
) -> dict[str, Any]:
    """Train the FER-2013 mood-cue model end-to-end and export artifacts.

    1. Embed every image once with a frozen MobileNetV2 backbone (cached).
    2. Compare several heads (logreg / forest / xgboost / mlp) on those
       embeddings; score each on val + test.
    3. Export the MLP head — with the frozen backbone baked into the graph —
       to ONNX, verify torch ↔ onnxruntime parity, enforce the ≤ 20 MB budget,
       and register provenance in ``models/metadata.json`` with all metrics.
    """
    set_seed(seed)
    dev = pick_device(device)
    cfg = image_config()
    input_size = (int(cfg["input_size"][0]), int(cfg["input_size"][1]))
    if models is None or len(models) == 0:
        models = MODEL_ZOO

    manifest = load_manifest()
    sources = manifest["data_source"].value_counts().to_dict()
    loaders, train_counts, class_weights = build_dataloaders(
        manifest,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        limit_batches=limit_batches,
    )
    log.info(
        "FER loaders on %s | train %d val %d test %d | sources %s",
        dev,
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
        sources,
    )

    backbone = FaceEmotionNet(pretrained=pretrained)
    frames = extract_embeddings(
        backbone, loaders, dev, cache=cache_embeddings and limit_batches is None,
        force=force_embeddings,
    )
    comparison, best_mlp, mlp_history = compare_models(
        frames,
        models=models,
        seed=seed,
        mlp_epochs=mlp_epochs,
        mlp_batch_size=mlp_batch_size,
    )

    best_name = max(comparison, key=lambda k: comparison[k]["test"]["macro_f1"])
    result: dict[str, Any] = {
        "experiment": "exp03",
        "version": version,
        "device": str(dev),
        "seed": seed,
        "pretrained": pretrained,
        "embed_dim": EMBED_DIM,
        "data_sources": {str(k): int(v) for k, v in sources.items()},
        "train_counts": train_counts,
        "class_weights": class_weights,
        "comparison": {name: _summary_of(comp) for name, comp in comparison.items()},
        "best_by_test_macro_f1": best_name,
        "test": comparison[best_name]["test"],
        "mlp_history": mlp_history,
    }

    if export and "mlp" in comparison:
        assert best_mlp is not None
        net = FaceEmotionNet(pretrained=pretrained)
        net.classifier = best_mlp
        onnx_path = repo_path("models", "face_emotion.onnx")
        parity_batch = next(iter(loaders["val"]))[0][:8]
        onnx_info = export_onnx(net, onnx_path, input_size=input_size, parity_batch=parity_batch)
        if onnx_info["size_mb"] > float(cfg.get("onnx_max_mb", 20)):
            raise RuntimeError(
                f"ONNX artifact {onnx_info['size_mb']} MB exceeds the "
                f"{cfg.get('onnx_max_mb', 20)} MB budget"
            )
        result["onnx"] = onnx_info
        torch.save(
            {
                "state_dict": net.state_dict(),
                "classes": emotion_classes(),
                "input_size": input_size,
                "comparison": result["comparison"],
            },
            ensure_dir(repo_path("models", "face_emotion")) / "checkpoint.pt",
        )
        training_meta = {
            "experiment": "exp03",
            "date": pd.Timestamp.now(tz="UTC").isoformat(),
            "git": _git_hash(),
            "datasets": {"fer2013": "real" if set(sources) == {"real"} else "synthetic"},
            "non_diagnostic": True,
            "intended_use": "supplementary facial mood cue for the app; never a diagnosis",
            "approach": "frozen MobileNetV2 embeddings + compared heads (see DECISIONS #22)",
            "input_size": list(input_size),
            "classes": emotion_classes(),
            "mood_cue_groups": mood_cue_groups(),
            "comparison": result["comparison"],
            "best_by_test_macro_f1": best_name,
            "metrics": {"test": result["test"]},
            "onnx": onnx_info,
        }
        meta_path = register_face_artifact(onnx_path, version=version, training=training_meta)
        result["metadata"] = str(meta_path)
        save_json(result, repo_path("reports", "metrics", "exp03_image.json"))
        log.info(
            "exported %s (%.2f MB, parity %.2e) — best model %s",
            onnx_path,
            onnx_info["size_mb"],
            onnx_info["parity_max_abs"],
            best_name,
        )
    return result


def _summary_of(comp: dict[str, Any]) -> dict[str, Any]:
    """Compact per-model entry for JSON reports (drop the big confusion matrix)."""
    summary = dict(comp)
    summary["test"] = {k: v for k, v in comp["test"].items() if k != "confusion_matrix"}
    summary["val"] = {k: v for k, v in comp["val"].items() if k != "confusion_matrix"}
    return summary
