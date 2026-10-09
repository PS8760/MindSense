# MindSense

Early, explainable mental-wellness risk screening. Streamlit web app over an
inference package (`src/mindsense/inference`), built around honest,
non-diagnostic boundaries: PHQ-9 / GAD-7 scoring, rule-based "what stands out"
feedback, optional Groq-drafted suggestions (falls back offline), community
insights, and crisis-helpline resources.

> **Educational academic project — not a substitute for professional care.**

## Quickstart

```bash
make setup     # venv + lean runtime deps (no torch)
make data      # optional: downloadable datasets -> data/processed (licence permitting)
make train     # optional: train/export model artifacts (needs training deps)
make app       # streamlit run app/Home.py  -> http://localhost:8501
```

Fast checks:

```bash
make lint      # ruff check (src app tests scripts)
make test      # pytest (141 tests, incl. AppTest smoke of all 9 pages)
python scripts/smoke_app.py
```

## Feature flags (`.env.example` → `.env`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `ENABLE_FACE` | `1` | Facial mood-cue module (needs `models/face_emotion.onnx`); set `0` to disable |
| `ENABLE_TRANSFORMER` | `0` | DistilBERT text path for comparisons (heavy) |
| `DEFAULT_REGION` | `India` | Default helpline region in the sidebar |
| `APP_ENV` | `local` | `local \| ci \| production` |
| `FORCE_SYNTHETIC_BANNER` | `0` | Louder synthetic-data banners in CI |
| `GROQ_API_KEY` | — | Enables Groq AI: check-in suggestions **and** the risk/intervention/severity/text features when no local artifact is present (offline fallback otherwise) |
| `MINDSENSE_DISABLE_GROQ` | `0` | `1` forces local/offline-only inference (set suite-wide by the tests) |

## Model experiments (`make train`)

`make train` runs every implemented experiment; each one compares several
model families, ranks them on validation, and scores the winner on the held-out
test split. Winners are summarised in `reports/metrics/leaderboard.json`.

| Exp | Task | Models compared | Best on test |
| --- | --- | --- | --- |
| 03 image | FER-2013 mood-cue classification | logistic / random-forest / XGBoost / MLP | MLP (macro-F1 0.516) → `models/face_emotion.onnx` |
| 04 tabular | risk screen (student + professional) | logistic / RF / gradient-boosting | gradient-boosting (macro-F1 0.843) |
| 06 text | MH-corpus screening (7 classes) | Naive-Bayes / logistic / LinearSVC | linear-SVC (macro-F1 0.684) |
| 07 prognosis | DASS-42 subscale regression | linear / RF / gradient-boosting | gradient-boosting (RMSE 10.50) |

The **Model comparison** tab on page 6 (`app/components/model_info.py`) renders
these rankings in the app — accuracy / precision / recall / F1 per model, the
🏆 best pick, and a clear note showing **which model answers today** (the shipped
artifact, or the Groq backend when none is present).

`make train --quick` (CI) runs all four on tiny samples; `python -m
mindsense.models.train_all --help` lists flags. The image zoo fits its
sklearn/XGBoost heads in a **torch-free subprocess** (`mindsense.models.zoo`)
so torch + XGBoost never share a process (macOS OpenMP/liblomp crash — see
`docs/DECISIONS.md`).

Only `models/face_emotion.onnx` + `models/metadata.json` are committed, so a
fresh clone/deploy has no risk/intervention/severity/text artifacts. Those
features then run on the **Groq AI backend** (set `GROQ_API_KEY`) and report
`source="groq"`; once `make train` produces local artifacts they take over
(local-first). `mindsense.groq` is the shared, streamlit/torch-free client.

The same key also powers **Calmer** (page 8): a calm-companion chat that shows a
short "thinking" line before its reply, plus a mood-lifting content section. With
no key, Calmer and the other AI features fall back to hand-written offline copy
and never break.

## Theme

Calm wellbeing palette (soft teal-green + lavender). One sidebar button toggles
**Light / Dark**; defaults to **System** (follows your device). Theming is
purely CSS-variable driven in `app/components/theme.py` — no per-mode
re-renders of the app logic.

## Deployment

Lean image (only `requirements.txt`), non-root user, health-checked.

- **Docker (local):**
  ```bash
  docker build -t mindsense .
  docker run --rm -p 8501:7860 -e PORT=7860 -e GROQ_API_KEY="$GROQ_API_KEY" mindsense
  # open http://localhost:8501  (health: GET /_stcore/health)
  ```
  Container port defaults to **7860** (Hugging Face convention). Build with
  `--build-arg PORT=8501` if you prefer native 8501, or use compose:
  `docker compose up --build` (maps `8501 -> 7860`, reads optional `.env`).
- **Hugging Face Spaces (primary):** create a Docker-sdk Space, copy
  `Dockerfile .streamlit/ app/ src/ config/ models/ reports/ requirements.txt
  pyproject.toml`, set `app_port: 7860`, add `GROQ_API_KEY` as a Space secret.
  Full step-by-step: [`deploy/hf_space_README.md`](deploy/hf_space_README.md).
- **Render / Railway:** point the service at this repo's `Dockerfile`; set env
  `PORT`, `GROQ_API_KEY`, feature flags.
- **Streamlit Community Cloud:** connect the repo, main file `app/Home.py`,
  use the lean `requirements.txt`, add secrets in the dashboard.

`data/` is **not** shipped (licences forbid redistributing sourced datasets);
pages show graceful "not available" states until you run `make data`.

## CI

`.github/workflows/ci.yml` runs on push/PR: ruff, pytest, an app smoke test, a
synthetic-data notebook smoke test, and a **Docker build + health check**
(`/_stcore/health` must return 200) with the reported image size.

## Layout

```
app/            Streamlit pages (Home + 1–8) and UI components
src/mindsense/  inference API, screening/scoring, models, utils (the package)
config/         global + helplines configuration
models/         exported artifacts + metadata.json (mood-cue ONNX)
reports/        metrics and figures (model cards, subgroup metrics)
scripts/        data download, pipeline, notebook executors, smoke tests
tests/          pytest suite (tests renamed around files, glob-based)
deploy/         Hugging Face Space README template
```

See `docs/` for model cards, data sources, design decisions, and the
presentation material.