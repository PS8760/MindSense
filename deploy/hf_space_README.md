---
title: MindSense
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# MindSense

Early, explainable mental-wellness risk screening — an educational academic
project, **not** a substitute for professional care.

Streamlit web app, Dockerized. Calm light/dark themes, in-browser screening
(PHQ-9 / GAD-7), rule-based "what stands out" feedback, optional Groq-powered
suggestions (falls back offline), community insights, and crisis-helpline
resources. See the repo README for the full docs.

## Deploy to this Space (one-time checklist)

1. Create a Space: **New Space → SDK: Docker** (free CPU is enough).
2. Add these files from the repo (everything the image needs):
   ```bash
   cd /path/to/Honors_Mini_Project
   HF=~/space-mindsense   # your Space clone
   # already includes a Dockerfile; copy the others over:
   cp -r .streamlit app src config models reports $HF/
   cp Dockerfile pyproject.toml requirements.txt $HF/
   cp README.md $HF/README.md
   ```
   **Never** copy `.env`, `data/`, `.venv`, or `notebooks/`.
3. Open the Space → **Settings → Variables and secrets** and add (optional):
   - `GROQ_API_KEY` — enables kinder, Groq-drafted suggestions; without it the
     app silently uses offline suggestions.
   - `APP_ENV=production`, `ENABLE_FACE=0` (only if you want to disable the
     facial mood-cue; the ONNX is already committed under `models/`).
4. Push from the Space repo:
   ```bash
   cd $HF
   git add .
   git commit -m "Deploy MindSense"
   git push
   ```
5. HF rebuilds the Space; the app serves on port 7860 (the Dockerfile's
   default `--build-arg PORT=7860`). Health check: `GET /_stcore/health`.

## Ports & LFS notes

- The container listens on **7860** by default (`ARG PORT`); map other targets
  with `-e PORT=8501` + `-p 8501:8501`. The Compose file maps `8501 -> 7860`.
- No tracked file exceeds 10 MB, so Git LFS is **not** needed today. If you
  later add larger model files, run:
  ```bash
  git lfs track 'models/**/*.onnx'
  ```

## Repo → Space sync

Keep the Space as a mirror of the main branch. Recommend a GitHub Action that
uses `huggingface/hub` or `git push hf main` after CI passes (see
`.github/workflows/ci.yml`).