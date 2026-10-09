# PPT Generation Prompt — MindSense

> Paste the block below verbatim into the AI agent that builds the deck. It
> should produce a **PowerPoint (.pptx)** via python-pptx. Keep the deck
> minimal, clean and teacher-friendly — one idea per slide, nothing dense.

---

**Role**: You are a slide-design expert and academic presenter. Create a
PowerPoint for **"MindSense — early, explainable mental-wellness risk
screening"**, an educational engineering project. I am presenting it to an
instructor for evaluation, so clarity and honesty matter more than length.

**Ground truth — read these files first** (project root):
- `README.md` (features, quickstart, experiment table)
- `docs/DECISIONS.md` (why decisions were made, incl. ethics & crash fix)
- `reports/metrics/leaderboard.json` and `reports/metrics/exp03* / exp04* / exp06* / exp07* .json` (do not invent numbers — use only these)
- `prompt.md` (original spec) only to understand requirements, do not copy verbatim

**Project facts (verified)**
- Streamlit web app (7 pages) + Python package (`src/mindsense`): PHQ-9/GAD-7/DASS scoring, rule-based "what stands out" feedback, optional Groq AI suggestions (offline fallback), community insights, crisis helplines.
- **Not a diagnostic tool** — is an educational project; this disclaimer must appear on the title slide.
- Datasets (10, real + synthetic fallback): student depression, sentiment-mh, DASS-42, OSMI, Dreaddit, Drugs.com, FER-2013, tabular risk (29,152 rows).
- Model experiments (all compare several model families and rank them):
  - Exp 03 image (FER-2013 mood cue): logreg / random-forest / XGBoost / MLP → **MLP wins, test macro-F1 0.516**, exported to `models/face_emotion.onnx` (10.97 MB).
  - Exp 04 tabular risk screen: logreg / RF / gradient-boosting → **gradient-boosting, macro-F1 0.840**.
  - Exp 06 text screening (7 classes): Naive-Bayes / logreg / LinearSVC → **LinearSVC, macro-F1 0.686**.
  - Exp 07 DASS-42 prognosis (3 output scores): linear / RF / gradient-boosting → **gradient-boosting, RMSE 10.47**.
- Engineering/deployment: Docker (non-root, port 7860, healthcheck), Hugging Face Spaces / Streamlit Cloud ready, CI (lint, 122 pytest tests, notebook smoke, docker-image build), GitHub Actions.
- Notable engineering story: fixed a macOS crash where torch+XGBoost conflicted over the OpenMP runtime by running sklearn/XGBoost heads in a torch-free subprocess.

**Slide plan (10–12 slides, use exactly this structure)**
1. Title — short tagline + the "not a diagnostic tool" disclaimer.
2. Problem & goal — why early mental-wellness screening matters; the one-line objective.
3. Approach at a glance — 4 pillars: data → features → models → app; a simple 4-box diagram.
4. Data — the 10 datasets, cleaned/harmonized 29K+ rows; 2–3 bullet points max.
5. App — 7 user-facing pages; screenshot-style mock or simple page flow diagram.
6. Models 1 — comparison table: experiment, models compared, winner metric (from leaderboard; only the real numbers above).
7. Models 2 — the image model detail: MobileNetV2 embeddings → heads, best MLP exported to ONNX.
8. Engineering story — the macOS OpenMP crash + subprocess isolation fix (keep it to 3 lines, it shows rigor).
9. Quality & tests — 122 tests pass, ruff clean, notebook smoke, CI docker build.
10. Deployment — Docker image, HF Spaces / Streamlit Cloud, graceful offline fallback.
11. Ethics & limits — non-diagnostic, synthetic fallback, bias caveats (FER-2013, imbalanced corpora), no personal data stored.
12. Summary / next steps — 3 bullets + a "Thank you / Questions?" close.

**Design rules (required)**
- **Color theme: minimal and clean.** White background, one accent color only (soft teal `#2E8B7A` or calm blue `#2E6E9E` — pick one), dark gray text (`#33333A`), light-gray section tints (`#F4F6F5`). No gradients, no clip-art, no emojis.
- One typeface family throughout (e.g., Calibri or Segoe UI); titles 28–32 pt, body 16–18 pt.
- Maximum **3 bullets per content slide**, 6 words per bullet; put everything else in a table or a small diagram.
- Every slide needs: a short title phrased as an insight (e.g., "Compare, don't guess") not a label.
- Tables must be wide, 12–14 pt, with header row in the accent color; round all numbers to the values already given.
- Use simple shapes (rounded rectangles + arrows) for any diagram: 4 boxes for approach, a linear flow for the pipeline, a 4-column card strip for experiments.
- No dense code on slides — at most one tiny snippet if truly needed (prefer none).
- Speaker-note each slide in 2–3 sentences so I know what to say.

**Output**: generate `MindSense_Deck.pptx` in the project root using
python-pptx (assume it can be installed). After generating, verify the file
opens and report the slide count + which accent color you used.