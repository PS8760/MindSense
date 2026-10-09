# PPT Generation Prompt — MindSense

> Paste the block below verbatim into the AI agent that builds the deck. It
> should produce a **PowerPoint (.pptx)** via python-pptx. Keep the deck
> minimal, clean, professional and easy to present — one idea per slide.

---

**Role**: You are a slide-design expert and academic presenter. Create a
PowerPoint for **"MindSense — early, explainable mental-wellness risk
screening"**, an educational engineering project. I am presenting it to an
instructor for evaluation, so clarity, structure and honesty matter more than
length. Use clear, simple, professional English throughout.

**Ground truth — read these files first** (project root):
- `README.md` (features, quickstart, experiment table)
- `docs/DECISIONS.md` (why decisions were made, incl. ethics & the crash fix)
- `reports/metrics/leaderboard.json` and `reports/metrics/exp03* / exp04* / exp06* / exp07* .json` (do not invent numbers — use only these)
- `prompt.md` (original spec) only to understand requirements; do not copy verbatim

**Project facts (verified — use these exact numbers)**
- Streamlit web app (**9 pages**: Home + 8) plus a Python package (`src/mindsense`): PHQ-9/GAD-7/DASS scoring, rule-based "what stands out" feedback, community insights, crisis helplines, a face mood-cue page, and a calm-companion chatbot (**Calmer**).
- **Not a diagnostic tool** — it is an educational project; this disclaimer must appear on the title slide.
- Datasets: **10 public sources** (student depression, sentiment-mh, DASS-42, OSMI, Dreaddit, Drugs.com, FER-2013, …), harmonized into a **29,152-row** tabular risk dataset for training.
- Every experiment **trains several model families and ranks them** on validation, then scores the winner on a held-out test split:
  - **Exp 03 — image (FER-2013 mood cue):** logistic regression / random forest / XGBoost / MLP → **MLP wins, macro-F1 0.516** (precision 0.505, recall 0.536); exported to `models/face_emotion.onnx`.
  - **Exp 04 tabular risk screen**: logistic / random forest / gradient boosting → **gradient boosting, macro-F1 0.843** (P 0.846, R 0.840).
  - Exp 06 text screening (7 classes): Naive-Bayes / logistic / LinearSVC → **LinearSVC, macro-F1 0.684** (P 0.760, R 0.647).
  - Exp 07 DASS-42 prognosis (3 output scores): linear / RF / gradient-boosting → **gradient-boosting, RMSE 10.50**.
- Engineering/deployment: Docker (non-root, port 7860, healthcheck), Hugging Face Spaces / Streamlit Cloud ready, CI (lint, **142 pytest tests**, notebook smoke, docker-image build), GitHub Actions.
- Notable engineering story: fixed a macOS crash where torch + XGBoost conflicted over the OpenMP runtime by running the sklearn/XGBoost heads in a torch-free subprocess.

**How AI is used (state this precisely — do not overstate it)**
- A third-party **AI assistant** is used **only** in two optional, non-clinical
  places: (1) the gentle **check-in suggestions** and (2) the **Calmer chatbot**.
- **All predictive models run locally** from trained artifacts and do not depend
  on the AI assistant. Every AI feature has a graceful **offline fallback**.
- Do **not** claim the AI/API powers the models or the predictions. Only the two
  places above use it.

**Slide plan (14 slides, use exactly this structure)**
1. **Title** — project name, one short tagline, and the "not a diagnostic tool" disclaimer.
2. **Problem Definition** — exactly **two paragraphs**, followed by **one line** at the end stating the project goal. (Para 1: the burden of untreated, early mental-health struggles and why early, low-stigma screening matters, especially for students. Para 2: existing tools are either clinical and inaccessible or unvalidated apps — there is a need for a private, explainable, non-diagnostic screener that is easy to run. Closing line: "The goal is to build an early, explainable and private wellness screening aid — not to diagnose.")
3. **Objectives** — **5 short objectives**, each one line:
   1. Build a privacy-first, non-diagnostic wellness screening tool.
   2. Harmonize multiple public datasets into one reproducible pipeline.
   3. Compare several model families per task and select the best objectively.
   4. Deliver a deployable app with an optional AI assistant for suggestions and Calmer.
   5. Assure quality, explainability and ethics through tests and transparent metrics.
4. **Methodology — Architecture** — present the system as a clean **layered / modular architecture** (NOT a plain linear pipeline): a **Data layer → Modeling layer → Inference layer → Presentation layer**, four stacked bands with labelled blocks inside each band. Briefly note what each layer does.
5. **Data** — the datasets, cleaning/harmonization, and the 29K+ rows; at most three bullets.
6. **The Application** — the 9 user-facing pages as a simple flow/label grid; call out Calmer explicitly.
7. **Performance Evaluation — Comparison of Every Model** — one wide table listing **every model of every experiment** with Accuracy, Precision, Recall, F1 (classification) and RMSE/R² (regression). Mark the best model in each experiment.
8. **Performance Evaluation — Comparison Charts** — **include the model comparison charts** here (see chart rules below): one grouped bar chart of **macro-F1 per model for each classification experiment (Exp 03, 04, 06)** and one bar chart of **RMSE per model for Exp 07**. Best bar highlighted in the accent colour.
9. **Deployed Model & AI Assistant** — which model is shipped (image MLP → ONNX) and that the AI assistant is used **only** for check-in suggestions and Calmer, with offline fallbacks.
9. **Engineering Rigor & Quality** — 142 tests pass, ruff clean, notebook smoke, CI docker build; the macOS OpenMP subprocess-isolation fix (3 lines max).
10. **Deployment** — Docker image, HF Spaces / Streamlit Cloud, graceful offline behaviour.
11. **Ethics & Limitations** — non-diagnostic, synthetic fallback, bias caveats (FER-2013, imbalanced corpora), no personal data stored.
12. **Conclusion** — exactly **one paragraph** that summarizes what was built, the key result (best models chosen by comparison), and the honest limits/next step.
13. **Thank you / Questions?** — clean closing slide.

**Chart rules (required)**
- Generate charts with matplotlib, save as PNG, and embed them (do not use fake screenshots).
- Use the single accent color for the best model and a light gray (`#C9D3D0`) for the rest; no gradients, no 3-D, no emojis.
- Chart titles should be insights (e.g., "The best model wins by a clear margin"), axis labels plain and short.
- Grouped bar chart for classification macro-F1 (models on the x-axis, one group per experiment); a separate bar chart for regression RMSE (lower is better).
- Round every value to 3 decimals, exactly as given above.

**Design rules (required)**
- **Color theme: minimal and clean.** White background, one accent color only (soft teal `#2E8B7A` or calm blue `#2E6E9E` — pick one), dark gray text (`#33333A`), light-gray section tints (`#F4F6F5`). No gradients, no clip-art, no emojis.
- One typeface family throughout (e.g., Calibri or Segoe UI); titles 28–32 pt, body 16–18 pt.
- Maximum **3 bullets per content slide**, about 6 words per bullet; put the rest in a table or a simple diagram.
- Every slide title phrased as a short insight, not a bare label.
- Tables wide and legible (12–14 pt) with the header row in the accent color.
- Diagrams use simple rounded rectangles + arrows only; the architecture slide is a **layered** stack, not a straight line.
- No dense code on slides — at most one tiny snippet if truly needed (prefer none).
- Add 2–3 sentences of **speaker notes** to every slide.
- Keep the tone formal, factual and professional throughout — no hype, no exaggeration.

**Output**: generate `MindSense_Deck.pptx` in the project root using
python-pptx (assume it can be installed) and any chart images it needs (save
them under `reports/figures/`). After generating, verify the file opens and
report the slide count, the accent color used, and the list of chart images.
