# MindSense — Honors Mini Project PPT Content

> Source of truth for the deck. Real numbers only, taken from
> `reports/metrics/*.json` and the runnable app. Structure follows the required
> order: Name → Intro/Motivation → Literature Review → Research Gaps →
> Objectives → Problem Definition → Methodology → Performance Evaluation →
> Conclusion. Present MindSense as **one project**, not as a list of experiments.

---

## SLIDE 1 — Name (Title)

**MindSense**
*Early, explainable, privacy-respecting mental-wellness risk screening*

- Honors Lab — AI & ML in Healthcare, Mini Project
- Team / Name: **[Your Name]** · Guide: **[Guide Name]**
- Stack: Python 3.11 · scikit-learn · XGBoost · Streamlit · ONNX · Docker
- Tagline: *Screening, not diagnosis.*
- Disclaimer (must appear): **Educational project — not a diagnostic tool.**

---

## SLIDE 2 — Introduction & Motivation

**Why this project**
- Mental-health conditions (stress, anxiety, depression) affect every age and gender; early screening is routinely missed due to **stigma, cost and access**.
- Existing tools are either **single-modality** (a questionnaire only) or **black-box** (no explanation, no trust).
- Free, self-reportable signals already exist: lifestyle data, PHQ-9/GAD-7/DASS questionnaires, journal text and facial affect cues.

**Our motivation:** make early screening *accessible, explainable and safe* — a tool that returns **risk level + reasons + guidance**, never a diagnosis.

| Motivation driver | What MindSense does |
|---|---|
| Access | Free, web-based, mobile-responsive, no login |
| Trust | Plain-language "what stands out" explanations |
| Safety | Crisis banner + helplines + non-dismissable disclaimer |
| Privacy | In-memory only — nothing stored or logged |

---

## SLIDE 3 — Literature Review (Table of Papers)

| # | Paper / Source | Method / Contribution | Relevance to MindSense |
|---|---|---|---|
| 1 | Kroenke, Spitzer & Williams (2001), *J Gen Intern Med* | PHQ-9 — 9-item depression screener, validated cut-offs | Check-In questionnaire bands |
| 2 | Spitzer et al. (2006), *Arch Intern Med* | GAD-7 — 7-item anxiety screener | Anxiety screening, standard bands |
| 3 | Lovibond & Lovibond (1995), *Manual of the DASS* | DASS-42: 42 items → 3 subscales + 5 severity bands | Prognosis / severity stratification |
| 4 | De Choudhury et al. (2013), *ICWSM* | Predicting depression from social-media behaviour | Text screening signal |
| 5 | Coppersmith et al. (2018), *JMIR* | Quantifying clinical signals from social media | Justifies NLP screening of free text |
| 6 | Turcan & McKeown (2019), *LOUHI* — Dreaddit | Labeled Reddit stress corpus | External / out-of-domain test set |
| 7 | Goodfellow et al. (2013), *IEEE FG* — FER-2013 | 35.9K facial expressions, 7 classes | Mood-cue dataset |
| 8 | Barrett et al. (2019), *Psych Science in the Public Interest* | Critical review of facial-emotion inference | Justifies "mood cue, not diagnosis" |
| 9 | Chen & Guestrin (2016), *KDD* — XGBoost | Scalable gradient boosting, native NaN handling | Risk + prognosis models |
| 10 | Selvaraju et al. (2017), *ICCV* — Grad-CAM | Visual explanations for CNNs | Mood-cue explanation |
| 11 | Ribeiro et al. (2016), *KDD* — LIME | Local interpretable explanations | Text + tabular explanation |
| 12 | Lundberg & Lee (2017), *NeurIPS* — SHAP | Unified feature-importance theory | Explainability layer |
| 13 | Hardt et al. (2016), *NeurIPS* | Equal opportunity / fairness metrics | Fairness audit |
| 14 | OSMI (2014–2021) Mental Health in Tech Surveys | Workplace MH data, `treatment` proxy target | Professional population |

*(Trim to 8–10 rows on the slide; keep the citations real.)*

---

## SLIDE 4 — Research Gaps (2 to 3)

**Gap 1 — No multimodal, same-subject public data**
No public dataset records tabular risk factors, journal text and facial images *for the same people*; published multimodal work is single-source or private.
→ *MindSense builds independent per-modality models with a transparent, rule-based late-fusion summary (never a single fused score).*

**Gap 2 — Explainability and fairness rarely reported together for mental-health screening**
Most screening models report only aggregate accuracy; subgroup (gender / age / population) behaviour and human-readable reasons are usually absent — unacceptable for a high-stakes domain.
→ *MindSense ships SHAP / LIME / Grad-CAM per modality plus a fairness audit (TPR, FPR, demographic-parity and equal-opportunity gaps).*

**Gap 3 — Circular targets and leakage in existing pipelines**
Predicting a DASS severity band from the very items that compute the band is circular, and popular sentiment corpora reuse Reddit posts that also appear in "external" test sets.
→ *Depression items are excluded from the prognosis features; 2,819 verbatim cross-corpus duplicates were detected and quarantined (`text_leakage_report.json`).*

---

## SLIDE 5 — Objectives

1. **Build a privacy-first, non-diagnostic wellness screening tool** that returns risk tier + reasons + guidance.
2. **Harmonize multiple public datasets** into one documented, reproducible pipeline with leakage guards.
3. **Compare several model families per capability** and select the best objectively on a held-out split.
4. **Deliver a deployable app** with an optional AI assistant for suggestions and Calmer, plus graceful offline fallbacks.
5. **Assure quality, explainability and ethics** through tests, transparent metrics and clear limitations.

---

## SLIDE 6 — Problem Definition

**Two paragraphs + one closing line.**

*Paragraph 1.* Stress, anxiety and depression are widespread and often first appear in student years, yet most people are never screened early because of stigma, cost and limited access to professionals. Self-reportable signals — lifestyle habits, short validated questionnaires, free-text journaling and facial affect — already exist and are cheap to collect, but they are rarely turned into a single, understandable risk picture.

*Paragraph 2.* Existing options sit at two extremes: clinical tools that need a professional and are not accessible day to day, or consumer apps that give a score without validation, explanation or safety handling. For a sensitive domain this is not enough — what is needed is a **private, explainable, non-diagnostic** screener that is easy to run, states its own limits, and routes anyone at risk to real help.

*Closing line:* **"The goal is to build an early, explainable and private wellness screening aid — not to diagnose."**

---

## SLIDE 7 — Methodology

**Architecture — layered / modular (NOT a straight pipeline)**

```
┌─ Presentation layer ──────────────────────────────────────────────┐
│  Streamlit app — 9 pages (Home + 1–8), light/dark theme           │
├─ Inference layer ─────────────────────────────────────────────────┤
│  mindsense.inference API · local artifacts first · AI fallback    │
├─ Modeling layer ──────────────────────────────────────────────────┤
│  Mood cue · Risk screening · Text screening · Prognosis           │
│  (each compares several families; best exported / selected)       │
├─ Data layer ──────────────────────────────────────────────────────┤
│  10 public sources → clean → harmonize → 29,152-row risk table    │
│  stratified 70/15/15 split (seed 42) · leakage guards             │
└───────────────────────────────────────────────────────────────────┘
```

| Stage | Technique |
|---|---|
| Data | Fallback ladder (Kaggle → manual → mirror → tagged synthetic), dedup, validity screen, stratified 70/15/15 split stored as a column |
| Harmonization | Union schema, `population` column, structural NaN kept (never imputed across populations) |
| Risk screening | Logistic baseline → RF → gradient boosting (native NaN), `RandomizedSearchCV`, 5-fold stratified, recall-favouring threshold |
| Prognosis | DASS-42 subscale regression with depression items excluded; mutual-information item selection |
| Text screening | TF-IDF + LinearSVC/logistic (compared vs Naive Bayes); class weights; macro-F1 |
| Mood cue | FER-2013 transfer learning, class weights → exported to ONNX |
| Safety | Crisis trigger (PHQ-9 item 9, suicidal-class probability, high-precision lexicon) + helplines |
| Deploy | Streamlit + `inference.py` API, Docker (non-root, healthcheck), HF Spaces / Streamlit Cloud |

**The 9 app pages:** `1 Check-In` · `2 Talk it Out` · `3 Mood Patterns` · `4 Mood Selfie` · `5 Community Insights` · `6 Models & Data` · `7 Care & Safety` · `8 Calmer`, under `Home`.

**AI usage (state precisely):** a third-party AI assistant is used **only** for (1) gentle check-in suggestions and (2) the Calmer chatbot. All predictive models run **locally** from trained artifacts; every AI feature has an offline fallback. Do not claim the AI powers the models.

---

## SLIDE 8 — Performance Evaluation (tables / graphs)

**Every model family tried, per capability.** `✔` = selected winner (scored on the held-out test split). Mood-cue figures are test-split; risk/text/prognosis figures are the validation ranking used to select the winner, with the winner's test score noted.

**A — Mood cue (face image, FER-2013), test split**

| Model | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| Logistic regression | 0.494 | 0.477 | 0.449 | 0.457 |
| Random forest | 0.475 | 0.541 | 0.442 | 0.462 |
| XGBoost | 0.528 | 0.592 | 0.471 | 0.493 |
| **MLP ✔** | **0.541** | 0.505 | 0.536 | **0.516** |

**B — Risk screening (tabular)**

| Model | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| **Gradient boosting ✔** | **0.848** | 0.846 | 0.840 | **0.843** |
| Logistic regression | 0.838 | 0.836 | 0.830 | 0.832 |
| Random forest | 0.831 | 0.827 | 0.823 | 0.825 |

*Winner on held-out test: accuracy 0.845, macro-F1 0.840.*

**C — Text screening (7 classes)**

| Model | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| **LinearSVC ✔** | **0.744** | 0.760 | 0.647 | **0.684** |
| Logistic regression | 0.748 | 0.801 | 0.612 | 0.663 |
| Naive Bayes | 0.590 | 0.714 | 0.327 | 0.336 |

*Winner on held-out test: accuracy 0.750, macro-F1 0.686.*

**D — Prognosis (DASS-42 subscale regression)**

| Model | RMSE | R² |
|---|---|---|
| **Gradient boosting ✔** | **10.503** | **0.070** |
| Linear regression | 10.548 | 0.062 |
| Random forest | 10.873 | 0.004 |

*Winner on held-out test: RMSE 10.47. Accuracy does not apply to regression.*

**Charts to render**
1. Grouped bar — **macro-F1 per model** across the three classification capabilities (mood cue, risk screening, text screening); best bar in the accent colour.
2. Bar — **RMSE per model** for prognosis (lower is better).
3. Confusion-matrix heatmaps — mood cue (7-class), text (7-class).

---

## SLIDE 9 — Conclusion (One paragraph)

MindSense is an end-to-end, explainable and privacy-respecting wellness screening aid: it harmonizes 10 public sources into a 29,152-row dataset, compares several model families for each capability and objectively ships the winner — mood-cue **MLP (accuracy 0.541, macro-F1 0.516)**, risk screening **gradient boosting (accuracy 0.848, macro-F1 0.843)**, text screening **LinearSVC (accuracy 0.744, macro-F1 0.684)** and prognosis **gradient boosting (RMSE 10.50)** — behind a 9-page Streamlit app whose only AI use is optional check-in suggestions and the Calmer chatbot, with local models and offline fallbacks. Its honest limits are self-reported, convenience-sampled, English-only, cross-sectional data with modest out-of-domain performance; the clear next step is longitudinal and multilingual validation with a clinician in the loop.

*(Then a clean **Thank you / Questions?** slide.)*

---

## Build status (accurate as of this commit)

- App: **9 pages** (Home + 8); **142 tests** pass, ruff clean, notebook smoke + Docker-build CI.
- Trained and verified: **mood cue (image)**, **risk screening (tabular)**, **text screening**, **prognosis** — metrics above are real.
- Shipped artifact: only `models/face_emotion.onnx` (+ `models/metadata.json`); the other capabilities run from local artifacts once trained, or via the AI assistant when no artifact is present, always with offline fallback.
