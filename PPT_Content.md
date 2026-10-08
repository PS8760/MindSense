# MindSense — Honors Mini Project PPT Content

> Only real numbers from this repo are used (`reports/metrics/*.json`, `reports/tables/*.csv`).
> Anything not yet trained is marked `[PENDING — fill after make train]` rather than invented.

---

## SLIDE 1 — Title

**MindSense: End-to-End Mental Health Risk Prediction**
*Early, explainable, privacy-respecting mental-wellness risk screening*

- Honors Lab — AI & ML in Healthcare, Mini Project
- Team / Name: **[Your Name]** · Guide: **[Guide Name]**
- Stack: Python 3.11 · scikit-learn · XGBoost · spaCy · Streamlit · Docker
- Tagline: *Screening, not diagnosis.*

---

## SLIDE 2 — Introduction & Motivation

**Why this project**

- Mental-health conditions (stress, anxiety, depression) affect every age and gender; early screening is routinely missed due to **stigma, cost and access**.
- Existing tools are either **single-modality** (a questionnaire only) or **black-box** (no explanation, no trust).
- Free, self-reportable signals already exist: lifestyle data, PHQ-9/GAD-7/DASS questionnaires, journal text, facial affect cues.

**Our motivation:** make early screening *accessible, explainable and safe* — a tool that returns **risk level + reasons + guidance**, never a diagnosis.

| Motivation driver | What MindSense does |
|---|---|
| Access | Free, web-based, mobile-responsive, no login |
| Trust | SHAP/LIME explanations in plain language |
| Safety | Crisis banner, helplines, non-dismissable disclaimer |
| Privacy | In-memory only — nothing is stored or logged |

---

## SLIDE 3 — Literature Review (Table)

| # | Paper / Source | Method / Contribution | Relevance to MindSense |
|---|---|---|---|
| 1 | Kroenke, Spitzer & Williams (2001), *J Gen Intern Med* | PHQ-9 — 9-item depression screener, validated cut-offs | Risk Assessment page questionnaire bands |
| 2 | Spitzer et al. (2006), *Arch Intern Med* | GAD-7 — 7-item anxiety screener | Anxiety screening, standard bands |
| 3 | Lovibond & Lovibond (1995), *Manual of the DASS* | DASS-42: 42 items → 3 subscales + 5 severity bands | Exp 4 prognosis / severity stratification |
| 4 | De Choudhury et al. (2013), *ICWSM* | Predicting depression from Twitter behaviour | Text-based mental-state signal (Exp 7) |
| 5 | Coppersmith et al. (2018), *JMIR* | Quantifying clinical signals from social media | Justifies NLP screening of free text |
| 6 | Turcan & McKeown (2019), *LOUHI* — Dreaddit | Labeled Reddit stress corpus | External / out-of-domain test set |
| 7 | Goodfellow et al. (2013), *IEEE FG* — FER-2013 | 35.9K facial expressions, 7 classes | Image modality dataset (Exp 3) |
| 8 | Barrett et al. (2019), *Psych Sci in the Public Interest* | Critical review of facial-emotion inference | Justifies labelling face output "mood cue, not diagnosis" |
| 9 | Chen & Guestrin (2016), *KDD* — XGBoost | Scalable gradient boosting, native NaN handling | Main tabular risk model (Exp 6) |
| 10 | Selvaraju et al. (2017), *ICCV* — Grad-CAM | Visual explanations for CNNs | Face model explanation (Exp 3/8) |
| 11 | Ribeiro et al. (2016), *KDD* — LIME | Local interpretable explanations | Text + tabular explanation (Exp 8) |
| 12 | Lundberg & Lee (2017), *NeurIPS* — SHAP | Unified feature-importance theory | Core explainability layer (Exp 8) |
| 13 | Sanh et al. (2019) — DistilBERT | Compact distilled transformer | Optional heavy text baseline (Exp 7) |
| 14 | Hardt et al. (2016), *NeurIPS* | Equal opportunity / fairness metrics | Fairness audit (Exp 8) |
| 15 | OSMI (2014–2021) Mental Health in Tech Surveys | Workplace MH data, `treatment` proxy target | Professional population + intervention likelihood |

---

## SLIDE 4 — Research Gaps

**Gap 1 — No multimodal, same-subject public data**
No public dataset records tabular risk factors, journal text and facial images *for the same people*; published multimodal work is either single-source or private.
→ *MindSense builds independent per-modality models with a transparent, rule-based late-fusion summary (never a single fused score).*

**Gap 2 — Explainability + fairness rarely reported together for mental-health screening**
Most screening models report only aggregate accuracy; subgroup (gender / age / population) behaviour and human-readable reasons are usually absent — unacceptable for a high-stakes domain.
→ *MindSense ships SHAP/LIME/Grad-CAM per modality plus a fairness audit (TPR, FPR, demographic-parity & equal-opportunity gaps).*

**Gap 3 — Circular targets and leakage in existing pipelines**
Predicting a DASS severity band from the very items that compute the band is circular, and the popular sentiment corpus aggregates Reddit posts that also appear in "external" test sets.
→ *Depression items are excluded from Exp 4 features; 2,819 verbatim cross-corpus duplicates were detected and quarantined (`text_leakage_report.json`).*

*(Optional 4th: most tools give a score without safety handling → MindSense has a dedicated crisis module.)*

---

## SLIDE 5 — Objectives

1. **Collect & harmonize** 10 ranked mental-health datasets into one documented schema with a data dictionary, inventory and leakage guards.
2. **Build a tabular risk model** (students + tech professionals) with threshold-tuned, recall-favouring evaluation and per-population metrics.
3. **Build a severity/prognosis module** — DASS-42 depression severity band from a compact (≤14-item) screener, plus intervention likelihood + what-if simulator.
4. **Build an NLP module** — 7-class mental-state text classifier, rule-based clinical entity extraction with negation handling, and a conservative crisis trigger.
5. **Build an image mood-cue module** — MobileNetV2 → ONNX facial-affect classifier, explicitly non-diagnostic.
6. **Explain & audit every prediction** (SHAP / LIME / Grad-CAM) and audit fairness across gender, age band and population.
7. **Deliver a deployable, safe Streamlit app** (Docker, HF Spaces) with disclaimers, crisis helplines and a Lab Results page exposing all 10 experiments.

---

## SLIDE 6 — Problem Definition

**Input:** structured lifestyle/questionnaire answers, free text, optional face image.
**Output:** risk tier + probability + top contributing factors + extracted entities + next steps.

- **Target (tabular):** `mh_risk` — *proxy label*: `Depression` (student self-report) OR `treatment == Yes` (professional). **Not a clinical diagnosis.**
- **Target (prognosis):** DASS depression severity band {Normal / Mild / Moderate / Severe / Extremely severe} — ordinal, cross-sectional (**stratification, not forecasting**).
- **Target (text):** 7 states {Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder}.
- **Constraints:** self-reported data, class imbalance (58.2% positive; text 15K vs <1K; FER 16:1 happy:disgust), structural missingness across populations, English-only, no longitudinal data.

---

## SLIDE 7 — Methodology (Architecture)

```
Data sources (10) → Exp1 Pipeline (download→clean→harmonize→split)
        │
        ├─ Tabular harmonized (29,152 rows) ──► Exp6 Risk model (LR/RF/XGBoost) ─┐
        ├─ DASS-42 (38,337) ────────────────► Exp4 Severity + item reduction ───┤
        ├─ Sentiment MH (50,843) ───────────► Exp7 TF-IDF(+DistilBERT) + crisis ┤→ inference.py
        ├─ Drug reviews ────────────────────► Exp7 sentiment/side-effect analytics│   │
        ├─ FER-2013 (35,887) ───────────────► Exp3 MobileNetV2 → ONNX ──────────┤   ▼
        └─ DREADdit (3,532, external) ──────► Exp7 out-of-domain test ──────────┘  Streamlit app
                                          Exp5 spaCy EntityRuler | Exp8 SHAP/LIME/Grad-CAM + fairness
```

| Stage | Technique |
|---|---|
| Data | Fallback ladder (Kaggle API → manual → mirror → tagged synthetic), dedup, validity screen, stratified 70/15/15 split (seed 42), split stored as a column |
| Harmonization | Union schema, `population` column, structural NaN kept (never imputed across populations) |
| Tabular | Logistic baseline → RF → XGBoost (native NaN), `RandomizedSearchCV`, 5-fold stratified, threshold tuned for recall |
| Prognosis | Ordinal/multinomial LogReg, RF, XGBoost; **depression items excluded**; item selection by mutual information; metrics QWK, macro-F1, Brier |
| Text | TF-IDF + LogReg/LinearSVM (deployed) vs DistilBERT (optional); class weights; macro-F1 |
| Entities | `spacy.blank("en")` + EntityRuler/PhraseMatcher + regex + NegEx-style negation, gold set ≥60 sentences |
| Image | MobileNetV2 transfer learning, augmentation, class weights → ONNX (≤20 MB, parity 1e-4) |
| Safety | Rule + probability crisis trigger (PHQ-9 item 9, suicidal-class ≥0.55, high-precision lexicon) |
| Deploy | Streamlit + `inference.py` API, Docker, Hugging Face Spaces; no torch at inference |

---

## SLIDE 8 — Performance Evaluation (Part 1: Data pipeline & EDA — real results)

**Table A — Data pipeline results (Exp 1)**

| Dataset | Rows in | Rows out | Dropped | Key cleaning action |
|---|---|---|---|---|
| Student Depression | 27,901 | 27,901 | 0 | Sleep/diet recoded to numeric |
| Sentiment (MH) | 53,043 | 50,843 | 2,200 | 1,608 duplicates, 230 too-short |
| DASS-42 | 39,775 | 38,337 | 1,438 | Validity screen + items 1-4 → 0-3 |
| OSMI 2014 | 1,259 | 1,251 | 8 | Gender 49 spellings → 3; age outliers |
| OSMI 2017–21 | 756 | 754 | 2 | Age outliers |
| FER-2013 | 35,887 | 35,887 | 0 | 35,887 images indexed |
| DREADdit | 3,553 | 3,532 | 21 | Cross-split duplicate removal |
| **Risk harmonized** | **29,152** | **29,152** | **0** | 70/15/15 stratified split, seed 42 |

Datasets configured/present: **10/10** (1 clearly tagged synthetic stand-in: DrugLib, tier-C).

**Table B — Headline EDA metrics (Exp 2)**

| Metric | Value | Implication |
|---|---|---|
| Positive class (`mh_risk`) | **58.2%** (student 58.6%, prof. 50.5%) | Accuracy misleading → report PR-AUC/recall |
| Top mutual-info features | stress_pressure 0.126, financial_stress 0.066 | `suicidal_thoughts` MI 0.141 → ablation only |
| Max feature correlation | \|r\| ≤ 0.24 | No collinearity blockers |
| Risk by financial stress | **0.32 → 0.81** (monotonic) | Strongest clean gradient |
| DASS depression bands | 37.7% N / 40.3% M / 21.2% Mo / 0.8% S | Skewed sample → ordinal metrics needed |
| Depression ↔ Anxiety band agreement | 46.2% | Partially independent → Exp 4 design valid |
| Text classes | 15K … <1K (7 classes) | Class weights + macro-F1 |
| VADER vs drug rating | Spearman ρ = 0.003 (n=2,341) | Sentiment only a secondary cue |
| FER class ratio | happy 8,989 vs disgust 547 (16:1) | Class weighting mandatory |

*Figures available to embed: `exp02_06_financial.png`, `exp02_10_mi.png`, `exp02_13_dass_bands.png`, `exp02_18_text_labels.png`, `exp02_24_fer_counts.png` — 25 figures total.*

---

## SLIDE 9 — Performance Evaluation (Part 2: Model results — graph/table template)

> ⚠️ `models/` is currently empty and `make train` (`mindsense.models.train_all`) has not been run — **no model metric exists yet.** Do **not** put invented numbers on this slide. Run training, then fill from `reports/metrics/*.json`:

**Table — Model comparison (fill from metrics JSON)**

| Model | Dataset | ROC-AUC | PR-AUC | Recall | F1 | Notes |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | risk_harmonized | ⬜ | ⬜ | ⬜ | ⬜ | Exp 6 |
| Random Forest | risk_harmonized | ⬜ | ⬜ | ⬜ | ⬜ | Exp 6 |
| **XGBoost (deployed)** | risk_harmonized | ⬜ | ⬜ | ⬜ | ⬜ | Threshold tuned for recall |
| External transfer | OSMI 2017–21 | ⬜ | ⬜ | ⬜ | ⬜ | Temporal-shift drop |
| Text: TF-IDF + LogReg | Sentiment MH | — | — | ⬜ | ⬜ macro-F1 | Exp 7 |
| Text: DistilBERT (optional) | Sentiment MH | — | — | ⬜ | ⬜ macro-F1 | Exp 7 |
| External text test | Dreaddit | — | — | ⬜ | ⬜ | Domain-shift gap |
| DASS severity (compact ≤14 items) | DASS-42 | — | — | ⬜ | ⬜ macro-F1 / QWK | Exp 4 |
| Face mood CNN → ONNX | FER-2013 | — | — | ⬜ | ⬜ macro-F1 | Exp 3 (human ≈65%) |

**Charts to render:**
1. Grouped bar — ROC-AUC / PR-AUC / Recall across tabular models
2. Confusion matrix heatmaps — text 7-class, DASS 5-band, FER 7-class
3. Fairness grouped bar — TPR/FPR by gender × age band × population (Exp 8)
4. SHAP summary + waterfall (one slide screenshot from the app)

---

## SLIDE 10 — Conclusion

- **MindSense delivers an end-to-end, explainable mental-wellness screening pipeline** — 10 ranked datasets ingested (10/10 present), 29,152-row harmonized risk table, 8 processed corpora, 25 EDA figures, data dictionary and leakage reports committed.
- **Methodological honesty is built in:** proxy targets declared, depression items excluded from the prognosis model, 2,819 cross-corpus text duplicates quarantined, synthetic data tagged everywhere, no fabricated metrics.
- **Safety & ethics first:** non-dismissable disclaimer, crisis trigger (PHQ-9 item 9 / suicidal-class ≥0.55 / lexicon), helplines (Tele-MANAS 14416, 988, findahelpline.com), in-memory-only privacy, per-subgroup fairness audit.
- **Deployment-ready:** lean runtime (no torch), `inference.py` API contract, Streamlit 8-page app, Docker, CI (lint + tests + notebook smoke).
- **Limitations:** self-reported & convenience samples, English-only, cross-sectional (no forecasting), modest expected performance on external shift sets — reported honestly.
- **Future work:** longitudinal data for true prognosis, multilingual text, federated/on-device inference, clinician-in-the-loop validation.

---

## Build status (before you present)

`exp01` + `exp02` notebooks, the Streamlit app and tests exist; **Exp 3–8 notebooks, trained models and model metrics are still pending**, and `docs/REPORT.md` / `PRESENTATION.md` don't exist yet. Slide 9 will stay empty until `make train` runs.
