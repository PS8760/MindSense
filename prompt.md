# MindSense: End-to-End Mental Health Risk Prediction
### Honors Lab (AI & ML in Healthcare) Mini Project: Final Agent Build Prompt (v2, with ranked datasets)

> **You are an autonomous senior ML engineer / full-stack developer.** Build the complete project described below, end to end: data pipeline, 10 lab-experiment notebooks, trained models, explainability, a deployable web app, tests, documentation and presentation material. Work in the order given in Section 13, verify everything actually runs, and do not stop at scaffolding.

---

## 0. Rules of Engagement

1. **Run everything you write.** Every notebook must execute top to bottom without errors, with outputs saved. Every `Makefile` target must work. Do not claim a result you did not produce.
2. **Never fabricate metrics or data silently.** If a real dataset can't be obtained (e.g. missing Kaggle credentials), follow the fallback ladder in Section 5.6 and label synthetic data clearly everywhere (notebook banner, README, app footer).
3. **Reproducibility:** global seed `42`, pinned dependencies, stratified splits, saved split indices.
4. **Keep it deployable.** The runtime app must be lightweight (no PyTorch/TensorFlow at inference by default). Heavy training dependencies live in separate requirements files.
5. **This is a screening/educational tool, not a diagnostic device.** Disclaimers and crisis support are mandatory (Section 10).
6. **Ask only if truly blocked.** Otherwise make a reasonable decision, document it in `docs/DECISIONS.md`, and move on.
7. Write clean, typed, documented Python (PEP8, docstrings, `ruff` clean). Reusable logic goes in `src/mindsense/`; notebooks import from it instead of duplicating code.
8. **Verify dataset facts yourself.** Dataset names, slugs, row counts and column names below come from research and may have drifted. Inspect the real files, record what you find in `docs/DATA_DICTIONARY.md`, and record any difference in `docs/DECISIONS.md`.

---

## 1. Project Context & Goal

**Problem:** In today's fast-paced world, mental health concerns (stress, anxiety, depression) affect people of every age and gender, and early screening is often missed due to stigma, cost and access.

**Goal:** Build **MindSense**, a web application that provides an early, explainable, privacy-respecting *mental-wellness risk screening* by combining:

- structured self-report data (lifestyle, pressure, sleep, history) and standard questionnaires (PHQ-9, GAD-7, DASS items),
- free-text journal / social-media-style input (NLP),
- an optional facial-expression mood cue (image modality),

and returning **risk level, severity/prognosis, explanation, extracted clinical entities, and guidance to professional help**.

**Academic constraint:** The project must incorporate and showcase the results of all **10 lab experiments** (Section 3). The app must contain a **"Lab Results"** area where each experiment's outputs (tables, plots, metrics) are visible.

---

## 2. Design Thoughts & Key Decisions (read before coding)

Record any deviation in `docs/DECISIONS.md`.

| # | Observation | Decision |
|---|---|---|
| 1 | No single public dataset covers tabular risk factors, text, and images for the same people. | Build **independent models** (tabular risk, questionnaire severity, text, image) and combine them via **late fusion for display only** (transparent dashboard and rule-based summary). Do **not** claim a jointly trained multimodal model. |
| 2 | Exp 3 mentions MRI/X-ray diagnosis. Mental health is **not clinically diagnosed from routine MRI/X-ray**, and no suitable public MRI dataset exists for general screening. | Implement Exp 3 as an **image-based affect / diagnostic-support module**: a CNN (transfer learning) on FER-2013 that outputs a mood cue. State clearly that it is a *supplementary, non-diagnostic signal*. Include a written justification and an **optional** brain-MRI extension only if time permits. |
| 3 | Public datasets are self-reported and have no clinician-verified labels. | Describe outputs as **"risk indicators / screening"**, never "diagnosis". Report limitations honestly in model cards. |
| 4 | Datasets skew towards students, tech workers and online volunteers. | Use a harmonized schema with a `population` column, report metrics **per population and per gender/age band**, and discuss bias. |
| 5 | No longitudinal data exists, so true time-series prognosis is impossible. | Define Exp 4 as **prognostic severity stratification**. The **DASS-42 dataset provides real severity bands** (scored from a validated instrument) as the main target, plus an intervention-likelihood model on OSMI, plus a **what-if simulator**. State clearly that this is cross-sectional and not a forecast. |
| 6 | Deployment must be easy. | **Python-only stack: Streamlit app** importing an inference package directly (no separate backend). Docker + Hugging Face Spaces as the primary target. Optional FastAPI wrapper as a stretch. |
| 7 | Transformers are heavy for free hosting. | Default deployed text model = TF-IDF + linear model. Fine-tune DistilBERT in the notebook as a comparison and keep it **optional** (`ENABLE_TRANSFORMER=1`). Export the face CNN to **ONNX** so the app uses `onnxruntime` (no torch). |
| 8 | Entity extraction normally needs large pretrained models. | Use `spacy.blank("en")` + `EntityRuler`/`PhraseMatcher` + regex with a curated lexicon: no model downloads, fast, deterministic. Evaluate against a small annotated gold set. |
| 9 | The large text dataset is an aggregation of many other datasets. | Treat it as a single source. **Never** also use its constituent datasets (or derived copies) for training or testing, because that causes duplicate/leakage. Deduplicate before splitting. |
| 10 | DASS severity is computed *from* the item responses. | Predicting a DASS subscale from its own items is circular. Exclude the target subscale's items from the features (Section 7, Exp 4). |

---

## 3. Experiment → Deliverable Mapping

| Lab Exp. | Title | Deliverable in this project |
|---|---|---|
| 1 | Collect, Clean, Integrate and Transform Healthcare Data (specific disease) | `exp01_data_pipeline.ipynb` + `src/mindsense/data/`: download, clean, harmonize multiple mental-health sources, transform, processed data, data dictionary, data inventory |
| 2 | Exploratory Data Analysis | `exp02_eda.ipynb` + app page **Population Insights** |
| 3 | AI for medical diagnosis (MRI/X-ray) | `exp03_image_diagnostic_support.ipynb`: CNN on facial-expression images (decision #2); app page **Face Mood Cue (optional)** |
| 4 | AI for medical prognosis | `exp04_prognosis.ipynb`: DASS severity-band model + OSMI intervention-likelihood model + what-if simulator; app page **Prognosis & What-If** |
| 5 | NLP entity extraction from medical reports | `exp05_entity_extraction.ipynb` + `src/mindsense/nlp/entities.py`; app: highlighted entities |
| 6 | Predict disease risk from patient data | `exp06_risk_prediction.ipynb`: main tabular risk model (Student + OSMI); app page **Risk Assessment** |
| 7 | Medical reviews analysis from social media data | `exp07_text_mining.ipynb`: mental-state classification of posts, stress detection, drug-review analytics; app page **Text Check-in** |
| 8 | Explainable AI | `exp08_explainability.ipynb` + `src/mindsense/explain/`; SHAP/LIME/Grad-CAM + fairness audit |
| 9 | Mini project: web/mobile AI app | The full **Streamlit app** (mobile-responsive), Dockerized and deployable |
| 10 | Documentation and presentation | `README.md`, `docs/REPORT.md`, `docs/MODEL_CARD.md`, `docs/PRESENTATION.md`, `docs/DEMO_SCRIPT.md`, screenshots |

---

## 4. Tech Stack (use exactly this unless blocked)

**Language:** Python 3.11

**Data & classical ML:** `pandas`, `numpy`, `scikit-learn`, `xgboost` (or `lightgbm`), `imbalanced-learn` (only if needed), `joblib`, `pyarrow`
**Explainability:** `shap`, `lime`
**NLP:** `spacy` (blank pipeline + rules), scikit-learn TF-IDF, `vaderSentiment` (reviews); training-only: `transformers`, `datasets`, `torch` (CPU is fine; document a Colab GPU option)
**Image (training only):** `torch`, `torchvision` (MobileNetV2 transfer learning) → export **ONNX**; runtime: `onnxruntime`, `opencv-python-headless`, `Pillow`
**Data access:** `kaggle` (API), `scikit-learn` `fetch_openml`, Hugging Face `datasets`, `requests`
**Visualisation:** `matplotlib`, `seaborn`, `plotly`
**App:** `streamlit` (multi-page), `plotly`
**Quality:** `pytest`, `ruff`, `nbconvert`/`papermill`, `pre-commit` (optional)
**Packaging/Deploy:** `Dockerfile`, `Makefile`, GitHub Actions CI (lint + tests + notebook smoke test), Hugging Face Spaces (primary), Streamlit Community Cloud / Render (alternatives)

**Dependency files**
- `requirements.txt`: **lean runtime** (streamlit, pandas, numpy, scikit-learn, xgboost, shap, lime, spacy, onnxruntime, opencv-python-headless, plotly, joblib, pyyaml, vaderSentiment). **No torch/transformers.**
- `requirements-train.txt`: everything for notebooks/training (adds torch, transformers, datasets, kaggle, jupyter, papermill).
- `requirements-transformer.txt`: optional extras for the DistilBERT path in the app.
Pin versions you have actually tested together.

---

## 5. Datasets: Ranked Inventory, Usage and Ingestion

### 5.1 Ranking method

Each dataset is scored out of 100 (editorial judgement, used only to prioritise effort):
**Relevance to the project objective (30)** + **Experiment coverage & irreplaceability (25)** + **Size/quality (20)** + **Ease of access (15)** + **Licence/deployability clarity (10)**.
Tier **A = core** (project is incomplete without it or an approved fallback), **B = supplementary**, **C = optional**.

### 5.2 Ranked list of datasets to use

| Rank | Dataset | Tier | Score | Modality / population | Primary role | Experiments |
|---|---|---|---|---|---|---|
| 1 | **Student Depression Dataset** (Kaggle, `hopesb/student-depression-dataset`) | A | 85 | Tabular, students (~27.9K rows; binary `Depression` label; age, gender, city, CGPA, sleep, academic/work pressure, satisfaction, diet, study hours, financial stress, family history, etc.) | Main risk-prediction data for the student population | 1, 2, 6, 8 |
| 2 | **Sentiment Analysis for Mental Health** (Kaggle, `suchintikasarkar/sentiment-analysis-for-mental-health`) | A | 80 | Text, social media (~52.7K statements; Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder; imbalanced) | Main mental-state text classifier and crisis-language signal | 1, 2, 5, 7, 8 |
| 3 | **DASS-42 raw responses** (Open Psychometrics: `https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip`; no login) | A | 78 | Questionnaire + demographics (42 items scored 0–3 after recoding; age, gender, country, education, urban/rural, etc.) | **Real severity labels** for prognosis; item-importance analysis; population insights | 1, 2, 4, 6, 8 |
| 4 | **OSMI Mental Health in Tech Survey 2014** (OpenML `Mental-Health-in-Tech-Survey`, data ID 43674, also listed under 43664; no login) | A | 74 | Tabular, tech professionals (age, gender, country, family history, `treatment`, `work_interfere`, workplace benefits/support, etc.) | Professional-population risk and **intervention-likelihood** target | 1, 2, 4, 6, 8 |
| 5 | **Drug Reviews (Drugs.com)** (UCI, dataset 462; Kaggle mirror `jessicali9530/kuc-hackathon-winter-2018`) | A | 70 | Text, patient reviews (drug name, condition, review, 10-star rating, date, useful count) | "Medical reviews analysis": sentiment/side-effect analytics for Depression/Anxiety/Bipolar/Insomnia; lexicon building for Exp 5 | 2, 5, 7 |
| 6 | **FER-2013** (Kaggle, `msambare/fer2013`) | A | 68 | Images, 48×48 grayscale faces, 7 emotions (~35.9K; Disgust is rare) | Only image modality; mood-cue CNN | 3, 8 |
| 7 | **Dreaddit** (Hugging Face, `andreagasparini/dreaddit`; original Turcan & McKeown, Louhi 2019; no login) | B | 62 | Text, Reddit posts, stress label (small labelled subset) | Stress-detection cross-check; **external test** for the stress/anxiety classes | 7, 8 |
| 8 | **OSMI Mental Health in Tech Surveys 2017–2021** (Kaggle `osmihelp` datasets; combined version on Mendeley, DOI 10.17632/mmnzx4w8cg.1) | B | 58 | Tabular, tech professionals, more recent | Temporal-shift / robustness check against the 2014 survey; extra features | 2, 6, 8 |
| 9 | **Depression Student Dataset** (Kaggle, `ikynahidwin/depressionstunt-dataset`, ~502 rows) | C | 43 | Tabular, students, small | **External sanity-check set** for the student model (only if columns map cleanly) | 6, 8 |
| 10 | **Drug Reviews (Druglib.com)** (UCI, dataset 461) | C | 43 | Text, aspect-rated drug reviews (benefits / side effects / overall) | Optional aspect-based sentiment and cross-source transfer test | 7 |

### 5.3 Do NOT use

| Dataset / source | Reason |
|---|---|
| Hugging Face merged copies such as `gokulan006/risk-analysis-dataset`, or any other derived merge of the Sentiment Analysis for Mental Health data | Duplicates rank-2 content; train/test leakage. |
| The constituent datasets that rank 2 aggregates (e.g. Depression Reddit Cleaned, Suicidal Tweet Detection, Human Stress Prediction, Reddit Mental Health Data, Students Anxiety and Depression) | Same posts already inside rank 2. Using them as "external" tests would be leakage. |
| Very small classroom survey datasets (e.g. ~100-row student surveys) | Too small for credible modelling. |
| Any dataset with scraped personal data or unclear provenance | Ethics and licence risk. |

### 5.4 Dataset → Experiment usage matrix

| Dataset | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 |
|---|---|---|---|---|---|---|---|---|
| 1 Student Depression | ● | ● | | ○ | | ● | | ● |
| 2 Sentiment (MH) | ● | ● | | | ● | | ● | ● |
| 3 DASS-42 | ● | ● | | ● | | ○ | | ● |
| 4 OSMI 2014 | ● | ● | | ● | | ● | | ● |
| 5 Drug Reviews | ● | ● | | | ○ | | ● | |
| 6 FER-2013 | ● | ● | ● | | | | | ● |
| 7 Dreaddit | ○ | | | | | | ● | ● |
| 8 OSMI 2017–21 | ○ | ● | | | | ○ | | ○ |
| 9 ikynahidwin students | | | | | | ○ | | ○ |
| 10 Druglib | | | | | | | ○ | |

● = primary use, ○ = secondary/optional use.

### 5.5 Dataset-specific handling notes

**1. Student Depression:** Inspect real columns first. Common issues: odd category values in `Sleep Duration`/`Dietary Habits`, a few rows with invalid or placeholder values, `Profession`/`Degree` high-cardinality fields, and the sensitive `Have you ever had suicidal thoughts?` field. **Exclude the suicidal-thoughts field from the default deployed model**, and report an ablation with it included. Note that this dataset is not guaranteed to represent clinical truth.

**2. Sentiment Analysis for Mental Health:** Drop null and duplicate statements, normalise text, check for label noise (spot-check ≥ 50 samples per class), and handle the class imbalance (class weights, macro-F1). Consider also reporting a merged "Anxiety/Stress" view. Keep the original 7-class labels as the primary target.

**3. DASS-42 (Open Psychometrics):**
- Read the codebook shipped in the zip. Item responses are typically recorded as 1–4 and must be **recoded to 0–3**. There are also timing/position columns, other personality-scale items and vocabulary validity-check items.
- Remove invalid respondents: those who endorse the fake vocabulary words (validity items), those with impossible ages, straight-lining/implausibly fast completion if timing is available.
- Compute subscale scores with the standard DASS-42 key. Depression items: 3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42. Stress items: 1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39. Anxiety items: the remaining 14 (2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41). **Verify this key and the severity cut-offs against the official DASS manual (UNSW DASS page) and cite it.**
- Severity bands (Lovibond & Lovibond; verify): Depression 0–9 Normal, 10–13 Mild, 14–20 Moderate, 21–27 Severe, 28+ Extremely severe. Anxiety 0–7, 8–9, 10–14, 15–19, 20+. Stress 0–14, 15–18, 19–25, 26–33, 34+.
- Check and cite usage terms for the DASS instrument and the Open Psychometrics data.

**4. OSMI 2014:** Clean gender free-text into {male, female, other/non-binary}, remove impossible ages, drop `comments`/`Timestamp`, treat `work_interfere` NaN as its own "not applicable" category (document it). Keep `treatment` as the proxy label ("has sought treatment"), and describe it as such.

**5. Drug Reviews (Drugs.com):** **Research-use only, no commercial use, no redistribution, citation required.** Do **not** commit the data to the repo; download via script. Filter to conditions of interest (Depression, Anxiety, Bipolar Disorder, Insomnia, ADHD, Panic Disorder, etc.; normalise names), remove HTML entities, drop empty reviews. Provide descriptive analytics only; no medical advice.

**6. FER-2013:** Use the folder-structured Kaggle version (train/test class folders). Handle class imbalance (Disgust). Stratified validation split from train.

**7. Dreaddit:** Use only as an **external test/stress cross-check**, not to inflate training on the same distribution. Keep the original train/test split.

**8. OSMI 2017–21:** Align columns carefully (question wording changed across years). Use only columns that map cleanly onto the harmonized schema.

**9–10:** Optional; skip if time is short and say so in the report.

### 5.6 Acquisition & fallback ladder

Put raw files in `data/raw/<dataset_key>/` (git-ignored), processed in `data/processed/`. `scripts/download_data.py` must, **for each dataset**, try in order:

1. **Automated download:** Kaggle API for Kaggle sets (`kaggle datasets download -d <owner>/<slug> -p data/raw/<key> --unzip`; document `kaggle.json` setup: Kaggle → Settings → API → Create New Token → `~/.kaggle/kaggle.json`, `chmod 600`), `fetch_openml` for OSMI 2014 (try ID 43674, else search by name), Hugging Face `datasets.load_dataset("andreagasparini/dreaddit")`, direct download for DASS, UCI page/zip or Kaggle mirror for Drugs.com.
2. **Manual placement:** if the download fails, print exact instructions (URL, target folder, expected filename) and accept files placed manually.
3. **Alternate source:** the mirror or alternate listed in the table (e.g. the Kaggle mirror for Drugs.com).
4. **Synthetic fallback (last resort):** `scripts/make_synthetic_data.py` generates a seeded substitute with the *same schema*. Tag rows `data_source="synthetic"`, show a visible warning in notebooks, the Lab Results page and the README, and never present synthetic-data metrics as real-world performance. Do **not** use synthetic data for a dataset that is available.

Also generate **`reports/tables/data_inventory.csv`**: dataset key, rank, source URL, access date, file hash, row/column counts, licence/terms, `real|synthetic`, and experiments used.

### 5.7 Data hygiene rules

- Strip/ignore direct identifiers; never store user inputs from the deployed app on disk.
- Maintain `docs/DATA_DICTIONARY.md` (column, type, source, meaning, transformation) and `docs/DATA_SOURCES.md` (citations, URLs, licence/terms, access date).
- Deduplicate text across and within datasets before splitting. Split **before** any fitting (vectorisers, scalers, selectors).
- Treat sensitive items carefully (see 5.5) and keep them out of the default deployed model.

---

## 6. Repository Layout

```
mindsense/
├── README.md
├── Makefile                     # setup | data | train | notebooks | test | app | docker
├── Dockerfile
├── requirements.txt  requirements-train.txt  requirements-transformer.txt
├── .gitignore  .dockerignore  .env.example
├── .github/workflows/ci.yml
├── config/{config.yaml, helplines.yaml}
├── data/{raw,interim,processed}/
├── notebooks/
│   ├── exp01_data_pipeline.ipynb
│   ├── exp02_eda.ipynb
│   ├── exp03_image_diagnostic_support.ipynb
│   ├── exp04_prognosis.ipynb
│   ├── exp05_entity_extraction.ipynb
│   ├── exp06_risk_prediction.ipynb
│   ├── exp07_text_mining.ipynb
│   ├── exp08_explainability.ipynb
│   ├── exp09_app_walkthrough.ipynb
│   └── exp10_results_summary.ipynb
├── src/mindsense/
│   ├── data/{download.py, clean.py, harmonize.py, features.py, dass.py, synthetic.py}
│   ├── models/{tabular.py, prognosis.py, text.py, image.py, train_*.py}
│   ├── nlp/{entities.py, lexicon.py, preprocess.py}
│   ├── explain/{shap_utils.py, lime_utils.py, fairness.py}
│   ├── screening/{phq9.py, gad7.py, dass.py, crisis.py}
│   ├── inference.py
│   └── utils/{io.py, logging.py, plotting.py}
├── app/
│   ├── Home.py
│   ├── pages/{1_Risk_Assessment, 2_Text_Check_in, 3_Prognosis_What_If, 4_Face_Mood_Cue, 5_Population_Insights, 6_Lab_Results, 7_About_Ethics}.py
│   ├── components/
│   └── assets/
├── models/                      # artifacts + metadata.json
├── reports/{figures,metrics,tables}/
├── tests/
├── scripts/{download_data.py, make_synthetic_data.py, run_all_notebooks.sh}
├── deploy/hf_space_README.md
└── docs/{REPORT.md, MODEL_CARD.md, DATA_SOURCES.md, DATA_DICTIONARY.md, DECISIONS.md, PRESENTATION.md, DEMO_SCRIPT.md}
```

---

## 7. Detailed Specification per Experiment

For every experiment: (a) markdown intro (aim, theory in 4–6 lines, method), (b) well-commented code, (c) results saved to `reports/metrics/expXX_*.json` and figures to `reports/figures/expXX_*.png`, (d) *Observations & Limitations* cell, (e) conclusion. Every trained model saves `models/<name>/model.*` plus `metadata.json` (training date, datasets and `real|synthetic` flag, git hash, metrics, features, version, intended use).

### Experiment 1: Collect, Clean, Integrate, Transform
- Ingest datasets 1–8 (9–10 optional) per Section 5.6; produce the data inventory.
- **Clean** per Section 5.5 (duplicates, inconsistent categories, impossible values, missing-value strategy documented per column, outliers, text normalisation: URLs, handles, emojis→tokens, lowercase, length filters).
- **Integrate (tabular risk):** a **harmonized schema** from the Student and OSMI datasets with common features such as `age`, `gender`, `population` (`student`/`professional`), `sleep_hours` (bucketed), `stress_pressure_score` (0–5), `satisfaction_score`, `work_study_hours`, `financial_stress`, `family_history`, `support_available`, `diet_quality`. Target `mh_risk` = `Depression` for students, `treatment == Yes` for professionals. **State clearly this is a proxy target.** Keep a per-population-model option if harmonization hurts performance.
- **Integrate (questionnaire track):** DASS-42 stays a **separate table** (items + demographics + computed subscale scores and severity bands). Do not force-merge it with the lifestyle schema.
- **Integrate (text):** Sentiment (rank 2) as main corpus; Dreaddit and Drug Reviews stay separate with their own schemas.
- **Transform:** encoding, scaling, stratified train/val/test (70/15/15 by target and population), leakage checks, `ColumnTransformer` saved with the model.
- Output: `data/processed/*.parquet`, data dictionary, **data-quality report** (missingness before/after, row counts per step), data-lineage diagram (Mermaid).

### Experiment 2: Exploratory Data Analysis
- Tabular: target balance, risk by age band, gender, sleep, pressure, financial stress, family history, work interference; correlation heatmap (Cramér's V for categoricals), mutual-information ranking; compare students vs professionals.
- DASS: subscale distributions and severity-band shares, co-occurrence (depression × anxiety × stress), differences by age/gender/country/urban-rural, item-level means.
- Text: class distribution, length distribution, top n-grams per class, emotion-lexicon trends; drug reviews: rating distributions per condition, review length, sentiment vs rating.
- Images: class counts, sample grids, pixel-intensity stats.
- At least **15 meaningful plots**, each with a 1–2 line takeaway. Save an interactive subset (plotly) for **Population Insights** with filters (population, gender, age band).

### Experiment 3: Image-Based Diagnostic Support (CNN)
- FER-2013 → keep 7-class output and add a **mood-cue** grouping (negative-affect: angry/disgust/fear/sad; neutral; positive-affect: happy/surprise).
- MobileNetV2 transfer learning (freeze → fine-tune), augmentation, class weighting. Report accuracy, macro-F1, per-class recall, confusion matrix, Grad-CAM examples. Mention that human accuracy on this dataset is only about 65%, so expectations must be realistic.
- Export to **ONNX**, verify numerical parity with PyTorch (tolerance 1e-4), target size ≤ 20 MB.
- State clearly that the facial cue is **not** a diagnostic indicator; discuss bias (lighting, demographics, dataset limitations).
- App: `st.camera_input` / upload, **in-memory only**, Haar-cascade face detection, graceful "no face found" handling.
- *Optional extension:* brain-MRI classifier notebook, flagged "research demo, not part of the screening flow".

### Experiment 4: Prognosis (severity stratification)
- **Model A (primary): DASS severity band.**
  - Target: depression severity band (Normal / Mild / Moderate / Severe / Extremely severe; consider merging the top two if sparse), computed from the DASS key.
  - Features: **anxiety and stress items + demographics only. Exclude all depression items** (decision #10).
  - Item reduction: select a compact subset (≤ 14 items) via mutual information / L1 / permutation importance. Show that macro-F1 stays within a documented margin of the 28-item model. The app asks only these items.
  - Models: ordinal/multinomial logistic regression, Random Forest, XGBoost; 5-fold CV; report QWK (quadratic weighted kappa), macro-F1, MAE over ordinal bands, confusion matrix; calibrate probabilities (Brier + reliability curve).
- **Model B (secondary): intervention likelihood** on OSMI 2014: predict `treatment` (and, as an ordinal view, `work_interfere`), with calibration and subgroup metrics.
- **What-if simulator:** for Model B and the lifestyle-risk model, vary modifiable factors (sleep, work/study hours, financial stress, support available) and show the change in predicted risk, with a clear "associational, not causal" warning.
- Output a **prognosis card**: severity tier, probability, top contributing factors, modifiable levers.
- State clearly: data are cross-sectional; "prognosis" here means risk/severity stratification, not forecasting.

### Experiment 5: NLP Entity Extraction from Medical Reports
- `lexicon.py` with curated term lists: **SYMPTOM** (insomnia, low mood, anhedonia, panic attacks, fatigue, appetite change…), **CONDITION** (major depressive disorder, GAD, bipolar, PTSD…), **MEDICATION** (sertraline, fluoxetine, escitalopram, alprazolam…; seed from drug names in the Drug Reviews data), **DURATION** (regex: "for 3 weeks", "since 2 months"), **SEVERITY** (mild/moderate/severe, "x/10"), **STRESSOR** (exam, layoffs, bereavement, breakup, debt…), **SUPPORT** (therapy, CBT, counselling, support group).
- Implement with `spacy.blank("en")` + `EntityRuler`/`PhraseMatcher` + regex; add NegEx-style negation ("denies suicidal ideation" must not be flagged as a positive finding).
- Create a **gold set of ≥ 60 short synthetic clinical-note sentences** (clearly labelled synthetic) with hand-checked annotations; report per-entity precision/recall/F1 and error analysis. Also run the extractor on a sample of real posts from the Sentiment dataset and Drug Reviews and report qualitative examples (no gold labels needed).
- Optional comparison with a pretrained biomedical NER (training environment only).
- App: highlighted entities (HTML spans + legend), structured table, downloadable JSON.

### Experiment 6: Disease Risk Prediction from Patient Data (core tabular model)
- Features from the harmonized schema (Student + OSMI). PHQ-9/GAD-7 scores are computed in the app as **separate rule-based screening outputs**, not model inputs.
- Models: Logistic Regression (baseline), Random Forest, XGBoost/LightGBM, optional small MLP. `RandomizedSearchCV`/Optuna with 5-fold stratified CV.
- Metrics: ROC-AUC, **PR-AUC**, recall, precision, F1, specificity, Brier, confusion matrix, **threshold tuning favouring recall** (justify the chosen threshold). Report **overall and per population/gender/age band**.
- Leakage audit and **ablation** (with/without sensitive features; with/without population indicator).
- Robustness checks: evaluate the student model on dataset 9 if columns map; evaluate the professional model on OSMI 2017–21 (temporal shift). Report the drop honestly.
- Secondary (optional): DASS-derived analysis of which items drive each severity band.
- Save the best pipeline via `joblib` with metadata; target artifact ≤ 50 MB.

### Experiment 7: Medical Reviews & Social Media Analysis
- **Part A (posts, Sentiment dataset):** classify the 7 states. Baselines: TF-IDF + Logistic Regression / Linear SVM. Advanced: fine-tune **DistilBERT** (2–3 epochs, CPU/Colab) for comparison. Metrics: macro-F1, per-class P/R/F1, confusion matrix; discuss confusable classes (Depression vs Suicidal vs Anxiety).
- **Part A2 (external check):** evaluate the stress-related behaviour of the trained model on **Dreaddit** (binary stress vs not; map Stress/Anxiety classes → "stress") and report the domain-shift gap.
- **Part B (Drug Reviews):** sentiment (VADER or TF-IDF model on rating-derived labels), side-effect keyword/aspect analysis, **condition-wise comparison** (e.g. which common antidepressants show more negative side-effect mentions). Present as descriptive analytics, not advice. Optional: cross-source transfer to Druglib (dataset 10).
- **Deployed text model:** the best model satisfying `size ≤ 50 MB` and `latency ≤ 200 ms` per input on CPU. DistilBERT path optional behind `ENABLE_TRANSFORMER`.
- **Safety layer** (`screening/crisis.py`): conservative rule + model-probability trigger for self-harm language; triggering immediately shows crisis resources (Section 10) and suppresses casual outputs. Evaluate recall of the trigger on a hand-written test set and on held-out "Suicidal" examples; report false-negative rate prominently.

### Experiment 8: Explainable AI
- **Tabular:** global SHAP (summary/bar/dependence), local SHAP waterfall (shown in the app), LIME comparison on 3 cases, permutation importance, partial dependence for sleep/pressure.
- **DASS model:** item-level importance and which items the compact screener retains.
- **Text:** top contributing n-grams per class, LIME text explanations; token highlighting in the app.
- **Image:** Grad-CAM from Exp 3.
- **Fairness audit** (`explain/fairness.py`): by gender/age band/population (TPR, FPR, selection rate, demographic-parity and equal-opportunity gaps), plus country group for DASS, with plots and a discussion of mitigations tried (re-weighting, per-group thresholds) and trade-offs.
- **Plain-language explanations:** convert top SHAP factors into short human-readable sentences.
- Document explanation limitations (SHAP ≠ causality).

### Experiment 9: Mini Project: The Web App
See Section 8.

### Experiment 10: Documentation & Presentation
- `README.md`: pitch, architecture diagram (Mermaid), screenshots/GIF, quickstart, deployment, structure, **results summary table**, **dataset table (Section 5.2)**, limitations, licence, acknowledgements.
- `docs/REPORT.md`: Abstract, Introduction, Related Work (verifiable sources only), Datasets (ranked table and justification), Methodology per experiment, Results, Discussion, Ethics, Limitations, Conclusion, Future Work, References.
- `docs/MODEL_CARD.md` per model (intended use, data, metrics, subgroup performance, limitations, ethics).
- `docs/PRESENTATION.md`: **12–15 slide outline with speaker notes** (problem → solution → architecture → datasets → experiments → results → live demo → ethics → future work → Q&A with 10 likely viva questions and answers).
- `docs/DEMO_SCRIPT.md`: 5-minute demo flow with sample inputs.
- If a `pptx` skill/tool is available, also generate `docs/MindSense_Presentation.pptx`.

---

## 8. Application Specification

**Name:** MindSense. **Tagline:** *Early, explainable mental-wellness screening.*

**Global UI:** calm palette (soft blues/greens), large readable type, mobile-responsive, light/dark friendly. Persistent sidebar with a **non-dismissible disclaimer** ("Not a medical diagnosis. If you are in distress, seek help.") and a **Help now** button. Region selector for helplines (default India).

**Pages**
1. **Home:** purpose, how it works (3 steps), privacy note, quick links.
2. **Risk Assessment (Exp 6 + 8):** short lifestyle form (~12 inputs) and optional **PHQ-9 + GAD-7** questionnaire with standard bands (PHQ-9: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderately severe, 20–27 severe; GAD-7: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–21 severe). PHQ-9 item 9 > 0 triggers the crisis banner immediately. Output: risk gauge + tier, SHAP waterfall, plain-language explanation, next steps.
3. **Text Check-in (Exp 7 + 5):** free-text box: predicted mental-state signal with probabilities, influential words, **extracted entities**, crisis detection. Also a "clinical-style note" mode demonstrating Exp 5.
4. **Prognosis & What-If (Exp 4):** (a) compact DASS-style item screener (the ≤ 14 items selected in Exp 4) + demographics → depression-severity band estimate with confidence; (b) intervention-likelihood from lifestyle/workplace inputs with what-if sliders and before/after chart; associational-not-causal warning. Optional **session-only mood tracker** (`st.session_state`, nothing persisted).
5. **Face Mood Cue (Exp 3, optional):** camera/upload, emotion + mood cue, Grad-CAM overlay, big "non-diagnostic" label. Feature flag `ENABLE_FACE=1`.
6. **Population Insights (Exp 2):** interactive plotly dashboards with filters.
7. **Lab Results (Exp 1–10):** one tab per experiment rendering saved metrics JSON, tables and figures; includes the **ranked dataset table**, data-source tags (real vs synthetic), model versions, subgroup/fairness tables.
8. **About & Ethics:** methodology, data sources, limitations, model cards, privacy policy, helplines, credits.

**Fusion summary:** when more than one module has been used in a session, show a **transparent rule-based summary** ("Questionnaire: moderate; text signal: anxiety; no crisis indicators") with a note that modules are independent. Never output a single fused numeric score.

**Engineering requirements**
- `inference.py` exposes pure functions: `predict_risk`, `predict_severity`, `predict_intervention`, `analyze_text`, `extract_entities`, `analyze_face`; the UI never loads models directly.
- Load models once (`st.cache_resource`); validate all inputs with clear errors.
- Cold start ≤ 20 s on a free CPU instance; single prediction ≤ 2 s (face/transformer ≤ 5 s).
- No user data logged or stored.
- Accessibility: colour-contrast safe, alt text, no colour-only encoding of risk.

---

## 9. Compliance with Dataset Terms (mandatory)

- **Drug Reviews (Drugs.com):** research use only, no commercial use, no redistribution, cite the source. Never commit the raw data or publish it from the deployed app; show only aggregated statistics.
- **Kaggle datasets:** respect each dataset's licence shown on its page; record it in `docs/DATA_SOURCES.md`. Do not re-host raw files in the repo or Docker image. Models trained on them may be shipped, with a model card noting the data origin.
- **OSMI, Dreaddit, DASS/Open Psychometrics:** attribute the sources and follow their terms. Cite Dreaddit as Turcan & McKeown (2019); cite the DASS authors (Lovibond & Lovibond) and Open Psychometrics (2019).
- If any licence is unclear, flag it in `docs/DECISIONS.md` for the user and avoid redistribution.

---

## 10. Safety, Ethics & Privacy (mandatory)

1. **Disclaimers** on every page; never use "diagnose" or "you have depression"; use "may indicate" / "screening result".
2. **Crisis handling:** if PHQ-9 item 9 > 0, self-harm language is detected, or the "Suicidal" class probability exceeds the configured threshold → show a prominent, calm support banner with helplines and encouragement to contact a trusted person or professional. Do not show risk charts or casual tips in that state. Never provide or discuss self-harm methods.
3. **Helplines** in `config/helplines.yaml` with a region selector. Defaults: **India: Tele-MANAS 14416**; **International directory: https://findahelpline.com**; **US: 988**. **Verify every number/URL is current before shipping.**
4. **Privacy:** in-memory processing only; no cookies/analytics; images never saved; privacy statement on the About page.
5. **Bias & fairness:** subgroup metrics, limitations (self-reported data, student/tech/online-volunteer skew, English-only text, facial-emotion bias), and avoid using gender as a causal factor in explanations.
6. **Transparency:** model cards, data sources, version stamps, real-vs-synthetic tags.
7. **Appropriate claims:** all docs state this is an educational academic project and not a substitute for professional care.

---

## 11. Deployment (make it easy)

**Primary: Hugging Face Spaces (free CPU)**
- `Dockerfile` (python:3.11-slim, non-root user, `pip install --no-cache-dir -r requirements.txt`, copy `src/ app/ models/ reports/ config/`, expose 7860, `CMD streamlit run app/Home.py --server.port=7860 --server.address=0.0.0.0`). Add HF Space README front-matter (`sdk: docker`, `app_port: 7860`) in `deploy/hf_space_README.md` with step-by-step instructions.
- Use Git LFS for any file > 10 MB (document it). **Do not include raw datasets in the image.**

**Alternatives (exact steps in README):**
- **Streamlit Community Cloud:** connect repo, main file `app/Home.py`, lean `requirements.txt` (1 GB RAM limit respected).
- **Render / Railway:** Docker deploy with the same Dockerfile.
- **Local:** `make setup && make app`, and `docker build -t mindsense . && docker run -p 8501:8501 mindsense`.

**Flags (`.env.example`):** `ENABLE_FACE`, `ENABLE_TRANSFORMER`, `DEFAULT_REGION`, `APP_ENV`.
**CI:** install, `ruff`, `pytest`, app import smoke test, lightweight notebook smoke test on synthetic data.
Verify the Docker image **builds and responds** (health check `/_stcore/health`) and report the image size.

---

## 12. Testing & Acceptance Criteria

**Tests (`pytest`)**
- Data: schema validation of processed datasets; no train/test leakage (including text duplicates across splits); no NaNs after the pipeline; DASS scoring key and severity bands reproduce known worked examples.
- Screening: PHQ-9/GAD-7 scoring edge cases (all-zero, max, item-9 trigger).
- Models: artifacts load; prediction shape/range; deterministic with the same seed; ONNX-vs-torch parity (if torch installed); the DASS model never receives depression items.
- NLP: entity extractor on known cases including negation; crisis-trigger recall on a hand-written set.
- Inference API: input validation and error handling.
- App: Streamlit `AppTest` smoke tests for each page.

**Definition of Done**
- [ ] `data_inventory.csv` lists every dataset used with source, date, counts, terms and `real|synthetic`.
- [ ] All 10 notebooks execute end to end; metrics JSON/figures exist.
- [ ] `make setup data train notebooks test app` works from a clean clone (synthetic fallback only where a real dataset truly can't be obtained).
- [ ] App runs locally and in Docker; Lab Results page shows every experiment and the ranked dataset table.
- [ ] Every model has a model card with subgroup metrics and limitations.
- [ ] Explanations (SHAP/LIME/Grad-CAM) available for each modality.
- [ ] Crisis flow tested manually and by unit tests.
- [ ] README, REPORT, PRESENTATION, DEMO_SCRIPT complete; no placeholders (`TODO`, `lorem ipsum`).
- [ ] Deployment instructions verified; exact commands and checklist provided for the manual parts (Kaggle token, HF Space creation).
- [ ] Final summary of **actual** results in `reports/metrics/summary.json` and the README.

---

## 13. Build Order (milestones)

1. **Scaffold:** layout, configs, Makefile, requirements, CI skeleton, `DECISIONS.md`.
2. **Data (Exp 1):** `download_data.py` with the Section 5.6 ladder, inventory, cleaning, harmonization, DASS scoring, processed data, dictionary, quality report.
3. **EDA (Exp 2).**
4. **Core tabular model (Exp 6)** → **prognosis (Exp 4)**, including DASS item reduction.
5. **Text: classifier + Dreaddit external check + drug-review analytics (Exp 7)**, **entity extraction (Exp 5)**, **crisis module**.
6. **Image module (Exp 3)** with ONNX export.
7. **Explainability & fairness (Exp 8)** across all models.
8. **Inference package** + unit tests.
9. **Streamlit app (Exp 9)**, page by page, using only `inference.py`.
10. **Docker + deployment docs;** build and smoke-test the container.
11. **Documentation & presentation (Exp 10);** consolidate results, capture screenshots.
12. **Final QA:** fresh-clone run, lint, tests, Definition of Done, handoff summary.

---

## 14. Honesty & Quality Rules (final reminders)

- Report **real** metrics only. If performance is modest, say so and analyse why; do not tune on the test set.
- Distinguish **real** and **synthetic** data in every artifact. Do not use synthetic data where a real dataset is obtainable.
- Respect the **Do NOT use** list (Section 5.3) to avoid leakage.
- Do not copy copyrighted text or datasets into the repo; reference sources and licences.
- Cite only references you are confident exist; mark uncertain ones for the user to verify.
- Keep the code understandable to a student: explain *why* in comments and notebook markdown so the author can defend the project in a viva.
- When finished, output a concise **handoff summary**: what exists, how to run it, real results table, deviations from this spec, and remaining manual steps for the user.