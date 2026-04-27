# DS4E Final Project — Marketing Campaign Response Predictor 🎯

**Course:** DS-UA 9111 — Data Science for Everyone @ NYU
**Professor:** Gaëtan Brison

A full end-to-end machine-learning Streamlit app that predicts which customers
will respond to a marketing campaign — and translates the model into a tiered
targeting strategy with a special focus on **how to re-strategise toward the
customers our current methods cannot reach**.

---

## What this project does

**Business problem.** Retail and CPG companies waste enormous sums blasting
promotions to customers who will never convert. Industry direct-mail response
rates sit around 2–5%; CAC has risen ~60% in 5 years; personalised targeting
delivers 5–8× ROI lift over mass campaigns.

**Our question.** *Which customers should we target in the next campaign to
maximise response while minimising wasted contact cost — and how do we
re-strategise our approach toward the groups least likely to respond to our
current methods?*

The app trains 5 classifiers on a 2,240-customer marketing dataset, lets you
interactively threshold predictions to simulate a campaign rollout, and
summarises everything into a tiered action plan (Tier A: contact directly,
Tier B: nurture cheaply, Tier C: try fundamentally different methods).

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

### Optional — W&B experiment tracking

Set `WANDB_API_KEY` to log every training and tuning run to your W&B project.
The Tuning page also gains a "Past Experiments (W&B)" tab that browses past
runs from the project.

```bash
export WANDB_API_KEY=your-key-here       # get one at https://wandb.ai/authorize
export WANDB_PROJECT=ds4e-final-project  # optional, default shown
streamlit run app.py
```

See `.env.example` for full options. A local `.env` file is also auto-loaded
if present (and gitignored).

### Optional — precompute SHAP importances

The Explainability page computes feature importance live by default. To
pre-compute and cache for faster page loads:

```bash
python precompute_importance.py
```

This writes pickles under `cache/` for each (dataset × model). The
Explainability page checks the cache first, then falls back to live compute.

---

## App pages

| Page | Description |
|------|-------------|
| 🏠 **Business Case & Data** | Problem statement, dataset overview, descriptive stats, data-quality check |
| 📊 **Data Visualization** | Target distribution, feature distributions (continuous / ordinal / binary tabs), correlation heatmap, multicollinearity check (VIF), feature-vs-target scatter, box plots, full interactive exploration report (overview / variable explorer / method-selectable correlation matrix / outlier analysis) |
| 🤖 **Model Prediction** | Train & compare 5 classifiers (with optional SMOTE oversampling via imblearn Pipeline); ROC + PR curves; threshold simulator with auto-suggested optimal thresholds (max F1 / max profit) and TP / FP / lift / net profit cards; metric-interpretation guide |
| 🔍 **Explainability (SHAP)** | Top drivers hero panel; tree importance / permutation importance / SHAP values / per-customer waterfall tabs; cross-check against Logistic Regression coefficients |
| ⚙️ **Hyperparameter Tuning** | Optuna Bayesian search with selectable scoring (PR AUC default, ROC AUC option) on stratified CV; W&B-backed past-experiment leaderboard |
| 📊 **Conclusions & Recommendations** | Live findings, modelling outcomes, research benchmarks, Tier A/B/C action plan, Tier C measurement framework, methodology, limitations |

---

## Models

This is a **binary classification** problem (`Response` ∈ {0, 1}, ~15% positive).

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline, fully interpretable coefficients |
| Decision Tree | Single-tree interpretable non-linear |
| Random Forest | Bagged ensemble — typically a top performer |
| Gradient Boosting | Boosted ensemble — typically a top performer |
| MLP (Neural Network) | Non-linear, included for comparison; tends to underperform on small tabular data |

All five use `class_weight="balanced"` where supported and stratified
train/test/CV splits. The headline metric is **PR AUC** (right choice for
~15% positive imbalance), with ROC AUC and F1 as secondary checks.

---

## Dataset

**Source.** `marketing_campaign.csv` — 2,240 customers × 29 raw features
covering demographics, household composition, two-year spend across six
product categories, channel behaviour (web / catalog / store), and outcomes
of five past campaigns plus the final `Response` campaign.

**Preprocessing.** `data_loader.preprocess()` engineers `Age`, `Tenure_Days`,
`TotalSpend`, `TotalPurchases`, `TotalAccepted`, `HasChildren`; ordinal-encodes
`Education`; binary-encodes `Marital_Status`; median-imputes and 99th-pct-caps
`Income`; drops zero-correlation columns (`Complain`, `NumDealsPurchases`,
`NumWebVisitsMonth`) and non-feature columns (`ID`, `Z_*`).

The Data Visualisation page shows the **raw** data; the modelling pages
show the **preprocessed** features (each has a "📌 What's preprocessed?"
expander explaining the transform).

---

## Tech stack

| Area | Tools |
|---|---|
| App | Streamlit |
| Visualisation | Plotly |
| ML | scikit-learn + imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer for ensembles; raw coefficients for Logistic) |
| Hyperparameter tuning | Optuna (Bayesian / TPE) |
| Experiment tracking | Weights & Biases |

---

## Project structure

```
.
├── app.py                       # Streamlit entry point + page nav + theme CSS
├── data_loader.py               # CSV loading, preprocessing, shared helpers (VIF, callouts)
├── precompute_importance.py     # CLI: cache SHAP importances for the Explainability page
├── marketing_campaign.csv       # Dataset
├── requirements.txt
├── .env.example                 # W&B configuration template
├── src/
│   ├── page_intro.py            # 🏠 Business Case & Data
│   ├── page_visualization.py    # 📊 Data Visualization (full EDA)
│   ├── page_prediction.py       # 🤖 5-classifier comparison + threshold widget
│   ├── page_explainability.py   # 🔍 SHAP + permutation + Logistic cross-check
│   ├── page_tuning.py           # ⚙️ Optuna + W&B past-experiment browser
│   ├── page_conclusions.py      # 📊 Methodology / limitations / Tier A-B-C plan
│   └── wandb_tracker.py         # W&B init / log / fetch_past_runs helpers
└── cache/                       # SHAP importance pickles (created on demand)
```

---

## License

See `LICENSE`.
