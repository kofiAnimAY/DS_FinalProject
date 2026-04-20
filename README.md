# DS4E Final Project — ML Prediction App 🎓

**Course:** DS-UA 9111 — Data Science for Everyone @ NYU  
**Professor:** Gaëtan Brison

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## App Structure

| Page | Description |
|------|-------------|
| 🏠 Business Case & Data | Problem statement, dataset overview, descriptive stats, data quality |
| 📊 Data Visualization | Interactive Plotly charts: distributions, correlations, scatter explorer |
| 🤖 Model Prediction | Train & compare 7 models (Linear, Ridge, Lasso, ElasticNet, DT, RF, GB) |
| 🔍 Explainability | SHAP values, permutation importance, feature importance analysis |
| ⚙️ Hyperparameter Tuning | Automated Optuna optimization with experiment tracking |

## Datasets

- **California Housing** — Predict median house values
- **Wine Quality** — Predict wine quality scores
- **Diabetes Progression** — Predict disease progression

## Technologies

- **App:** Streamlit
- **Visualization:** Plotly, Seaborn
- **ML:** Scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **Tuning:** Optuna
