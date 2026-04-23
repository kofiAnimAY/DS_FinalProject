"""
Shared data loading and preprocessing utilities.
Provides cached dataset loading for all pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.datasets import (
#     fetch_california_housing,
#     load_wine,
#     load_diabetes,
# )


# ── Available datasets ──────────────────────────────────────────────
DATASETS = {
    
    "📧 Marketing Campaign": "marketing",
}

DATASET_DESCRIPTIONS = {
    
    "marketing": {
        "title": "Marketing Campaign Response",
        "problem": (
"<style>"
".biz-wrap{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1F2937;line-height:1.6;}"
".biz-wrap a{color:#4338CA;text-decoration:none;border-bottom:1px dotted #A5B4FC;}"
".biz-wrap a:hover{border-bottom-color:#4338CA;}"
".biz-wrap abbr{text-decoration:underline dotted;cursor:help;font-style:normal;}"
".biz-intro{background:#EEF2FF;border:1px solid #C7D2FE;border-radius:14px;padding:20px 24px;margin-bottom:20px;display:flex;gap:18px;align-items:flex-start;}"
".biz-intro-icon{font-size:2rem;line-height:1;flex-shrink:0;}"
".biz-intro h3{margin:0 0 4px 0;color:#312E81;font-size:1.15rem;font-weight:700;}"
".biz-intro p{margin:0;color:#4B5563;font-size:0.95rem;}"
".biz-section-label{font-size:0.72rem;text-transform:uppercase;letter-spacing:1.2px;color:#6366F1;font-weight:700;margin:22px 0 8px 0;}"
".biz-problem-box{background:white;border:1px solid #E5E7EB;border-radius:12px;padding:16px 20px;margin-bottom:18px;}"
".biz-problem-box p{margin:0;color:#374151;}"
".stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:18px;}"
".stat-card{background:white;border:1px solid #E0E7FF;border-left:4px solid #4338CA;border-radius:10px;padding:14px 16px;transition:box-shadow 0.15s,transform 0.15s;}"
".stat-card:hover{box-shadow:0 6px 16px rgba(67,56,202,0.10);transform:translateY(-2px);}"
".stat-figure{font-size:1.4rem;font-weight:800;color:#4338CA;line-height:1.1;margin-bottom:2px;}"
".stat-text{font-size:0.85rem;color:#374151;margin-bottom:6px;}"
".stat-source{font-size:0.72rem;color:#6B7280;}"
".biz-question{background:linear-gradient(135deg,#4338CA 0%,#6366F1 100%);color:white;border-radius:12px;padding:20px 24px;margin-top:6px;box-shadow:0 8px 20px rgba(67,56,202,0.18);}"
".biz-question-label{font-size:0.7rem;text-transform:uppercase;letter-spacing:1.2px;opacity:0.8;margin-bottom:6px;font-weight:700;}"
".biz-question p{margin:0;font-size:1.02rem;font-style:italic;line-height:1.55;}"
"</style>"
'<div class="biz-wrap">'
'<div class="biz-intro"><div class="biz-intro-icon">🎯</div><div>'
'<h3>Predicting Customer Response to Marketing Campaigns</h3>'
'<p>2,240 customers · 29 features · demographics, spend, channel behavior, past campaigns</p>'
'</div></div>'
'<div class="biz-section-label">The Problem</div>'
'<div class="biz-problem-box"><p>Retail and <abbr title="Consumer Packaged Goods: everyday items consumers use regularly and often replenish — food &amp; beverages, cosmetics, cleaning products (Investopedia)">CPG</abbr> companies waste enormous sums blasting promotions to customers who will never convert. The dataset (<code>marketing_campaign.csv</code>) captures demographics, household composition, 2-year spend across 6 product categories, channel behavior (web / catalog / store), and outcomes of 5 past campaigns plus a final <strong>Response</strong> campaign.</p></div>'
'<div class="biz-section-label">Why It Matters</div>'
'<div class="stat-grid">'
'<div class="stat-card"><div class="stat-figure">2–5%</div><div class="stat-text">Average direct-mail response rate — 95%+ of spend reaches non-buyers.</div><div class="stat-source"><a href="https://www.mailpro.org/post/direct-mail-response-rates/" target="_blank">Mailpro</a> · <a href="https://www.ana.net/miccontent/show/id/rr-2025-07-response-rate-report" target="_blank">ANA</a></div></div>'
'<div class="stat-card"><div class="stat-figure">+60%</div><div class="stat-text">Rise in customer acquisition cost over 5 years across retail — retention beats broad acquisition.</div><div class="stat-source"><a href="https://www.simplicitydx.com/blogs/customer-acquisition-crisis" target="_blank">SimplicityDX, 2023</a></div></div>'
'<div class="stat-card"><div class="stat-figure">5–8×</div><div class="stat-text">ROI lift from personalized targeting vs. mass campaigns.</div><div class="stat-source"><a href="https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/the-value-of-getting-personalization-right-or-wrong-is-multiplying" target="_blank">McKinsey</a></div></div>'
'<div class="stat-card"><div class="stat-figure">80 / 20</div><div class="stat-text">Pareto reality — ~80% of revenue comes from ~20% of customers. Identifying them is the highest-leverage decision.</div><div class="stat-source"><a href="https://www.salesforce.com/ap/blog/80-20-rule/" target="_blank">Salesforce</a></div></div>'
'</div>'
'<div class="biz-question"><div class="biz-question-label">💡 Business question</div><p>Which customers should we target in the next campaign to maximize response rate while minimizing wasted contact cost — and how do we restrategise our approach toward the groups least likely to respond to our current methods?</p></div>'
'</div>'
        ),
        "target": "Response",
        "target_desc": "Response to last campaign (1=yes, 0=no)",
        "source": "Marketing Campaign Dataset",
        "rows": "2,240 customers",
        "features_desc": {
            "Year_Birth": "Year of birth",
            "Education": "Education level",
            "Marital_Status": "Marital status",
            "Income": "Annual household income",
            "Kidhome": "Number of children",
            "Teenhome": "Number of teenagers",
            "Dt_Customer": "Date became customer",
            "Recency": "Days since last purchase",
            "MntWines": "Total $ spent on wine (last 2 years)",
            "MntFruits": "Total $ spent on fruits (last 2 years)",
            "MntMeatProducts": "Total $ spent on meat products (last 2 years)",
            "MntFishProducts": "Total $ spent on fish products (last 2 years)",
            "MntSweetProducts": "Total $ spent on sweet products (last 2 years)",
            "MntGoldProds": "Total $ spent on gold/luxury products (last 2 years)",
            "NumDealsPurchases": "Number of purchases with discount",
            "NumWebPurchases": "Number of web purchases",
            "NumCatalogPurchases": "Number of catalog purchases",
            "NumStorePurchases": "Number of store purchases",
            "NumWebVisitsMonth": "Number of web visits per month",
            "AcceptedCmp3": "Accepted campaign 3 (1=yes, 0=no)",
            "AcceptedCmp4": "Accepted campaign 4 (1=yes, 0=no)",
            "AcceptedCmp5": "Accepted campaign 5 (1=yes, 0=no)",
            "AcceptedCmp1": "Accepted campaign 1 (1=yes, 0=no)",
            "AcceptedCmp2": "Accepted campaign 2 (1=yes, 0=no)",
            "Complain": "Number of complaints",
            "Z_CostContact": "Cost to contact customer",
            "Z_Revenue": "Revenue from customer",
        },
    },
}


@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    df= pd.read_csv("marketing_campaign.csv", sep='\t')
    return df


@st.cache_data
def preprocess(_df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready DataFrame from the raw marketing campaign data.

    Transformations applied:
    - Year_Birth → Age
    - Dt_Customer → Tenure_Days (days since enrollment)
    - Income: missing values filled with median
    - Education: ordinal encoded (Basic=0 … PhD=3)
    - Marital_Status: binary (together=1, alone=0)
    - Drops ID, Z_CostContact, Z_Revenue
    """
    d = _df.copy()

    # Age from birth year
    d["Age"] = 2025 - d["Year_Birth"]
    d.drop(columns=["Year_Birth"], inplace=True)

    # Customer tenure in days
    d["Dt_Customer"] = pd.to_datetime(d["Dt_Customer"], dayfirst=True, errors="coerce")
    ref_date = d["Dt_Customer"].max()
    d["Tenure_Days"] = (ref_date - d["Dt_Customer"]).dt.days
    d.drop(columns=["Dt_Customer"], inplace=True)

    # Fill missing income
    d["Income"] = d["Income"].fillna(d["Income"].median())

    # Ordinal-encode Education
    edu_order = {"Basic": 0, "Graduation": 1, "2n Cycle": 1, "Master": 2, "PhD": 3}
    d["Education"] = d["Education"].map(edu_order).fillna(1).astype(int)

    # Binary-encode Marital_Status (1 = partnered, 0 = single/alone)
    partnered = {"Married", "Together"}
    d["Marital_Status"] = d["Marital_Status"].apply(lambda x: 1 if x in partnered else 0)

    # Drop non-informative columns
    d.drop(columns=["ID", "Z_CostContact", "Z_Revenue"], inplace=True, errors="ignore")

    return d


def get_target(dataset_key: str) -> str:
    return DATASET_DESCRIPTIONS[dataset_key]["target"]


def get_features(df: pd.DataFrame, target: str) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if c != target]


def categorical_chart_kind(s: pd.Series, max_unique_int: int = 5) -> str:
    """Classify a variable by its best visualization kind:
    - 'pie' for binary (2 levels) — best for yes/no proportions
    - 'bar' for small-count integers and text categoricals — preserves ordering
    - 'continuous' for true numeric variables — histogram / box / violin
    """
    if not pd.api.types.is_numeric_dtype(s):
        return "bar"
    vals = s.dropna().unique()
    if len(vals) == 2:
        return "pie"
    if len(vals) <= max_unique_int:
        try:
            if all(float(v).is_integer() for v in vals):
                return "bar"
        except (TypeError, ValueError):
            pass
    return "continuous"


def categorical_labels(values):
    """Produce friendly labels: 0/1 binary → 'No (0)' / 'Yes (1)', else str(v)."""
    vals = list(values)
    if set(vals) == {0, 1}:
        return ["No (0)" if v == 0 else "Yes (1)" for v in vals]
    return [str(v) for v in vals]


def dataset_selector() -> tuple[str, pd.DataFrame, dict]:
    """Render a dataset selector in the sidebar and return (key, df, info)."""
    with st.sidebar:
        st.markdown("### 📂 Dataset")
        choice = st.selectbox(
            "Choose a dataset",
            list(DATASETS.keys()),
            label_visibility="collapsed",
        )
    key = DATASETS[choice]
    df = load_data(key)
    info = DATASET_DESCRIPTIONS[key]
    return key, df, info
