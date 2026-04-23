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
        "problem": """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.7;color:#fff;">

<h3 style="color:#57068C;margin-top:0;">🎯 Business Case: Predicting Customer Response to Marketing Campaigns</h3>

<h4 style="color:#fff;margin-bottom:4px;">The Problem</h4>
<p>Retail and <abbr title="Consumer Packaged Goods (CPG): everyday items that consumers use regularly and often replenish — e.g. food &amp; beverages, cosmetics, and cleaning products (Investopedia)" style="text-decoration:underline dotted;cursor:help;">CPG</abbr> companies waste enormous sums blasting promotions to customers who will never convert.
The dataset (<code>marketing_campaign.csv</code>, 2,240 customers × 29 features) captures demographics,
household composition, 2-year spend across 6 product categories, channel behavior (web/catalog/store),
and outcomes of 5 past campaigns plus a final <strong>Response</strong> campaign.</p>

<h4 style="color:#fff;margin-bottom:4px;">Why It Matters</h4>
<ul style="padding-left:20px;">
  <li><strong>Marketing waste is massive.</strong> Average direct-mail response rates sit around 2–5%
  (<a href="https://www.mailpro.org/post/direct-mail-response-rates/" target="_blank">Mailpro</a> /
  <a href="https://www.ana.net/miccontent/show/id/rr-2025-07-response-rate-report" target="_blank">ANA benchmarks</a>),
  meaning 95%+ of spend reaches non-buyers.</li>

  <li><strong>Customer acquisition cost has risen ~60%</strong> over the last 5 years across retail
  (<a href="https://www.simplicitydx.com/blogs/customer-acquisition-crisis" target="_blank">SimplicityDX, 2023</a>),
  making retention and targeted reactivation more profitable than broad acquisition.</li>

  <li><strong>Personalized targeting lifts ROI 5–8×</strong> vs. mass campaigns
  (<a href="https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/the-value-of-getting-personalization-right-or-wrong-is-multiplying" target="_blank">McKinsey, "The value of getting personalization right"</a>).</li>

  <li><strong>Pareto reality:</strong> ~80% of revenue typically comes from ~20% of customers — identifying
  who in that tail will respond is the single highest-leverage marketing decision
  (<a href="https://www.salesforce.com/ap/blog/80-20-rule/" target="_blank">Salesforce</a>).</li>
</ul>

<h4 style="color:#fff;margin-bottom:4px;">Business Question</h4>
<blockquote style="border-left:4px solid #57068C;margin:0;padding:10px 16px;background:#f8f4fc;border-radius:0 8px 8px 0;font-style:italic;color:#57068C;font-weight:600;">
  Which customers should we target in the next campaign to maximize response rate while minimizing wasted contact cost — and how do we restrategise our approach toward the groups least likely to respond to our current methods?
</blockquote>

</div>
""",
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
            "Age": "Customer age (derived from Year_Birth)",
            "Tenure_Days": "Days since customer enrollment (derived from Dt_Customer)",
        },
    },
}


@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    df = pd.read_csv("marketing_campaign.csv", sep="\t")
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
