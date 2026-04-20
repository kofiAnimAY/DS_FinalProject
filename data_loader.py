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
            "**Business Problem:** A retail company wants to predict customer response "
            "to marketing campaigns based on demographic and behavioral data to optimize "
            "campaign targeting and increase conversion rates."
        ),
        "target": "Response",
        "target_desc": "Response to last campaign (1=yes, 0=no)",
        "source": "Marketing Campaign Dataset",
        "rows": "2,240 customers",
        "features_desc": {
            "ID": "Customer ID",
            "Year_Birth": "Year of birth",
            "Education": "Education level",
            "Marital_Status": "Marital status",
            "Income": "Annual household income",
            "Kidhome": "Number of children",
            "Teenhome": "Number of teenagers",
            "Dt_Customer": "Date became customer",
            "Recency": "Days since last purchase",
            "MntWines": "Amount spent on wine",
            "MntFruits": "Amount spent on fruits",
            "MntMeatProducts": "Amount spent on meat",
            "MntFishProducts": "Amount spent on fish",
            "MntSweetProducts": "Amount spent on sweets",
            "MntGoldProds": "Amount spent on gold products",
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
