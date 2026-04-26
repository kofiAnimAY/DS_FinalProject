"""
Shared data loading and preprocessing utilities.
Provides cached dataset loading for all pages.
"""

import streamlit as st
import pandas as pd


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
            "Age": "Customer age (capped at 100 to remove data-entry errors)",
            "Tenure_Days": "Days since enrollment (derived from Dt_Customer)",
            "Education": "Education level (ordinal: 0=Basic … 3=PhD)",
            "Marital_Status": "Partnered=1, single/alone=0",
            "Income": "Annual household income (outliers capped at 99th pct)",
            "Kidhome": "Number of children at home",
            "Teenhome": "Number of teenagers at home",
            "HasChildren": "Any child or teen at home (1=yes, 0=no)",
            "Recency": "Days since last purchase (lower = more recently active)",
            "MntWines": "$ spent on wine (last 2 years)",
            "MntFruits": "$ spent on fruits (last 2 years)",
            "MntMeatProducts": "$ spent on meat products (last 2 years)",
            "MntFishProducts": "$ spent on fish products (last 2 years)",
            "MntSweetProducts": "$ spent on sweet products (last 2 years)",
            "MntGoldProds": "$ spent on gold/luxury products (last 2 years)",
            "TotalSpend": "Total $ spent across all 6 categories — aggregate of Mnt* columns",
            "NumWebPurchases": "Number of web purchases",
            "NumCatalogPurchases": "Number of catalog purchases",
            "NumStorePurchases": "Number of store purchases",
            "TotalPurchases": "Total purchases across all channels — aggregate of Num*Purchases",
            "AcceptedCmp1": "Accepted campaign 1 (1=yes, 0=no) — corr 0.29 with Response",
            "AcceptedCmp2": "Accepted campaign 2 (1=yes, 0=no) — corr 0.17 with Response",
            "AcceptedCmp3": "Accepted campaign 3 (1=yes, 0=no) — corr 0.25 with Response",
            "AcceptedCmp4": "Accepted campaign 4 (1=yes, 0=no) — corr 0.18 with Response",
            "AcceptedCmp5": "Accepted campaign 5 (1=yes, 0=no) — corr 0.33 with Response",
            "TotalAccepted": "Sum of past 5 campaign acceptances — strongest predictor of Response",
        },
    },
}


@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    df= pd.read_csv("marketing_campaign.csv", sep='\t')
    return df


@st.cache_data
def preprocess(_df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready DataFrame from the raw marketing campaign data."""
    d = _df.copy()

    # Age from birth year — cap at 100 to remove data-entry errors (e.g. born 1893)
    d["Age"] = (2025 - d["Year_Birth"]).clip(upper=100)
    d.drop(columns=["Year_Birth"], inplace=True)

    # Customer tenure in days
    d["Dt_Customer"] = pd.to_datetime(d["Dt_Customer"], dayfirst=True, errors="coerce")
    ref_date = d["Dt_Customer"].max()
    d["Tenure_Days"] = (ref_date - d["Dt_Customer"]).dt.days
    d.drop(columns=["Dt_Customer"], inplace=True)

    # Fill missing income then cap extreme outliers at 99th percentile
    d["Income"] = d["Income"].fillna(d["Income"].median())
    d["Income"] = d["Income"].clip(upper=d["Income"].quantile(0.99))

    # Ordinal-encode Education
    edu_order = {"Basic": 0, "Graduation": 1, "2n Cycle": 1, "Master": 2, "PhD": 3}
    d["Education"] = d["Education"].map(edu_order).fillna(1).astype(int)

    # Binary-encode Marital_Status (1 = partnered, 0 = single/alone)
    partnered = {"Married", "Together"}
    d["Marital_Status"] = d["Marital_Status"].apply(lambda x: 1 if x in partnered else 0)

    # Engineered aggregates — reduces severe multicollinearity (r=0.35–0.72 among components)
    # and creates the strongest single predictors of Response
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                     "AcceptedCmp4", "AcceptedCmp5"]

    d["TotalSpend"] = d[spend_cols].sum(axis=1)
    d["TotalPurchases"] = d[purchase_cols].sum(axis=1)
    d["TotalAccepted"] = d[campaign_cols].sum(axis=1)
    d["HasChildren"] = ((d["Kidhome"] + d["Teenhome"]) > 0).astype(int)
    # Keep individual columns too — tree models use the granularity;
    # Logistic Regression handles the collinearity through L2 regularization

    # Drop zero-correlation features (correlation with Response ≈ 0.00)
    d.drop(columns=["Complain", "NumDealsPurchases", "NumWebVisitsMonth"],
           inplace=True, errors="ignore")

    # Drop the customer ID — Z_CostContact / Z_Revenue are already dropped in load_data()
    d.drop(columns=["ID"], inplace=True, errors="ignore")

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


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF interpretation:
    - 1.0 = no multicollinearity
    - 1 < VIF < 5 = moderate, usually fine
    - 5 < VIF < 10 = high — investigate
    - VIF > 10 = severe multicollinearity — feature is a near-linear combination
      of others, parameter estimates are unreliable
    """
    from sklearn.linear_model import LinearRegression

    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    rows = []
    for feat in features:
        y = X[feat].values
        X_other = X.drop(columns=[feat]).values
        if X_other.shape[1] == 0:
            rows.append({"Feature": feat, "VIF": float("nan")})
            continue
        try:
            r2 = LinearRegression().fit(X_other, y).score(X_other, y)
            vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf")
        except Exception:
            vif = float("nan")
        rows.append({"Feature": feat, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def preprocessing_callout() -> None:
    """Render an expander explaining the feature transformations.

    Used at the top of the prediction / tuning / explainability pages so the
    column names there make sense after intro / visualization showed raw data.
    """
    with st.expander("📌 What's preprocessed?", expanded=False):
        st.markdown(
            "Features below are the output of `preprocess()`, **not** the raw "
            "columns shown on the Business Case and Visualization pages:\n\n"
            "- **`Age`** — derived from `Year_Birth` (capped at 100)\n"
            "- **`Tenure_Days`** — derived from `Dt_Customer` (days as customer)\n"
            "- **`TotalSpend`** — sum of the 6 product spend columns\n"
            "- **`TotalPurchases`** — sum of web + catalog + store purchases\n"
            "- **`TotalAccepted`** — count of past accepted campaigns (0–5)\n"
            "- **`HasChildren`** — binary (1 if `Kidhome + Teenhome > 0`)\n"
            "- **`Education`** — ordinal-encoded (Basic = 0 … PhD = 3)\n"
            "- **`Marital_Status`** — binary (1 = partnered, 0 = single)\n"
            "- **`Income`** — median-imputed and 99th-percentile capped\n\n"
            "Dropped: `Complain`, `NumDealsPurchases`, `NumWebVisitsMonth` "
            "(zero-correlation with target), plus `ID` / `Z_CostContact` / `Z_Revenue` "
            "(non-features)."
        )


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
