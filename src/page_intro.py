"""
Page 1 — Business Case & Data Presentation
============================================
Presents the problem statement, dataset overview,
descriptive statistics, and data quality checks.
The interactive data exploration report lives on the
Data Visualization page.
"""

import streamlit as st
import pandas as pd

from data_loader import dataset_selector, get_target, get_features


def render():
    # ── Dataset selection ───────────────────────────────────────────
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    # ── Hero banner ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner">
        <h1>📈 {info["title"]}</h1>
        <p>DS4EVERYONE @ NYU — Final Project Demo App</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Business problem ────────────────────────────────────────────
    st.markdown("## 🎯 Business Problem")
    st.markdown(info["problem"], unsafe_allow_html=True)
    st.markdown("---")

    # ── Key metrics ─────────────────────────────────────────────────
    st.markdown("## 📋 Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Rows", f"{len(df):,}"),
        (c2, "Features", str(len(features))),
        (c3, "Target", info["target"]),
        (c4, "Source", info["source"]),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p style="font-size:1.3rem;word-break:break-word;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Data preview ────────────────────────────────────────────────
    tab_head, tab_tail, tab_sample = st.tabs(["First rows", "Last rows", "Random sample"])
    with tab_head:
        st.dataframe(df.head(10), width='stretch')
    with tab_tail:
        st.dataframe(df.tail(10), width='stretch')
    with tab_sample:
        st.dataframe(df.sample(10, random_state=42), width='stretch')

    st.markdown("---")

    # ── Descriptive statistics ──────────────────────────────────────
    st.markdown("## 📊 Descriptive Statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), width='stretch')

    # ── Data quality ────────────────────────────────────────────────
    st.markdown("## ✅ Data Quality Check")
    col_a, col_b = st.columns(2)
    with col_a:
        missing = df.isnull().sum()
        miss_pct = (missing / len(df) * 100).round(2)
        quality_df = pd.DataFrame({"Missing": missing, "% Missing": miss_pct})
        st.dataframe(quality_df, width='stretch')
    with col_b:
        completeness = (1 - df.isnull().mean().mean()) * 100
        duplicates = df.duplicated().sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Completeness</h3>
            <p>{completeness:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Duplicate Rows</h3>
            <p>{duplicates}</p>
        </div>
        <div class="metric-card">
            <h3>Memory Usage</h3>
            <p>{df.memory_usage(deep=True).sum() / 1024:.0f} KB</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "📊 Continue to the **Data Visualization** page for the full interactive "
        "exploration report — distributions, correlations, multicollinearity check, "
        "and outlier analysis."
    )
