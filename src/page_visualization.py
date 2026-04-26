"""
Page 2 — Data Visualization
=============================
Interactive charts exploring distributions, correlations,
and relationships in the dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import (
    dataset_selector, get_target, get_features,
    categorical_chart_kind, categorical_labels,
    preprocess, compute_vif,
)


INDIGO_DISCRETE = ["#4338CA", "#6366F1", "#A5B4FC", "#C7D2FE", "#312E81", "#818CF8"]


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 📊 Data Visualization")
    st.caption("Explore the dataset through interactive charts to uncover patterns and insights.")
    st.markdown("---")

    # ── 1. Target distribution ──────────────────────────────────────
    st.markdown("### 🎯 Target Variable Distribution")
    target_s = df[target].dropna()
    target_kind = categorical_chart_kind(target_s)
    col1, col2 = st.columns([2, 1])
    with col1:
        if target_kind == "pie":
            counts = target_s.value_counts().sort_index()
            labels = categorical_labels(counts.index)
            fig = px.pie(
                names=labels, values=counts.values,
                color_discrete_sequence=INDIGO_DISCRETE,
                title=f"Class balance for {target}",
                hole=0.45,
            )
            fig.update_traces(
                textposition="inside", textinfo="percent+label",
                sort=False, textfont_size=16,
            )
            fig.update_layout(
                template="plotly_white", height=560,
                margin=dict(l=10, r=10, t=60, b=10),
            )
        elif target_kind == "bar":
            counts = target_s.value_counts().sort_index()
            fig = px.bar(
                x=[str(v) for v in counts.index], y=counts.values,
                color_discrete_sequence=["#4338CA"],
                title=f"Counts of {target}",
            )
            fig.update_layout(
                template="plotly_white", showlegend=False, height=380,
                xaxis_title=info["target_desc"], yaxis_title="Count",
            )
        else:
            fig = px.histogram(
                df, x=target, nbins=50, color_discrete_sequence=["#4338CA"],
                title=f"Distribution of {target}",
            )
            fig.update_layout(
                template="plotly_white",
                xaxis_title=info["target_desc"],
                yaxis_title="Count",
            )
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("")
        st.markdown("")
        if target_kind in ("pie", "bar"):
            counts = target_s.value_counts().sort_index()
            top = counts.idxmax()
            top_label = categorical_labels([top])[0]
            card_items = [
                ("Rows", f"{len(target_s):,}"),
                ("Levels", f"{int(target_s.nunique())}"),
                ("Most Common", top_label),
                ("Top Share", f"{counts.max() / counts.sum() * 100:.1f}%"),
            ]
            if target_kind == "pie" and set(target_s.unique()) == {0, 1}:
                card_items.insert(3, ("Positive Rate", f"{target_s.mean() * 100:.1f}%"))
            for label, value in card_items:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <p>{value}</p>
                </div>""", unsafe_allow_html=True)
        else:
            stats = target_s.describe()
            for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{stat_name.upper()}</h3>
                    <p>{stats[stat_name]:.2f}</p>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Feature distributions ────────────────────────────────────
    st.markdown("### 📈 Feature Distributions")

    feature_kinds = {f: categorical_chart_kind(df[f]) for f in features}
    continuous_feats = [f for f, k in feature_kinds.items() if k == "continuous"]
    bar_feats = [f for f, k in feature_kinds.items() if k == "bar"]
    pie_feats = [f for f, k in feature_kinds.items() if k == "pie"]

    dist_tab_cont, dist_tab_bar, dist_tab_pie = st.tabs([
        f"📊 Continuous ({len(continuous_feats)})",
        f"📶 Small-count / ordinal ({len(bar_feats)})",
        f"🥧 Binary flags ({len(pie_feats)})",
    ])

    with dist_tab_cont:
        if continuous_feats:
            selected_features = st.multiselect(
                "Select continuous features",
                continuous_feats,
                default=continuous_feats[:4],
                key="dist_continuous",
            )
            if selected_features:
                n_cols = min(len(selected_features), 3)
                n_rows = (len(selected_features) + n_cols - 1) // n_cols
                fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=selected_features)
                colors = px.colors.sequential.Blues_r
                for i, feat in enumerate(selected_features):
                    r, c = divmod(i, n_cols)
                    fig.add_trace(
                        go.Histogram(
                            x=df[feat], name=feat, nbinsx=30,
                            marker_color=colors[i % len(colors)],
                            showlegend=False,
                        ),
                        row=r + 1, col=c + 1,
                    )
                fig.update_layout(height=300 * n_rows, template="plotly_white")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No continuous features in this dataset.")

    with dist_tab_bar:
        if bar_feats:
            selected_bars = st.multiselect(
                "Select count / ordinal features",
                bar_feats,
                default=bar_feats[:min(4, len(bar_feats))],
                key="dist_bars",
            )
            if selected_bars:
                n_cols = min(len(selected_bars), 3)
                n_rows = (len(selected_bars) + n_cols - 1) // n_cols
                fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=selected_bars)
                for i, feat in enumerate(selected_bars):
                    r, c = divmod(i, n_cols)
                    counts = df[feat].dropna().value_counts().sort_index()
                    fig.add_trace(
                        go.Bar(
                            x=[str(v) for v in counts.index], y=counts.values,
                            marker_color="#4338CA", showlegend=False, name=feat,
                        ),
                        row=r + 1, col=c + 1,
                    )
                fig.update_layout(height=300 * n_rows, template="plotly_white")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No small-count or ordinal features in this dataset.")

    with dist_tab_pie:
        if pie_feats:
            selected_pies = st.multiselect(
                "Select binary features",
                pie_feats,
                default=pie_feats[: min(4, len(pie_feats))],
                key="dist_pies",
            )
            if selected_pies:
                n_cols = min(len(selected_pies), 3)
                n_rows = (len(selected_pies) + n_cols - 1) // n_cols
                specs = [[{"type": "pie"} for _ in range(n_cols)] for _ in range(n_rows)]
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    specs=specs, subplot_titles=selected_pies,
                )
                for i, feat in enumerate(selected_pies):
                    r, c = divmod(i, n_cols)
                    counts = df[feat].dropna().value_counts().sort_index()
                    labels = categorical_labels(counts.index)
                    fig.add_trace(
                        go.Pie(
                            labels=labels, values=counts.values,
                            hole=0.45, sort=False,
                            marker=dict(colors=INDIGO_DISCRETE),
                            textinfo="percent+label",
                        ),
                        row=r + 1, col=c + 1,
                    )
                fig.update_layout(height=320 * n_rows, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No binary features in this dataset.")

    st.markdown("---")

    # ── 3. Correlation heatmap ──────────────────────────────────────
    st.markdown("### 🔥 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, width='stretch')

    # ── Top correlations with target ────────────────────────────────
    target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
    st.markdown(f"**Top features correlated with `{target}`:**")
    for feat, val in target_corr.head(5).items():
        direction = "+" if corr.loc[feat, target] > 0 else "−"
        bar_width = int(val * 100)
        st.markdown(
            f"- **{feat}** → {direction}{val:.3f} "
            f'<span style="display:inline-block;height:10px;width:{bar_width}px;'
            f'background:#4338CA;border-radius:4px;"></span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 3b. Multicollinearity check (on modeling features) ─────────
    st.markdown("### 🧬 Multicollinearity Check — Modeling Features")
    st.caption(
        "Some of our engineered features (`TotalSpend`, `TotalAccepted`, "
        "`TotalPurchases`) are sums of their components, so they're collinear "
        "by construction. Tree models exploit the granularity; regularized "
        "Logistic Regression handles the redundancy via L2. This panel makes "
        "the issue visible rather than silent."
    )
    with st.expander("📖 Why this matters"):
        st.markdown(
            "When two features carry the same information, linear models "
            "split the coefficient between them arbitrarily and can produce "
            "**counterintuitive signs** (e.g. negative coefficient on a "
            "clearly positive predictor). VIF (Variance Inflation Factor) is "
            "the standard diagnostic:\n\n"
            "- **VIF ≈ 1** — no collinearity\n"
            "- **VIF 1–5** — moderate, usually fine\n"
            "- **VIF 5–10** — high — investigate\n"
            "- **VIF > 10** — severe — feature is a near-linear combination "
            "of others; coefficient estimates are unreliable\n\n"
            "Tree-based models are unaffected by VIF, but the interpretation "
            "story still benefits from knowing which features are duplicates."
        )

    df_pp = preprocess(df)
    pp_features = [c for c in df_pp.select_dtypes(include="number").columns
                   if c != target]

    mc_col1, mc_col2 = st.columns([1, 1])
    with mc_col1:
        st.markdown("**High-correlation pairs** (|r| ≥ 0.8)")
        corr_pp = df_pp[pp_features].corr().abs()
        # Upper triangle only, exclude diagonal
        pairs = []
        for i, f1 in enumerate(pp_features):
            for f2 in pp_features[i + 1:]:
                r = corr_pp.loc[f1, f2]
                if r >= 0.8:
                    pairs.append({"Feature A": f1, "Feature B": f2, "|r|": r})
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("|r|", ascending=False)
            st.dataframe(
                pairs_df.style.format({"|r|": "{:.3f}"})
                .background_gradient(subset=["|r|"], cmap="Reds"),
                width='stretch', hide_index=True,
            )
        else:
            st.success("✅ No pairs above |r| = 0.8 — collinearity is mild.")

    with mc_col2:
        st.markdown("**Variance Inflation Factor (VIF)**")
        vif_df = compute_vif(df_pp, pp_features)
        # Cap inf for display
        vif_display = vif_df.copy()
        vif_display["VIF"] = vif_display["VIF"].replace(float("inf"), 999.0)
        st.dataframe(
            vif_display.style.format({"VIF": "{:.2f}"})
            .background_gradient(subset=["VIF"], cmap="Reds", vmin=1, vmax=10),
            width='stretch', hide_index=True, height=400,
        )
        n_high = int((vif_df["VIF"] >= 10).sum())
        if n_high > 0:
            st.caption(f"⚠️ {n_high} feature(s) with VIF ≥ 10 — expected here, "
                       "since `Total*` engineered features are sums of their components.")

    st.markdown("---")

    # ── 4. Scatter plot explorer ────────────────────────────────────
    st.markdown("### 🔗 Feature vs Target Explorer")
    col_a, col_b = st.columns(2)
    with col_a:
        x_feat = st.selectbox("X-axis feature", features, index=0)
    with col_b:
        color_feat = st.selectbox(
            "Color by (optional)", ["None"] + features, index=0
        )

    color = color_feat if color_feat != "None" else None
    fig = px.scatter(
        df, x=x_feat, y=target, color=color,
        color_continuous_scale="Blues",
        opacity=0.5, title=f"{x_feat} vs {target}",
    )
    fig.update_layout(template="plotly_white", height=500)
    # Force integer ticks when either axis is an integer-valued variable
    # (e.g. Kidhome, Teenhome, AcceptedCmp, Complain, Response)
    if categorical_chart_kind(df[x_feat]) in ("pie", "bar"):
        fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, tickformat="d")
    if categorical_chart_kind(df[target]) in ("pie", "bar"):
        fig.update_yaxes(tickmode="linear", tick0=0, dtick=1, tickformat="d")
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── 5. Box plots ────────────────────────────────────────────────
    st.markdown("### 📦 Box Plots — Outlier Detection")
    st.caption("Box plots only make sense for continuous variables — binary flags and small-count integers are excluded.")
    if continuous_feats:
        box_feats = st.multiselect(
            "Features for box plots", continuous_feats,
            default=continuous_feats[: min(3, len(continuous_feats))], key="box",
        )
        if box_feats:
            fig = go.Figure()
            for feat in box_feats:
                fig.add_trace(go.Box(y=df[feat], name=feat, marker_color="#4338CA"))
            fig.update_layout(template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No continuous features available for box plots.")
