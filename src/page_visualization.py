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
EXCLUDE = {"ID"}


# ── Data exploration report helpers (moved from page_intro) ────────
def _pie_counts(series: pd.Series, title: str):
    counts = series.dropna().value_counts().sort_index()
    labels = categorical_labels(counts.index)
    fig = px.pie(
        names=labels, values=counts.values,
        color_discrete_sequence=INDIGO_DISCRETE,
        title=title, hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", sort=False)
    fig.update_layout(template="plotly_white", height=380)
    return fig


def _bar_counts(series: pd.Series, title: str, xlabel: str = None):
    counts = series.dropna().value_counts().sort_index()
    labels = [str(v) for v in counts.index]
    fig = px.bar(
        x=labels, y=counts.values,
        color_discrete_sequence=["#4338CA"],
        title=title,
    )
    fig.update_layout(
        template="plotly_white", showlegend=False, height=380,
        xaxis_title=xlabel or series.name, yaxis_title="Count",
    )
    return fig


def _outlier_summary(df, features, method="IQR", z_thresh=3.0, iqr_k=1.5):
    """Return a list of (feature, n_outliers, pct, lower, upper)."""
    results = []
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].dropna()
        if method == "IQR":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        else:
            mu, sigma = s.mean(), s.std()
            lower, upper = mu - z_thresh * sigma, mu + z_thresh * sigma
        mask = (df[col] < lower) | (df[col] > upper)
        n_out = int(mask.sum())
        pct = n_out / len(df) * 100
        results.append((col, n_out, pct, lower, upper))
    return results


def _overview_cards_html(df):
    n_rows, n_cols = df.shape
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    missing_total = df.isnull().sum().sum()
    missing_pct = (missing_total / (n_rows * n_cols)) * 100
    duplicates = df.duplicated().sum()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    return f"""
    <style>
        .report-card {{
            background: #EEF2FF;
            border: 1px solid #C7D2FE;
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            transition: transform 0.15s, box-shadow 0.15s;
        }}
        .report-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67,56,202,0.12);
        }}
        .report-card-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 4px;
        }}
        .report-card-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #4338CA;
        }}
    </style>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:8px;">
        <div class="report-card">
            <div class="report-card-label">Observations</div>
            <div class="report-card-value">{n_rows:,}</div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Variables</div>
            <div class="report-card-value">{n_cols}</div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Missing Cells</div>
            <div class="report-card-value" style="color:{'#2ea043' if missing_total == 0 else '#d73a49'};">{missing_total:,} <span style="font-size:0.75rem;">({missing_pct:.1f}%)</span></div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Duplicate Rows</div>
            <div class="report-card-value" style="color:{'#2ea043' if duplicates == 0 else '#d73a49'};">{duplicates:,}</div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Memory</div>
            <div class="report-card-value">{mem_kb:.0f} <span style="font-size:0.75rem;">KB</span></div>
        </div>
        <div class="report-card">
            <div class="report-card-label">Numeric</div>
            <div class="report-card-value">{len(numeric_cols)}</div>
        </div>
    </div>
    """


def _render_data_report(df, features, target, info):
    """Render an interactive data exploration report."""

    # ── Overview ───────────────────────────────────────────────────
    st.markdown("#### 📋 Overview")
    st.markdown(_overview_cards_html(df), unsafe_allow_html=True)
    st.markdown("")
    st.markdown("---")

    # ── Variable Explorer ──────────────────────────────────────────
    st.markdown("#### 🔬 Variable Explorer")
    st.caption("Inspect individual features — chart type is chosen automatically: pie for binary, bar for small-count / text categorical, histogram for continuous.")

    explorable = [c for c in df.columns if c not in EXCLUDE and c != target]
    vcol1, vcol2, vcol3 = st.columns([2, 1, 1])
    with vcol1:
        feat = st.selectbox("Feature", explorable, key="var_explorer_feat")
    feat_kind = categorical_chart_kind(df[feat])
    with vcol2:
        if feat_kind == "continuous":
            chart_type = st.selectbox(
                "Chart type", ["Histogram", "Box", "Violin"], key="var_explorer_chart",
            )
        else:
            chart_type = "Pie" if feat_kind == "pie" else "Bar"
            st.text_input(
                "Chart type (auto)", value=chart_type,
                disabled=True, key="var_explorer_chart_auto",
            )
    with vcol3:
        group_by_target = st.checkbox(
            f"Split by {target}",
            value=False,
            key="var_explorer_group",
            help=f"Split the distribution by the target variable ({target}).",
        )

    desc = info.get("features_desc", {}).get(feat, "")
    if desc:
        st.markdown(
            f'<div style="color:#666;font-style:italic;font-size:0.85rem;margin-bottom:8px;">{desc}</div>',
            unsafe_allow_html=True,
        )

    target_is_binary = set(df[target].dropna().unique()) == {0, 1}
    color_arg = target if group_by_target else None
    plot_df = df.copy()
    if color_arg and target_is_binary:
        plot_df[target] = plot_df[target].astype(str)

    vchart_col, vstats_col = st.columns([3, 1])
    with vchart_col:
        s = df[feat]
        if chart_type == "Pie":
            if group_by_target:
                target_levels = sorted(df[target].dropna().unique())
                target_labels = categorical_labels(target_levels)
                specs = [[{"type": "pie"} for _ in target_levels]]
                fig = make_subplots(
                    rows=1, cols=len(target_levels),
                    specs=specs,
                    subplot_titles=[f"{target} = {lab}" for lab in target_labels],
                )
                for i, level in enumerate(target_levels):
                    sub = df.loc[df[target] == level, feat]
                    counts = sub.dropna().value_counts().sort_index()
                    labels = categorical_labels(counts.index)
                    fig.add_trace(go.Pie(
                        labels=labels, values=counts.values, hole=0.45,
                        marker=dict(colors=INDIGO_DISCRETE),
                        textinfo="percent+label", sort=False,
                    ), row=1, col=i + 1)
                fig.update_layout(
                    template="plotly_white", height=400,
                    title=f"Distribution of {feat} by {target}",
                )
            else:
                fig = _pie_counts(s, title=f"Distribution of {feat}")
        elif chart_type == "Bar":
            if group_by_target:
                ct = (df.groupby([feat, target]).size()
                        .rename("count").reset_index())
                ct[target] = ct[target].astype(str)
                fig = px.bar(
                    ct, x=feat, y="count", color=target, barmode="group",
                    color_discrete_sequence=INDIGO_DISCRETE,
                    title=f"Counts of {feat} by {target}",
                )
                fig.update_xaxes(type="category")
            else:
                fig = _bar_counts(s, title=f"Counts of {feat}", xlabel=feat)
        elif chart_type == "Histogram":
            bins = st.slider("Bins", 10, 100, 30, key="var_bins")
            fig = px.histogram(
                plot_df, x=feat, nbins=bins, color=color_arg,
                color_discrete_sequence=["#4338CA", "#C7D2FE"],
                barmode="overlay" if color_arg else "relative",
                opacity=0.75 if color_arg else 1.0,
                title=f"Distribution of {feat}",
            )
        elif chart_type == "Box":
            fig = px.box(
                plot_df, y=feat, color=color_arg, points="outliers",
                color_discrete_sequence=["#4338CA", "#C7D2FE"],
                title=f"Box plot of {feat}",
            )
        else:  # Violin
            fig = px.violin(
                plot_df, y=feat, color=color_arg, box=True, points=False,
                color_discrete_sequence=["#4338CA", "#C7D2FE"],
                title=f"Violin plot of {feat}",
            )
        fig.update_layout(template="plotly_white", height=fig.layout.height or 400)
        st.plotly_chart(fig, width='stretch')

    with vstats_col:
        s = df[feat]
        n_miss = int(s.isnull().sum())
        miss_pct = n_miss / len(df) * 100
        n_unique = int(s.nunique())
        if feat_kind == "continuous":
            stat_pairs = [
                ("μ (mean)", f"{s.mean():.2f}"),
                ("σ (std)", f"{s.std():.2f}"),
                ("min", f"{s.min():.2f}"),
                ("max", f"{s.max():.2f}"),
                ("unique", f"{n_unique}"),
                ("missing", f"{n_miss} ({miss_pct:.1f}%)"),
            ]
        else:
            counts = s.dropna().value_counts().sort_values(ascending=False)
            top_val = counts.index[0] if not counts.empty else "—"
            top_label = categorical_labels([top_val])[0] if not counts.empty else "—"
            stat_pairs = [
                ("levels", f"{n_unique}"),
                ("most common", str(top_label)),
                ("top share", f"{counts.iloc[0] / counts.sum() * 100:.1f}%" if not counts.empty else "—"),
                ("missing", f"{n_miss} ({miss_pct:.1f}%)"),
            ]
            if feat_kind == "pie" and set(s.dropna().unique()) == {0, 1}:
                stat_pairs.insert(2, ("% yes (1)", f"{s.mean() * 100:.1f}%"))
        for label, value in stat_pairs:
            st.markdown(f"""
            <div class="metric-card" style="min-height:52px;padding:0.5rem 1rem;margin-bottom:4px;">
                <h3 style="font-size:0.72rem;">{label}</h3>
                <p style="font-size:1rem;">{value}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Correlation Matrix ─────────────────────────────────────────
    st.markdown("#### 🔗 Correlation Matrix (method-selectable)")
    corr_cols = [c for c in features + [target] if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c]) and c not in EXCLUDE]

    ccol1, ccol2 = st.columns([1, 1])
    with ccol1:
        corr_method = st.selectbox(
            "Method", ["pearson", "spearman", "kendall"], index=0, key="corr_method"
        )
    with ccol2:
        min_abs = st.slider(
            "Hide |corr| below", 0.0, 1.0, 0.0, 0.05, key="corr_thresh",
            help="Cells with absolute correlation below this threshold are hidden.",
        )

    corr = df[corr_cols].corr(method=corr_method)
    display_corr = corr.where(corr.abs() >= min_abs)

    fig = px.imshow(
        display_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1,
        title=f"{corr_method.capitalize()} correlation matrix",
    )
    fig.update_layout(template="plotly_white", height=max(450, 26 * len(corr_cols)))
    st.plotly_chart(fig, width='stretch')

    if target in corr.columns:
        target_corr = corr[target].drop(target).dropna().sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        st.markdown(f"**Top features correlated with `{target}`:**")
        for feat, val in target_corr.head(8).items():
            direction = "+" if val > 0 else "−"
            bar_width = int(abs(val) * 100)
            st.markdown(
                f"- **{feat}** → {direction}{abs(val):.3f} "
                f'<span style="display:inline-block;height:10px;width:{bar_width}px;'
                f'background:#4338CA;border-radius:4px;vertical-align:middle;"></span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Outlier Analysis ───────────────────────────────────────────
    st.markdown("#### ⚠️ Outlier Analysis")
    ocol1, ocol2 = st.columns([1, 1])
    with ocol1:
        method = st.radio(
            "Detection method", ["IQR", "Z-score"], horizontal=True, key="outlier_method"
        )
    with ocol2:
        if method == "IQR":
            k = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1, key="iqr_k")
            z = 3.0
        else:
            z = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1, key="z_thresh")
            k = 1.5

    outlier_feats = [f for f in features if f not in EXCLUDE]
    outlier_data = _outlier_summary(df, outlier_feats, method=method, z_thresh=z, iqr_k=k)
    outlier_data.sort(key=lambda x: -x[2])

    if outlier_data:
        odf = pd.DataFrame(outlier_data, columns=["Feature", "Count", "% of rows", "Lower", "Upper"])

        bar_fig = px.bar(
            odf, x="Feature", y="% of rows",
            color="% of rows", color_continuous_scale="Blues",
            title=f"Outlier rate per feature ({method})",
            hover_data={"Count": True, "Lower": ":.2f", "Upper": ":.2f"},
        )
        bar_fig.update_layout(
            template="plotly_white", height=380, xaxis_tickangle=-35,
            coloraxis_showscale=False,
        )
        st.plotly_chart(bar_fig, width='stretch')

        inspect_feat = st.selectbox(
            "Inspect a feature's outliers",
            [r[0] for r in outlier_data],
            key="outlier_inspect",
        )
        row = next(r for r in outlier_data if r[0] == inspect_feat)
        _, n_out, pct, lower, upper = row

        icol1, icol2 = st.columns([2, 1])
        with icol1:
            s = df[inspect_feat].dropna()
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=s, name=inspect_feat, marker_color="#4338CA",
                boxpoints="outliers", jitter=0.3, pointpos=0,
            ))
            fig.add_hline(y=lower, line_dash="dash", line_color="#d73a49",
                          annotation_text=f"lower={lower:.2f}", annotation_position="right")
            fig.add_hline(y=upper, line_dash="dash", line_color="#d73a49",
                          annotation_text=f"upper={upper:.2f}", annotation_position="right")
            fig.update_layout(
                template="plotly_white", height=420,
                title=f"{inspect_feat} — {n_out:,} outliers flagged ({pct:.1f}%)",
                showlegend=False,
            )
            st.plotly_chart(fig, width='stretch')
        with icol2:
            severity = "Low" if pct < 1 else ("Medium" if pct < 5 else "High")
            sev_color = "#2ea043" if pct < 1 else ("#f0ad4e" if pct < 5 else "#d73a49")
            st.markdown(f"""
            <div class="metric-card"><h3>Outlier Count</h3><p>{n_out:,}</p></div>
            <div class="metric-card"><h3>Share of Rows</h3><p>{pct:.1f}%</p></div>
            <div class="metric-card"><h3>Lower Bound</h3><p style="font-size:1.3rem;">{lower:.2f}</p></div>
            <div class="metric-card"><h3>Upper Bound</h3><p style="font-size:1.3rem;">{upper:.2f}</p></div>
            <div class="metric-card" style="border-left-color:{sev_color};">
                <h3>Severity</h3><p style="color:{sev_color};">{severity}</p>
            </div>
            """, unsafe_allow_html=True)


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

    st.markdown("---")

    # ── 6. Interactive Data Exploration Report ──────────────────────
    st.markdown("## 📑 Data Exploration Report")
    st.caption("Interactive profiling — pan, zoom, hover, and pick features to uncover patterns.")
    _render_data_report(df, features, target, info)
