"""
Page 1 — Business Case & Data Presentation
============================================
Presents the problem statement, dataset overview,
descriptive statistics, data quality checks,
and an interactive data exploration report.
"""

import streamlit as st

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import (
    dataset_selector, get_target, get_features,
    categorical_chart_kind, categorical_labels,
)


INDIGO_DISCRETE = ["#4338CA", "#6366F1", "#A5B4FC", "#C7D2FE", "#312E81", "#818CF8"]


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


EXCLUDE = {"ID"}


# ── Helper: outlier detection summary ──────────────────────────────
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
        else:  # Z-score
            mu, sigma = s.mean(), s.std()
            lower, upper = mu - z_thresh * sigma, mu + z_thresh * sigma
        mask = (df[col] < lower) | (df[col] > upper)
        n_out = int(mask.sum())
        pct = n_out / len(df) * 100
        results.append((col, n_out, pct, lower, upper))
    return results


# ── Overview cards (static) ────────────────────────────────────────
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


# ── Interactive data exploration report ────────────────────────────
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

    # Feature description
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
    st.markdown("#### 🔗 Correlation Matrix")
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

    # Top correlations with target
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

        # Bar chart of outlier percentage per feature
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

        # Feature-level inspection
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
            mask = (s < lower) | (s > upper)
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

    # ── Interactive Data Exploration Report ─────────────────────────
    st.markdown("## 📑 Data Exploration Report")
    st.caption("Interactive profiling — pan, zoom, hover, and pick features to uncover patterns.")
    _render_data_report(df, features, target, info)
