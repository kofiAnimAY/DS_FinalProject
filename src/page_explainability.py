"""
Page 4 — Explainability (SHAP) — Classification
================================================
Three lenses on what drives Response = 1:
- Built-in/model importance (impurity reduction for trees, permutation for others)
- Permutation importance (ROC AUC drop when feature is shuffled)
- SHAP values (per-prediction contributions)

Plus a "Key Drivers" hero summary, per-tab "How to read this" guides,
and a per-customer explanation view.

Results are loaded from `cache/importance_<dataset>_<model>.pkl` when
available (precomputed via `python precompute_importance.py`). If the
cache is missing, a button lets the user compute on the fly.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import (
    dataset_selector, get_target, get_features, preprocess,
    preprocessing_callout,
)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


ENGINEERED_DESC = {
    "Age": "Age in years (from Year_Birth, capped at 100)",
    "Tenure_Days": "Days as customer (from Dt_Customer)",
    "TotalSpend": "Total $ spent across all 6 product categories",
    "TotalPurchases": "Total purchases via web + catalog + store",
    "TotalAccepted": "Number of past campaigns accepted (0–5)",
    "HasChildren": "Has children at home (1 = yes)",
    "Education": "Education level (0 = Basic … 3 = PhD, ordinal)",
    "Marital_Status": "1 = partnered, 0 = single",
}


def _feature_desc(feat: str, info: dict) -> str:
    return (info.get("features_desc", {}).get(feat)
            or ENGINEERED_DESC.get(feat) or "")


def _direction(feature_values, shap_contribs) -> tuple[str, str]:
    """Return (arrow, plain-English) for how the feature pushes prediction."""
    s = pd.Series(feature_values)
    c = pd.Series(shap_contribs)
    if s.nunique() < 2:
        return ("→", "constant in the test set")
    corr = s.corr(c, method="spearman")
    if pd.isna(corr):
        return ("→", "no clear effect")
    if corr > 0.25:
        return ("↑", "higher values → more likely to respond")
    if corr < -0.25:
        return ("↓", "higher values → less likely to respond")
    return ("↔", "non-monotonic — direction depends on context")


def _cache_path(ds_key: str, model_name: str) -> Path:
    safe = model_name.lower().replace(" ", "_")
    return CACHE_DIR / f"importance_{ds_key}_{safe}.pkl"


def load_cached(ds_key: str, model_name: str) -> dict | None:
    path = _cache_path(ds_key, model_name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _extract_positive_class_shap(shap_values_raw) -> np.ndarray:
    if isinstance(shap_values_raw, list):
        arr = np.asarray(shap_values_raw[-1])
    else:
        arr = np.asarray(shap_values_raw)
    if arr.ndim == 3:
        arr = arr[..., -1]
    return arr


def _extract_expected_value(explainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)) and np.asarray(ev).size > 1:
        return float(np.asarray(ev).flatten()[-1])
    return float(np.asarray(ev).flatten()[0])


def compute_live(ds_key: str, model_name: str, df: pd.DataFrame,
                 target: str, features: list[str]) -> dict:
    MAX_TRAIN = 5000
    MAX_TEST = 500

    X_full, y_full = df[features], df[target].astype(int)
    if len(df) > MAX_TRAIN + MAX_TEST:
        X_full = X_full.sample(n=MAX_TRAIN + MAX_TEST, random_state=42)
        y_full = y_full.loc[X_full.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=min(MAX_TEST / len(X_full), 0.2),
        random_state=42, stratify=y_full,
    )

    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=1,
        )
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            random_state=42, class_weight="balanced",
        )
    elif model_name == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=1000, random_state=42,
        )
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42,
        n_jobs=1, scoring="roc_auc",
    )
    perm_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Importance", ascending=True)

    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
    else:
        # For models without built-in importance (e.g., MLP), use permutation importance
        imp_df = perm_df.copy()

    payload = {
        "features": features,
        "imp_df": imp_df,
        "perm_df": perm_df,
        "X_test": X_test.reset_index(drop=True),
    }

    try:
        import shap
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_arr = _extract_positive_class_shap(explainer.shap_values(X_test))
        else:
            # For other models like MLP, use KernelExplainer (slower)
            background = X_train.sample(min(100, len(X_train)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test, nsamples=100)
            if isinstance(shap_values, list):
                shap_arr = np.array(shap_values[1])  # For positive class
            else:
                shap_arr = np.array(shap_values)
        payload["shap_values"] = shap_arr
        payload["expected_value"] = _extract_expected_value(explainer)
        payload["shap_df"] = pd.DataFrame({
            "Feature": features,
            "Mean |SHAP|": np.abs(shap_arr).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=True)
    except Exception as exc:
        payload["shap_error"] = str(exc)

    return payload


def render() -> None:
    ds_key, df, info = dataset_selector()
    df = preprocess(df)
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 🔍 Explainability — Feature Importance")
    st.caption(
        "Three lenses on which features drive predictions of **Response = 1**: "
        "built-in importance, permutation importance (ROC AUC drop), and SHAP values. "
        "Plus a per-customer explanation view at the bottom."
    )
    preprocessing_callout()
    st.markdown("---")

    model_name = st.selectbox(
        "Model for explainability analysis",
        ["Random Forest", "Gradient Boosting", "Decision Tree", "MLP"],
    )

    cache_key = f"importance_{ds_key}_{model_name}"

    payload = load_cached(ds_key, model_name)
    if payload is None:
        payload = st.session_state.get(cache_key)

    if payload is not None:
        st.success(f"✨ Importances ready for **{ds_key}** · **{model_name}**.")
    else:
        st.info(
            "No precomputed cache available. "
            "Click below to train the model and compute importances live."
        )
        if st.button("🔬 Compute Feature Importance Now",
                     type="primary", width='stretch'):
            with st.spinner(
                "Training classifier and computing importances — "
                "this may take a moment..."
            ):
                payload = compute_live(ds_key, model_name, df, target, features)
            st.session_state[cache_key] = payload
            st.rerun()
        return

    # ── Key Drivers hero summary ────────────────────────────────────
    _render_key_drivers(payload, info)
    st.markdown("---")

    # ── Tabs: 4 if SHAP available, else 2 ─────────────────────────
    has_shap = "shap_df" in payload
    tab_names = ["🌲 Tree Importance", "🔀 Permutation Importance"]
    if has_shap:
        tab_names += ["💎 SHAP Values", "👤 Per-Customer"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        _render_tree_tab(payload, model_name, features, info)
        st.markdown("---")
        _render_logistic_crosscheck(df, target, features, info)
    with tabs[1]:
        _render_permutation_tab(payload, features, info)
    if has_shap:
        with tabs[2]:
            _render_shap_tab(payload, features, info)
        with tabs[3]:
            _render_per_customer_tab(payload, info)

    st.markdown("---")
    st.markdown(
        "💡 **How these tie together:** if all three methods agree on the top "
        "drivers, you can trust them. If they disagree (e.g. tree importance "
        "ranks `Income` high but permutation says it doesn't matter), you've "
        "found a feature the model is using as a shortcut rather than a "
        "genuinely predictive signal."
    )

    # ── Cross-check: Logistic Regression coefficients ──────────────



def _render_logistic_crosscheck(df, target, features, info) -> None:
    st.markdown("### 📐 Cross-Check — Logistic Regression Coefficients")
    st.caption(
        "A linear model is fully interpretable: each feature has one coefficient "
        "telling us, in standardized units, how much one standard deviation of "
        "that feature shifts the log-odds of Response = 1. Use this as a "
        "sanity check on the SHAP / tree importance story above."
    )
    with st.expander("📖 How to read this"):
        st.markdown(
            "**Why it's different from SHAP.** Tree models capture interactions "
            "and non-linearities; Logistic Regression assumes a single linear "
            "additive contribution per feature. A feature can be a top SHAP "
            "driver in the trees but a small Logistic coefficient if its effect "
            "depends on interactions — and vice versa.\n\n"
            "**Why we standardize first.** Coefficients on raw `Income` "
            "(50,000-scale) and `HasChildren` (0/1) aren't comparable. "
            "Standardizing puts them on the same per-σ scale so you can "
            "rank by magnitude.\n\n"
            "**Sign convention.** Positive = pushes log-odds toward Response = 1. "
            "Negative = pushes away. Multicollinearity (see Visualization page) "
            "can flip signs unexpectedly — if you see a counterintuitive sign, "
            "check the VIF table first."
        )

    X = df[features].fillna(df[features].median(numeric_only=True)).values
    y = df[target].astype(int).values
    X_s = StandardScaler().fit_transform(X)
    try:
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear",
        )
        model.fit(X_s, y)
    except Exception as exc:
        st.warning(f"Could not fit Logistic Regression: {exc}")
        return

    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefs,
        "|Coefficient|": np.abs(coefs),
    }).sort_values("|Coefficient|", ascending=True)

    colors = ["#dc2626" if c < 0 else "#4338CA" for c in coef_df["Coefficient"]]
    fig = go.Figure(go.Bar(
        x=coef_df["Coefficient"], y=coef_df["Feature"], orientation="h",
        marker_color=colors,
        text=[f"{c:+.3f}" for c in coef_df["Coefficient"]],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white", height=max(400, len(features) * 28),
        title="Standardized Logistic Coefficients (positive → more likely to respond)",
        xaxis_title="Coefficient (per standard deviation of feature)",
    )
    fig.add_vline(x=0, line_color="#666", line_width=1)
    st.plotly_chart(fig, width='stretch')

    # Top positive / negative drivers
    coef_sorted = coef_df.sort_values("Coefficient", ascending=False)
    pos_top = coef_sorted.head(3)
    neg_top = coef_sorted.tail(3).iloc[::-1]

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**Top positive drivers (push toward responding)**")
        for _, row in pos_top.iterrows():
            desc = _feature_desc(row["Feature"], info)
            suffix = f" — *{desc}*" if desc else ""
            st.markdown(
                f"- **`{row['Feature']}`** → `+{row['Coefficient']:.3f}`{suffix}"
            )
    with pcol2:
        st.markdown("**Top negative drivers (push away from responding)**")
        for _, row in neg_top.iterrows():
            desc = _feature_desc(row["Feature"], info)
            suffix = f" — *{desc}*" if desc else ""
            st.markdown(
                f"- **`{row['Feature']}`** → `{row['Coefficient']:.3f}`{suffix}"
            )

    st.caption(
        "💡 If a sign here disagrees with the SHAP direction-arrow above, "
        "**multicollinearity is the most likely culprit** — check the VIF "
        "table on the Data Visualization page."
    )


# ── Key Drivers card row ──────────────────────────────────────────
def _render_key_drivers(payload: dict, info: dict) -> None:
    """Top 3 drivers with friendly description and direction."""
    if "shap_df" in payload and "shap_values" in payload:
        ranked = payload["shap_df"]
        value_col = "Mean |SHAP|"
        method_label = "SHAP"
    else:
        ranked = payload["imp_df"]
        value_col = "Importance"
        method_label = "Tree importance"

    top3 = ranked.tail(3).iloc[::-1].reset_index(drop=True)
    st.markdown(f"### 🌟 Top Drivers of Response · *(by {method_label})*")
    cols = st.columns(3)
    for i, (col, row) in enumerate(zip(cols, top3.itertuples(index=False))):
        feat = row.Feature
        val = getattr(row, "_2", None) or getattr(row, value_col.replace(" ", "_"), None)
        # Robust value access:
        val = ranked.set_index("Feature").loc[feat, value_col]

        if "shap_values" in payload and "X_test" in payload:
            features_list = payload["features"]
            try:
                idx = features_list.index(feat)
                arrow, dir_text = _direction(
                    payload["X_test"][feat].values,
                    payload["shap_values"][:, idx],
                )
            except (ValueError, KeyError):
                arrow, dir_text = "→", ""
        else:
            arrow, dir_text = "→", ""

        desc = _feature_desc(feat, info)

        with col:
            st.markdown(f"""
            <div style="
                background: #EEF2FF;
                border-left: 4px solid #4338CA;
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 8px;
                min-height: 170px;
            ">
                <div style="font-size:0.7rem;color:#888;text-transform:uppercase;
                            letter-spacing:0.5px;">Driver #{i + 1}</div>
                <div style="font-size:1.05rem;font-weight:700;color:#312E81;
                            margin:4px 0;">{feat}</div>
                <div style="font-size:0.78rem;color:#666;font-style:italic;
                            margin-bottom:8px;">{desc}</div>
                <div style="font-size:1.4rem;color:#4338CA;font-weight:800;
                            line-height:1;">{arrow} {val:.3f}</div>
                <div style="font-size:0.78rem;color:#444;
                            margin-top:6px;">{dir_text}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Tab renderers ─────────────────────────────────────────────────
def _render_tree_tab(payload: dict, model_name: str,
                     features: list[str], info: dict) -> None:
    is_tree = model_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]
    importance_type = "Built-in" if not is_tree else "Tree"
    st.markdown(f"#### {model_name} — {importance_type} Feature Importance")

    with st.expander("📖 How to read this"):
        if is_tree:
            st.markdown(
                "**What it measures.** When the tree splits on a feature, it asks: "
                "*how much purer are the resulting groups?* Built-in importance "
                "sums up that improvement across every split, across every tree.\n\n"
                "**Strengths.** Fast, no extra computation needed. Good for getting "
                "a first ranking.\n\n"
                "**Weaknesses.** Biased toward high-cardinality features (e.g. "
                "`Income` with thousands of distinct values can score artificially "
                "high vs. a binary like `HasChildren`). Cross-check with "
                "permutation importance for a fairer view."
            )
        else:
            st.markdown(
                "**What it measures.** For models without built-in feature importance "
                "(like neural networks), we use permutation importance as a proxy: "
                "the drop in performance when a feature is shuffled.\n\n"
                "**Strengths.** Model-agnostic, directly measures feature impact. "
                "Fair to all feature types.\n\n"
                "**Note.** This is the same as the Permutation Importance tab, "
                "shown here for consistency."
            )

    imp_df = payload["imp_df"]
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white",
                      height=max(400, len(features) * 30))
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Top 3 driving features:**")
    for _, row in imp_df.tail(3).iloc[::-1].iterrows():
        desc = _feature_desc(row["Feature"], info)
        suffix = f" — *{desc}*" if desc else ""
        st.markdown(
            f"- **{row['Feature']}** — importance = {row['Importance']:.4f}{suffix}"
        )


def _render_permutation_tab(payload: dict, features: list[str], info: dict) -> None:
    st.markdown("#### Permutation Importance")

    with st.expander("📖 How to read this"):
        st.markdown(
            "**What it measures.** Take the trained model. Shuffle one column "
            "of the test set so it carries no information, then re-score. "
            "The drop in **ROC AUC** is the permutation importance.\n\n"
            "**Strengths.** Model-agnostic. Directly answers *'how much does "
            "the model depend on this feature?'* in a unit you care about "
            "(AUC drop). Not biased by feature cardinality.\n\n"
            "**Weaknesses.** Can underestimate importance when two features "
            "are correlated (shuffling one barely hurts because the other "
            "still carries the signal). Slower than built-in importance."
        )

    perm_df = payload["perm_df"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=perm_df["Feature"], x=perm_df["Importance"],
        orientation="h",
        marker_color="#4338CA",
        error_x=dict(type="data", array=perm_df["Std"]),
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Mean ROC AUC drop when feature is shuffled",
        height=max(400, len(features) * 30),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        "🟦 Bars to the **right** mean shuffling that feature hurt AUC — the "
        "model relied on it. Bars near zero (or slightly negative) mean the "
        "model could do without it."
    )


def _render_shap_tab(payload: dict, features: list[str], info: dict) -> None:
    shap_df = payload["shap_df"]
    shap_values = payload["shap_values"]
    X_test_df = payload["X_test"]

    st.markdown("#### SHAP — SHapley Additive exPlanations")

    with st.expander("📖 How to read this"):
        st.markdown(
            "**What it measures.** Every prediction can be decomposed into a "
            "**base value** (the model's average output) plus one **SHAP "
            "value** per feature, where the SHAP values sum exactly to the "
            "prediction. SHAP values are in **log-odds** space — positive "
            "values push toward Response = 1, negative push away.\n\n"
            "**Mean |SHAP| bar.** The average size of a feature's "
            "contribution across all customers. Reading: *'how much does this "
            "feature shift predictions, on average?'*\n\n"
            "**Beeswarm.** One dot per customer per feature. Position on the "
            "x-axis is the SHAP contribution; **color is the feature value** "
            "(dark = high, light = low). If high values cluster on the right, "
            "high values push toward 'will respond'. Spread shows how much the "
            "feature's effect varies across the population."
        )

    fig = px.bar(
        shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
        color="Mean |SHAP|", color_continuous_scale="Blues",
        title="Mean |SHAP| Value per Feature (Positive Class)",
    )
    fig.update_layout(template="plotly_white",
                      height=max(400, len(features) * 30))
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### SHAP Feature Impact (Beeswarm-style)")
    top_n = min(10, len(features))
    top_features = shap_df.tail(top_n)["Feature"].tolist()

    MAX_POINTS = 400
    n_samples = len(shap_values)
    if n_samples > MAX_POINTS:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_samples, size=MAX_POINTS, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    fig = go.Figure()
    for feat in top_features:
        idx = features.index(feat)
        fig.add_trace(go.Scatter(
            x=shap_values[sample_idx, idx],
            y=[feat] * len(sample_idx),
            mode="markers",
            marker=dict(
                color=X_test_df[feat].values[sample_idx],
                colorscale="Blues",
                size=5,
                opacity=0.55,
                colorbar=dict(title="Feature value")
                    if feat == top_features[-1] else None,
                showscale=(feat == top_features[-1]),
            ),
            name=feat, showlegend=False,
        ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="SHAP value (impact on P(Response = 1), log-odds)",
        height=max(400, top_n * 50),
    )
    fig.add_vline(x=0, line_color="#666", line_width=1)
    st.plotly_chart(fig, width='stretch')


def _render_per_customer_tab(payload: dict, info: dict) -> None:
    st.markdown("#### 👤 Per-Customer Explanation")
    st.markdown(
        "Pick a customer from the test set to see exactly which features "
        "pushed the model toward **'will respond'** (indigo) vs. "
        "**'will not respond'** (red). Useful for explaining a single "
        "decision — e.g. why was customer #42 flagged as a high-priority "
        "contact?"
    )

    shap_values = payload["shap_values"]
    X_test = payload["X_test"]
    features = payload["features"]
    expected_value = payload.get("expected_value", 0.0)
    has_expected = "expected_value" in payload

    n = len(shap_values)
    idx = st.number_input(
        f"Customer index (0 – {n - 1})",
        min_value=0, max_value=n - 1, value=0, step=1,
        key="explain_customer_idx",
    )

    contribs = shap_values[idx]
    log_odds = float(expected_value + contribs.sum())
    proba = 1.0 / (1.0 + np.exp(-log_odds))
    base_proba = 1.0 / (1.0 + np.exp(-float(expected_value)))

    abs_order = np.argsort(np.abs(contribs))[::-1][:10]
    sorted_idx = sorted(abs_order, key=lambda i: contribs[i])  # ascending → bottom-to-top

    feat_names = [features[i] for i in sorted_idx]
    feat_values = [X_test.iloc[idx, i] for i in sorted_idx]
    contribs_top = [float(contribs[i]) for i in sorted_idx]
    colors = ["#dc2626" if c < 0 else "#4338CA" for c in contribs_top]
    y_labels = [
        f"{n} = {v:.1f}" if isinstance(v, (int, float)) and abs(v) >= 1
        else f"{n} = {v}"
        for n, v in zip(feat_names, feat_values)
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure(go.Bar(
            x=contribs_top, y=y_labels, orientation="h",
            marker_color=colors,
            text=[f"{c:+.3f}" for c in contribs_top],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_white", height=440,
            xaxis_title="SHAP contribution (log-odds toward Response = 1)",
            title=f"Customer #{idx} — top 10 contributing features",
            margin=dict(l=180),
        )
        fig.add_vline(x=0, line_color="#666", line_width=1)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted P(Response)</h3>
            <p>{proba * 100:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Average baseline</h3>
            <p>{base_proba * 100:.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>+ contributions</h3>
            <p>{int((contribs > 0).sum())}</p>
        </div>
        <div class="metric-card">
            <h3>− contributions</h3>
            <p>{int((contribs < 0).sum())}</p>
        </div>
        """, unsafe_allow_html=True)

    if not has_expected:
        st.caption(
            "⚠️ This payload is from an older cache (no `expected_value` saved) — "
            "predicted probabilities use 0 as the baseline log-odds. "
            "Rerun `python precompute_importance.py` for accurate values."
        )

    delta = (proba - base_proba) * 100
    direction = "**above**" if delta > 0 else "**below**"
    st.markdown(
        f"📍 **Reading**: customer #{idx} is predicted at "
        f"{proba * 100:.1f}% — {abs(delta):.1f} percentage points "
        f"{direction} the average baseline of {base_proba * 100:.1f}%. "
        f"The features above show why."
    )
