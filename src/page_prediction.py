"""
Page 3 — Model Prediction (Classification)
============================================
Train and compare 5 classifiers side-by-side on the binary Response target.
Users can select features, adjust train/test split, and view classification
metrics plus ROC / PR curves and a business-targeting threshold widget.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from data_loader import dataset_selector, get_target, get_features, preprocess
from src import wandb_tracker


def _mlp_factory():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500,
        random_state=42, early_stopping=True,
    )


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced",
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        random_state=42, class_weight="balanced", max_depth=10,
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight="balanced", n_jobs=-1,
    ),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        random_state=42, n_estimators=200,
    ),
    "🧠 MLP (Neural Net)": _mlp_factory,
}


def render():
    ds_key, df, info = dataset_selector()
    df = preprocess(df)
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## 🤖 Model Prediction")
    st.caption(
        "Train 5 classifiers to predict customer **Response** to a marketing campaign. "
        "Compare ROC AUC, PR AUC, F1 — then slide the threshold in the targeting "
        "widget to simulate a campaign rollout."
    )
    st.markdown("---")

    # ── Feature & split config ──────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns([3, 1])
    with col_cfg1:
        selected_features = st.multiselect(
            "Select explanatory variables",
            features,
            default=features,
        )
    with col_cfg2:
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        scale_data = st.checkbox("Standardize features", value=True)

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # ── Prepare data ────────────────────────────────────────────────
    X = df[selected_features].values
    y = df[target].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    pos_rate_tr = y_train.mean() * 100
    pos_rate_te = y_test.mean() * 100
    st.markdown(
        f"**Training set:** {len(X_train):,} samples ({pos_rate_tr:.1f}% positive) · "
        f"**Test set:** {len(X_test):,} samples ({pos_rate_te:.1f}% positive)"
    )
    st.markdown("---")

    # ── Model selection ─────────────────────────────────────────────
    st.markdown("### 🏗️ Select Models to Train")
    model_choices = st.multiselect(
        "Choose models",
        list(MODELS.keys()),
        default=list(MODELS.keys()),
        label_visibility="collapsed",
    )

    if len(model_choices) < 1:
        st.warning("Select at least one model.")
        return

    # ── W&B toggle ──────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log runs to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
        help="Set WANDB_API_KEY in .env to enable.",
    )

    # ── Train all models ────────────────────────────────────────────
    if st.button("🚀 Train Models", type="primary", width='stretch'):
        results = []
        probas = {}
        preds = {}

        progress = st.progress(0, text="Training models...")
        for i, name in enumerate(model_choices):
            run = None
            if track_wandb:
                run = wandb_tracker.init_run(
                    run_name=f"{ds_key}-{name}",
                    config={
                        "dataset": ds_key, "model": name, "target": target,
                        "n_features": len(selected_features),
                        "features": selected_features,
                        "test_size": test_size, "scale_data": scale_data,
                        "train_samples": len(X_train), "test_samples": len(X_test),
                    },
                    job_type="classify-baseline",
                )

            model = MODELS[name]()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            preds[name] = y_pred
            probas[name] = y_proba

            cv_scores = cross_val_score(
                MODELS[name](), X_train, y_train, cv=5, scoring="roc_auc",
            )

            metrics = {
                "ROC AUC": roc_auc_score(y_test, y_proba),
                "PR AUC": average_precision_score(y_test, y_proba),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "Accuracy": accuracy_score(y_test, y_pred),
                "CV ROC AUC (mean)": cv_scores.mean(),
                "CV ROC AUC (std)": cv_scores.std(),
            }
            results.append({"Model": name, **metrics})

            wandb_tracker.log_metrics(run, {
                "test/roc_auc": metrics["ROC AUC"],
                "test/pr_auc": metrics["PR AUC"],
                "test/f1": metrics["F1"],
                "test/precision": metrics["Precision"],
                "test/recall": metrics["Recall"],
                "test/accuracy": metrics["Accuracy"],
                "cv/roc_auc_mean": metrics["CV ROC AUC (mean)"],
                "cv/roc_auc_std": metrics["CV ROC AUC (std)"],
            })
            wandb_tracker.finish_run(run)

            progress.progress(
                (i + 1) / len(model_choices),
                text=f"Trained {name} ✓",
            )

        progress.empty()

        st.session_state["pred_results"] = results
        st.session_state["pred_probas"] = probas
        st.session_state["pred_preds"] = preds
        st.session_state["pred_y_test"] = y_test
        st.session_state["pred_model_choices"] = model_choices

    # ── Display results ─────────────────────────────────────────────
    if "pred_results" not in st.session_state:
        st.info("Click **Train Models** to see results.")
        return

    results = st.session_state["pred_results"]
    probas = st.session_state["pred_probas"]
    preds = st.session_state["pred_preds"]
    y_test = st.session_state["pred_y_test"]
    model_choices = st.session_state["pred_model_choices"]

    results_df = pd.DataFrame(results).set_index("Model")

    # ── Leaderboard ─────────────────────────────────────────────────
    st.markdown("### 🏆 Model Leaderboard")
    sorted_df = results_df.sort_values("ROC AUC", ascending=False)
    best_model = sorted_df.index[0]
    st.success(
        f"**Best model: {best_model}** — ROC AUC = "
        f"{sorted_df.loc[best_model, 'ROC AUC']:.4f} · "
        f"PR AUC = {sorted_df.loc[best_model, 'PR AUC']:.4f} · "
        f"F1 = {sorted_df.loc[best_model, 'F1']:.4f}"
    )

    st.dataframe(
        sorted_df.style
        .format({
            "ROC AUC": "{:.4f}",
            "PR AUC": "{:.4f}",
            "F1": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Accuracy": "{:.4f}",
            "CV ROC AUC (mean)": "{:.4f}",
            "CV ROC AUC (std)": "{:.4f}",
        })
        .background_gradient(subset=["ROC AUC", "PR AUC", "F1"], cmap="Blues"),
        width='stretch',
    )

    st.markdown("---")

    # ── Performance comparison ─────────────────────────────────────
    st.markdown("### 📊 Performance Comparison")
    metric_choice = st.selectbox(
        "Metric to compare",
        ["ROC AUC", "PR AUC", "F1", "Precision", "Recall", "Accuracy", "CV ROC AUC (mean)"],
    )
    fig = px.bar(
        sorted_df.reset_index(),
        x="Model", y=metric_choice,
        color=metric_choice,
        color_continuous_scale="Blues",
        title=f"{metric_choice} by Model",
    )
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── ROC & PR curves ────────────────────────────────────────────
    st.markdown("### 📈 ROC & Precision-Recall Curves")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig = go.Figure()
        for name in model_choices:
            fpr, tpr, _ = roc_curve(y_test, probas[name])
            auc = roc_auc_score(y_test, probas[name])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={auc:.3f})",
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Random", showlegend=False,
        ))
        fig.update_layout(
            template="plotly_white", height=450,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            title="ROC Curves",
        )
        st.plotly_chart(fig, width='stretch')

    with col_r2:
        base_rate = y_test.mean()
        fig = go.Figure()
        for name in model_choices:
            pr, rc, _ = precision_recall_curve(y_test, probas[name])
            ap = average_precision_score(y_test, probas[name])
            fig.add_trace(go.Scatter(
                x=rc, y=pr, mode="lines",
                name=f"{name} (AP={ap:.3f})",
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[base_rate, base_rate], mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name=f"Base rate ({base_rate:.1%})", showlegend=False,
        ))
        fig.update_layout(
            template="plotly_white", height=450,
            xaxis_title="Recall", yaxis_title="Precision",
            title="Precision-Recall Curves",
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Model inspector ────────────────────────────────────────────
    st.markdown("### 🔍 Model Inspector")
    model_to_plot = st.selectbox("Select model to inspect", model_choices)
    y_pred = preds[model_to_plot]
    y_proba = probas[model_to_plot]

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        total = cm.sum()
        text = [
            [
                f"{cm[0, 0]}<br>({cm[0, 0] / total * 100:.1f}%)",
                f"{cm[0, 1]}<br>({cm[0, 1] / total * 100:.1f}%)",
            ],
            [
                f"{cm[1, 0]}<br>({cm[1, 0] / total * 100:.1f}%)",
                f"{cm[1, 1]}<br>({cm[1, 1] / total * 100:.1f}%)",
            ],
        ]
        fig = go.Figure(data=go.Heatmap(
            z=cm, text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            x=["Pred: No (0)", "Pred: Yes (1)"],
            y=["Actual: No (0)", "Actual: Yes (1)"],
        ))
        fig.update_layout(
            template="plotly_white", height=400,
            title=f"Confusion Matrix — {model_to_plot}",
        )
        st.plotly_chart(fig, width='stretch')

    with col_i2:
        df_p = pd.DataFrame({
            "Probability": y_proba,
            "Actual": ["Yes (1)" if v == 1 else "No (0)" for v in y_test],
        })
        fig = px.histogram(
            df_p, x="Probability", color="Actual", nbins=30,
            color_discrete_map={"No (0)": "#C7D2FE", "Yes (1)": "#4338CA"},
            barmode="overlay", opacity=0.75,
            title=f"Predicted Probability — {model_to_plot}",
        )
        fig.update_layout(
            template="plotly_white", height=400,
            xaxis_title="P(Response = 1)", yaxis_title="Count",
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Business-targeting widget ──────────────────────────────────
    st.markdown("### 🎯 Campaign Targeting — Threshold Simulator")
    st.caption(
        "Pick a contact threshold to simulate a targeted campaign. Customers with "
        "predicted probability **above** the threshold would be contacted. Lower "
        "the threshold to contact more people (higher recall, lower precision)."
    )
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        default_idx = (model_choices.index(best_model)
                       if best_model in model_choices else 0)
        bm = st.selectbox(
            "Model for targeting", model_choices,
            index=default_idx, key="target_model",
        )
        threshold = st.slider(
            "Contact threshold", 0.05, 0.95, 0.5, 0.01, key="target_thresh",
        )

    proba_sel = probas[bm]
    contact_mask = proba_sel >= threshold
    tp = int(((contact_mask) & (y_test == 1)).sum())
    fp = int(((contact_mask) & (y_test == 0)).sum())
    fn = int(((~contact_mask) & (y_test == 1)).sum())
    n_contacted = tp + fp
    total_positives = tp + fn
    contacted_pct = n_contacted / len(y_test) * 100 if len(y_test) else 0.0
    precision_at_t = tp / n_contacted if n_contacted > 0 else 0.0
    recall_at_t = tp / total_positives if total_positives > 0 else 0.0
    base_rate = float(y_test.mean())
    lift = (precision_at_t / base_rate) if base_rate > 0 else 0.0

    with col_t2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contacted", f"{n_contacted:,}",
                  delta=f"{contacted_pct:.1f}% of base", delta_color="off")
        m2.metric("Responders caught (TP)", f"{tp:,}",
                  delta=f"{recall_at_t:.1%} recall", delta_color="off")
        m3.metric("Wasted contacts (FP)", f"{fp:,}",
                  delta=f"{precision_at_t:.1%} precision", delta_color="off")
        m4.metric("Missed (FN)", f"{fn:,}", delta_color="off")

        m5, m6, m7 = st.columns(3)
        m5.metric("Response rate (contacted)", f"{precision_at_t:.1%}")
        m6.metric("Base rate", f"{base_rate:.1%}")
        m7.metric("Lift over base", f"{lift:.2f}×")
