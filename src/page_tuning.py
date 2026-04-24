"""
Page 5 — Hyperparameter Tuning (Classification)
=================================================
Automated hyperparameter optimization using Optuna for the 5 classifiers,
scored by ROC AUC on stratified cross-validation.
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
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve,
)
from data_loader import dataset_selector, get_target, get_features, preprocess
from src import wandb_tracker


def render():
    ds_key, df, info = dataset_selector()
    df = preprocess(df)
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## ⚙️ Hyperparameter Tuning")
    st.caption(
        "Optimize classifier hyperparameters with Optuna. Scoring is ROC AUC "
        "on stratified 5-fold cross-validation — replacing manual trial-and-error "
        "with automated Bayesian search."
    )
    st.markdown("---")

    # ── Data prep (stratified split) ────────────────────────────────
    X = df[features].values
    y = df[target].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Config ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox(
            "Model to tune",
            ["🧠 MLP (Neural Network)", "Random Forest", "Gradient Boosting",
             "Logistic Regression", "Decision Tree"],
        )
    with col2:
        n_trials = st.slider("Number of trials", 5, 100, 15, step=5)
    with col3:
        cv_folds = st.slider("CV folds", 3, 10, 5)

    # ── Hyperparameter search spaces ────────────────────────────────
    st.markdown("### 🔧 Search Space")
    search_spaces = {
        "🧠 MLP (Neural Network)": {
            "n_hidden_layers": "1 — 4",
            "neurons_per_layer": "16 — 256",
            "activation": "relu, tanh, logistic",
            "learning_rate_init": "0.0001 — 0.01",
            "alpha (L2 penalty)": "0.0001 — 0.1",
            "batch_size": "16 — 128",
            "max_iter": "200 — 1000",
        },
        "Random Forest": {
            "n_estimators": "50 — 500",
            "max_depth": "3 — 30",
            "min_samples_split": "2 — 20",
            "min_samples_leaf": "1 — 10",
            "max_features": "sqrt, log2",
        },
        "Gradient Boosting": {
            "n_estimators": "50 — 500",
            "max_depth": "2 — 10",
            "learning_rate": "0.01 — 0.3",
            "subsample": "0.6 — 1.0",
            "min_samples_split": "2 — 20",
        },
        "Logistic Regression": {
            "C (inverse regularization)": "0.001 — 100",
            "penalty": "l1, l2",
            "class_weight": "balanced, none",
        },
        "Decision Tree": {
            "max_depth": "2 — 25",
            "min_samples_split": "2 — 20",
            "min_samples_leaf": "1 — 10",
            "criterion": "gini, entropy",
        },
    }

    # ── MLP architecture visualizer ─────────────────────────────────
    if model_name == "🧠 MLP (Neural Network)":
        st.markdown("### 🏗️ Neural Network Architecture Preview")
        st.markdown(
            "The MLP (Multi-Layer Perceptron) is a fully-connected feedforward "
            "neural network. Optuna searches over number of hidden layers, neurons "
            "per layer, activation function, learning rate, and regularization."
        )
        st.markdown(
            "```\n"
            "Input Layer ──▶ Hidden Layer(s) ──▶ Output Layer\n"
            "  (features)    (relu/tanh/logistic)    (P(Response=1))\n"
            "```"
        )

    space_df = pd.DataFrame(
        [{"Parameter": k, "Range": v} for k, v in search_spaces[model_name].items()]
    )
    st.dataframe(space_df, width='stretch', hide_index=True)

    # ── W&B toggle ──────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log study to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
        help="Set WANDB_API_KEY in .env to enable.",
    )

    # ── Run optimization ────────────────────────────────────────────
    if st.button("🚀 Start Optimization", type="primary", width='stretch'):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            st.error("Install Optuna: `pip install optuna`")
            return

        wb_run = None
        if track_wandb:
            wb_run = wandb_tracker.init_run(
                run_name=f"{ds_key}-tune-{model_name}",
                config={
                    "dataset": ds_key, "model": model_name,
                    "n_trials": n_trials, "cv_folds": cv_folds,
                    "target": target, "n_features": len(features),
                },
                job_type="classify-hparam-search",
            )

        def objective(trial):
            if model_name == "🧠 MLP (Neural Network)":
                n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
                hidden_layers = tuple(
                    trial.suggest_int(f"neurons_layer_{i}", 16, 256, log=True)
                    for i in range(n_layers)
                )
                params = {
                    "hidden_layer_sizes": hidden_layers,
                    "activation": trial.suggest_categorical(
                        "activation", ["relu", "tanh", "logistic"]),
                    "learning_rate_init": trial.suggest_float(
                        "learning_rate_init", 1e-4, 1e-2, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-4, 0.1, log=True),
                    "batch_size": trial.suggest_int("batch_size", 16, 128, log=True),
                    "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
                    "random_state": 42,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                }
                model = MLPClassifier(**params)
                X_use, y_use = X_train_s, y_train
            elif model_name == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                    "class_weight": "balanced",
                    "random_state": 42, "n_jobs": -1,
                }
                model = RandomForestClassifier(**params)
                X_use, y_use = X_train, y_train
            elif model_name == "Gradient Boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "random_state": 42,
                }
                model = GradientBoostingClassifier(**params)
                X_use, y_use = X_train, y_train
            elif model_name == "Logistic Regression":
                params = {
                    "C": trial.suggest_float("C", 0.001, 100, log=True),
                    "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                    "class_weight": trial.suggest_categorical(
                        "class_weight", ["balanced", None]),
                    "solver": "liblinear",
                    "max_iter": 2000,
                    "random_state": 42,
                }
                model = LogisticRegression(**params)
                X_use, y_use = X_train_s, y_train
            else:  # Decision Tree
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 25),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    "class_weight": "balanced",
                    "random_state": 42,
                }
                model = DecisionTreeClassifier(**params)
                X_use, y_use = X_train, y_train

            scores = cross_val_score(
                model, X_use, y_use, cv=cv_folds, scoring="roc_auc", n_jobs=1,
            )
            return scores.mean()

        progress = st.progress(0, text="Optimizing...")
        live_log = st.empty()
        log_lines: list[str] = []
        study = optuna.create_study(direction="maximize", study_name=model_name)

        def callback(study, trial):
            progress.progress(
                (trial.number + 1) / n_trials,
                text=f"Trial {trial.number + 1}/{n_trials} — Best ROC AUC: "
                     f"{study.best_value:.4f}",
            )
            score = trial.value if trial.value is not None else float("nan")
            params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
            log_lines.append(
                f"Trial {trial.number + 1:>3}/{n_trials} │ "
                f"ROC AUC={score:.4f} │ best={study.best_value:.4f} │ {params_str}"
            )
            live_log.code("\n".join(log_lines), language="text")
            wandb_tracker.log_metrics(wb_run, {
                "trial/roc_auc": score if score == score else 0.0,
                "trial/best_roc_auc": study.best_value,
            }, step=trial.number)

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        progress.empty()

        # ── Evaluate best model on test set ─────────────────────────
        best_params = dict(study.best_params)
        display_params = dict(best_params)  # copy for UI
        if model_name == "🧠 MLP (Neural Network)":
            n_layers = best_params.pop("n_hidden_layers")
            hidden_layers = tuple(
                best_params.pop(f"neurons_layer_{i}") for i in range(n_layers)
            )
            for k in list(best_params.keys()):
                if k.startswith("neurons_layer_"):
                    best_params.pop(k)
            best_model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                **best_params, random_state=42,
                early_stopping=True, validation_fraction=0.1,
            )
            best_model.fit(X_train_s, y_train)
            y_proba = best_model.predict_proba(X_test_s)[:, 1]
            display_params = {
                "n_hidden_layers": n_layers,
                "architecture": " → ".join(str(n) for n in hidden_layers),
                **best_params,
            }
        elif model_name == "Random Forest":
            best_model = RandomForestClassifier(
                **best_params, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            best_model.fit(X_train, y_train)
            y_proba = best_model.predict_proba(X_test)[:, 1]
        elif model_name == "Gradient Boosting":
            best_model = GradientBoostingClassifier(**best_params, random_state=42)
            best_model.fit(X_train, y_train)
            y_proba = best_model.predict_proba(X_test)[:, 1]
        elif model_name == "Logistic Regression":
            best_model = LogisticRegression(
                **best_params, solver="liblinear",
                max_iter=2000, random_state=42,
            )
            best_model.fit(X_train_s, y_train)
            y_proba = best_model.predict_proba(X_test_s)[:, 1]
        else:  # Decision Tree
            best_model = DecisionTreeClassifier(
                **best_params, class_weight="balanced", random_state=42,
            )
            best_model.fit(X_train, y_train)
            y_proba = best_model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= 0.5).astype(int)

        # Store results
        trials_data = []
        for t in study.trials:
            row = {"Trial": t.number, "ROC AUC (CV)": t.value}
            row.update(t.params)
            trials_data.append(row)

        st.session_state["tune_study"] = study
        st.session_state["tune_trials"] = pd.DataFrame(trials_data)
        st.session_state["tune_best_params"] = display_params
        st.session_state["tune_test_metrics"] = {
            "ROC AUC": roc_auc_score(y_test, y_proba),
            "PR AUC": average_precision_score(y_test, y_proba),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
        }
        st.session_state["tune_y_test"] = y_test
        st.session_state["tune_y_pred"] = y_pred
        st.session_state["tune_y_proba"] = y_proba
        st.session_state["tune_model_name"] = model_name
        st.session_state["tune_ready"] = True

        if wb_run is not None:
            wandb_tracker.log_metrics(wb_run, {
                "final/best_cv_roc_auc": study.best_value,
                "final/test_roc_auc": st.session_state["tune_test_metrics"]["ROC AUC"],
                "final/test_pr_auc": st.session_state["tune_test_metrics"]["PR AUC"],
                "final/test_f1": st.session_state["tune_test_metrics"]["F1"],
                "final/test_accuracy": st.session_state["tune_test_metrics"]["Accuracy"],
            })
            try:
                wb_run.summary["best_params"] = {
                    k: v for k, v in display_params.items()
                    if isinstance(v, (int, float, str))
                }
            except Exception:
                pass
            wandb_tracker.finish_run(wb_run)

    # ── Display results ─────────────────────────────────────────────
    if not st.session_state.get("tune_ready"):
        st.info("Click **Start Optimization** to begin hyperparameter search.")
        return

    trials_df = st.session_state["tune_trials"]
    best_params = st.session_state["tune_best_params"]
    test_metrics = st.session_state["tune_test_metrics"]
    y_test = st.session_state["tune_y_test"]
    y_pred = st.session_state["tune_y_pred"]
    y_proba = st.session_state["tune_y_proba"]
    tuned_model = st.session_state["tune_model_name"]

    st.markdown("---")

    # ── Best parameters ─────────────────────────────────────────────
    st.markdown("### 🏆 Best Hyperparameters")
    st.success(
        f"**{tuned_model}** — Best CV ROC AUC: "
        f"{st.session_state['tune_study'].best_value:.4f}"
    )

    param_cols = st.columns(max(1, len(best_params)))
    for i, (k, v) in enumerate(best_params.items()):
        with param_cols[i]:
            display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
            st.metric(k, display_val)

    # ── Test set performance ────────────────────────────────────────
    st.markdown("### 📈 Test Set Performance (Best Model)")
    m_cols = st.columns(4)
    for i, (metric, val) in enumerate(test_metrics.items()):
        m_cols[i].metric(metric, f"{val:.4f}")

    st.markdown("---")

    # ── Optimization history + confusion matrix + ROC ──────────────
    st.markdown("### 📉 Optimization & Diagnostics")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        best_so_far = trials_df["ROC AUC (CV)"].cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trials_df["Trial"], y=trials_df["ROC AUC (CV)"],
            mode="markers", name="Trial score",
            marker=dict(color="#A5B4FC", size=6, opacity=0.5),
        ))
        fig.add_trace(go.Scatter(
            x=trials_df["Trial"], y=best_so_far,
            mode="lines", name="Best so far",
            line=dict(color="#4338CA", width=3),
        ))
        fig.update_layout(
            template="plotly_white", height=400,
            title="Optimization Progress",
            xaxis_title="Trial", yaxis_title="ROC AUC (CV)",
        )
        st.plotly_chart(fig, width='stretch')

    with col_h2:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        total = cm.sum()
        text = [
            [f"{cm[0, 0]}<br>({cm[0, 0] / total * 100:.1f}%)",
             f"{cm[0, 1]}<br>({cm[0, 1] / total * 100:.1f}%)"],
            [f"{cm[1, 0]}<br>({cm[1, 0] / total * 100:.1f}%)",
             f"{cm[1, 1]}<br>({cm[1, 1] / total * 100:.1f}%)"],
        ]
        fig = go.Figure(data=go.Heatmap(
            z=cm, text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            x=["Pred: No (0)", "Pred: Yes (1)"],
            y=["Actual: No (0)", "Actual: Yes (1)"],
        ))
        fig.update_layout(
            template="plotly_white", height=400,
            title=f"Best {tuned_model} — Confusion Matrix (threshold=0.5)",
        )
        st.plotly_chart(fig, width='stretch')

    # ── ROC curve for best model ───────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color="#4338CA", width=3),
        name=f"{tuned_model} (AUC={test_metrics['ROC AUC']:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Random", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_white", height=400,
        title=f"Best {tuned_model} — ROC Curve (Test Set)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # ── Parallel coordinates ────────────────────────────────────────
    st.markdown("### 🔀 Hyperparameter Exploration")
    param_names = [c for c in trials_df.columns if c not in ("Trial", "ROC AUC (CV)")]
    if len(param_names) >= 2:
        dims = [dict(label="ROC AUC (CV)", values=trials_df["ROC AUC (CV)"])]
        for p in param_names:
            col = trials_df[p]
            if col.dtype == object:
                # categorical — encode to integers for parcoords
                codes, uniques = pd.factorize(col.astype(str))
                dims.append(dict(
                    label=p, values=codes,
                    tickvals=list(range(len(uniques))),
                    ticktext=list(uniques),
                ))
            else:
                dims.append(dict(label=p, values=col))
        fig = go.Figure(go.Parcoords(
            line=dict(
                color=trials_df["ROC AUC (CV)"],
                colorscale="Blues",
                showscale=True,
                cmin=trials_df["ROC AUC (CV)"].min(),
                cmax=trials_df["ROC AUC (CV)"].max(),
            ),
            dimensions=dims,
        ))
        fig.update_layout(height=500, title="Parallel Coordinates — All Trials")
        st.plotly_chart(fig, width='stretch')

    # ── Experiment log ──────────────────────────────────────────────
    st.markdown("### 📋 Full Experiment Log")
    st.dataframe(
        trials_df.sort_values("ROC AUC (CV)", ascending=False).style.format(
            {c: "{:.4f}" for c in trials_df.select_dtypes(include="float").columns}
        ),
        width='stretch',
        height=400,
    )
