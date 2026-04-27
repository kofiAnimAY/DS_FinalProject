"""
Precompute feature importances for the Explainability page (classification).

Trains Random Forest Classifier and Gradient Boosting Classifier on each
dataset, computes built-in, permutation (ROC AUC drop), and SHAP importances,
and pickles the results to `cache/importance_<dataset>_<model>.pkl` so the
Streamlit page can render them instantly.

Run once:  python precompute_importance.py
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from data_loader import DATASETS, get_features, get_target, load_data, preprocess

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODELS = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1,
    ),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, random_state=42,
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        random_state=42, class_weight="balanced",
    ),
    "MLP": lambda: MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=1000, random_state=42,
    ),
}


def cache_path(dataset_key: str, model_name: str) -> Path:
    safe_model = model_name.lower().replace(" ", "_")
    return CACHE_DIR / f"importance_{dataset_key}_{safe_model}.pkl"


def _extract_positive_class_shap(shap_values_raw) -> np.ndarray:
    """Extract SHAP values for the positive (class=1) output across versions."""
    if isinstance(shap_values_raw, list):
        arr = np.asarray(shap_values_raw[-1])
    else:
        arr = np.asarray(shap_values_raw)
    if arr.ndim == 3:
        arr = arr[..., -1]
    return arr


def _extract_expected_value(explainer) -> float:
    """Extract the base log-odds value for the positive class."""
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)) and np.asarray(ev).size > 1:
        return float(np.asarray(ev).flatten()[-1])
    return float(np.asarray(ev).flatten()[0])


def compute_for(dataset_key: str, model_name: str) -> dict:
    df = preprocess(load_data(dataset_key))
    target = get_target(dataset_key)
    features = get_features(df, target)
    X = df[features]
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        n_jobs=-1, scoring="roc_auc",
    )
    perm_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm_result.importances_mean,
        "Std": perm_result.importances_std,
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
    except ImportError:
        print("  (shap not installed — skipping SHAP)")

    return payload


def main() -> None:
    print(f"Cache directory: {CACHE_DIR}")
    for ds_key in DATASETS.values():
        for model_name in MODELS:
            t0 = time.time()
            print(f"→ {ds_key:10s} · {model_name}")
            payload = compute_for(ds_key, model_name)
            out = cache_path(ds_key, model_name)
            with out.open("wb") as fh:
                pickle.dump(payload, fh)
            print(f"  saved {out.name} ({time.time() - t0:.1f}s)")
    print("Done.")


if __name__ == "__main__":
    main()
