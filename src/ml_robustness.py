from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_ml_regression(df: pd.DataFrame, y: str, features: Iterable[str], seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run simple predictive robustness models and permutation importance.

    This is not intended to replace econometric hypothesis testing. It helps show whether the main variables
    have predictive relevance for earnings quality.
    """
    features = [f for f in features if f in df.columns]
    data = df[[y] + features].apply(pd.to_numeric, errors="coerce").dropna(subset=[y])
    x = data[features]
    target = data[y]

    models = {
        "LASSO": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, random_state=seed, max_iter=20000)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=seed)),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=seed)),
        ]),
    }

    n_splits = min(5, max(2, len(target)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    perf_rows = []
    imp_rows = []
    for name, model in models.items():
        preds = cross_val_predict(model, x, target, cv=cv)
        perf_rows.append(
            {
                "model": name,
                "n": int(len(target)),
                "r2_cv": r2_score(target, preds),
                "rmse_cv": float(np.sqrt(mean_squared_error(target, preds))),
                "mae_cv": mean_absolute_error(target, preds),
            }
        )
        fitted = model.fit(x, target)
        perm = permutation_importance(fitted, x, target, n_repeats=30, random_state=seed, scoring="r2")
        for feature, mean, sd in zip(features, perm.importances_mean, perm.importances_std):
            imp_rows.append({"model": name, "feature": feature, "importance_mean": mean, "importance_sd": sd})

    return pd.DataFrame(perf_rows), pd.DataFrame(imp_rows).sort_values(["model", "importance_mean"], ascending=[True, False])
