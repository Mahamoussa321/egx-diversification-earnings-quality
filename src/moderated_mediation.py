from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .econometrics import fit_ols_fe


def bootstrap_moderated_mediation(
    df: pd.DataFrame,
    y: str,
    x: str,
    mediator: str,
    moderator: str,
    controls: Iterable[str],
    firm_col: str,
    year_col: str,
    n_boot: int = 5000,
    seed: int = 2026,
    cluster_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap conditional indirect effects for a first-stage mediation and outcome moderation model.

    First-stage model:
        mediator ~ x + controls + firm FE + year FE

    Outcome model:
        y ~ x + mediator + moderator + mediator*moderator + controls + firm FE + year FE

    Conditional indirect effect:
        a * (b_mediator + b_interaction * moderator_value)
    """
    rng = np.random.default_rng(seed)
    controls = [c for c in controls if c in df.columns]
    needed = [y, x, mediator, moderator, firm_col, year_col] + controls
    data = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if data.empty:
        raise ValueError("No complete cases available for moderated mediation.")

    interaction = f"{mediator}_x_{moderator}"
    data[interaction] = (data[mediator] - data[mediator].mean()) * (data[moderator] - data[moderator].mean())

    mod_values = {
        "Low governance (-1 SD)": float(data[moderator].mean() - data[moderator].std()),
        "Mean governance": float(data[moderator].mean()),
        "High governance (+1 SD)": float(data[moderator].mean() + data[moderator].std()),
    }

    draws = []
    firms = data[firm_col].dropna().unique()
    cluster = cluster_col if cluster_col in data.columns else firm_col

    for b in range(n_boot):
        sampled_firms = rng.choice(firms, size=len(firms), replace=True)
        sample = pd.concat([data.loc[data[firm_col] == f] for f in sampled_firms], ignore_index=True)
        try:
            m1, _, _ = fit_ols_fe(
                sample,
                y=mediator,
                xvars=[x] + controls,
                model_name="first_stage",
                firm_fe=firm_col,
                year_fe=year_col,
                cluster_col=cluster,
            )
            m2, _, _ = fit_ols_fe(
                sample,
                y=y,
                xvars=[x, mediator, moderator, interaction] + controls,
                model_name="outcome",
                firm_fe=firm_col,
                year_fe=year_col,
                cluster_col=cluster,
            )
            a = _coef(m1, x)
            b_med = _coef(m2, mediator)
            b_int = _coef(m2, interaction)
            for label, value in mod_values.items():
                draws.append(
                    {
                        "bootstrap": b,
                        "moderator_level": label,
                        "moderator_value": value,
                        "a_path": a,
                        "b_path_conditional": b_med + b_int * (value - data[moderator].mean()),
                        "conditional_indirect_effect": a * (b_med + b_int * (value - data[moderator].mean())),
                    }
                )
        except Exception:
            # Keep the bootstrap robust to occasional singular resamples.
            continue

    boot = pd.DataFrame(draws)
    if boot.empty:
        raise RuntimeError("Bootstrap failed for all resamples. Check variables and sample size.")

    summary = (
        boot.groupby("moderator_level")
        .agg(
            moderator_value=("moderator_value", "first"),
            mean_indirect=("conditional_indirect_effect", "mean"),
            ci_2_5=("conditional_indirect_effect", lambda x: np.quantile(x, 0.025)),
            ci_97_5=("conditional_indirect_effect", lambda x: np.quantile(x, 0.975)),
            n_successful_bootstraps=("conditional_indirect_effect", "count"),
        )
        .reset_index()
    )
    return summary, boot


def _coef(result, variable: str) -> float:
    if variable in result.params.index:
        return float(result.params[variable])
    quoted = f"Q('{variable}')"
    if quoted in result.params.index:
        return float(result.params[quoted])
    # statsmodels can keep original interaction label; fallback search by ending.
    matches = [idx for idx in result.params.index if variable in idx]
    if matches:
        return float(result.params[matches[0]])
    raise KeyError(f"Could not find coefficient for {variable}. Available: {list(result.params.index)}")
