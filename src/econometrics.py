from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class ModelResultSummary:
    model_name: str
    formula: str
    nobs: int
    r2: float | None
    table: pd.DataFrame


def _quote(name: str) -> str:
    """Quote a variable for patsy formulas if needed."""
    if name.replace("_", "").isalnum() and not name[0].isdigit():
        return name
    return f"Q('{name}')"


def make_formula(
    y: str,
    xvars: Iterable[str],
    firm_fe: Optional[str] = None,
    year_fe: Optional[str] = None,
    industry_fe: Optional[str] = None,
) -> str:
    terms: List[str] = [_quote(x) for x in xvars]
    if firm_fe:
        terms.append(f"C({_quote(firm_fe)})")
    if year_fe:
        terms.append(f"C({_quote(year_fe)})")
    if industry_fe:
        terms.append(f"C({_quote(industry_fe)})")
    return f"{_quote(y)} ~ " + " + ".join(terms)


def fit_ols_fe(
    df: pd.DataFrame,
    y: str,
    xvars: Iterable[str],
    model_name: str,
    firm_fe: Optional[str] = None,
    year_fe: Optional[str] = None,
    industry_fe: Optional[str] = None,
    cluster_col: Optional[str] = None,
):
    """Fit OLS with dummy fixed effects and optional clustered standard errors.

    This is compatible with a wide range of Python environments. For publication, compare with
    linearmodels.PanelOLS if desired.
    """
    xvars = list(xvars)
    needed = [y] + xvars + [c for c in [firm_fe, year_fe, industry_fe, cluster_col] if c]
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in needed if c in df.columns]).copy()
    formula = make_formula(y, xvars, firm_fe=firm_fe, year_fe=year_fe, industry_fe=industry_fe)
    model = smf.ols(formula, data=data)
    if cluster_col and cluster_col in data.columns and data[cluster_col].nunique() > 1:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    else:
        result = model.fit(cov_type="HC3")
    return result, formula, data


def tidy_result(result, model_name: str, keep_vars: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return a publication-style coefficient table."""
    params = result.params
    se = result.bse
    pvals = result.pvalues
    rows = []
    keep_set = set(keep_vars) if keep_vars else None
    for var in params.index:
        if var.startswith("C("):
            continue
        if keep_set and var not in keep_set and var != "Intercept":
            # try quoted variable forms used by patsy
            cleaned = var.replace("Q('", "").replace("')", "")
            if cleaned not in keep_set:
                continue
        p = float(pvals[var]) if var in pvals else np.nan
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        rows.append(
            {
                "model": model_name,
                "variable": var,
                "coef": float(params[var]),
                "std_error": float(se[var]) if var in se else np.nan,
                "t_value": float(result.tvalues[var]) if var in result.tvalues else np.nan,
                "p_value": p,
                "sig": stars,
            }
        )
    return pd.DataFrame(rows)


def model_stats(result, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "nobs": int(result.nobs),
                "r_squared": getattr(result, "rsquared", np.nan),
                "adj_r_squared": getattr(result, "rsquared_adj", np.nan),
                "f_pvalue": getattr(result, "f_pvalue", np.nan),
            }
        ]
    )


def descriptive_table(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    vars_present = [v for v in variables if v in df.columns]
    if not vars_present:
        return pd.DataFrame(columns=["Variable", "N", "Mean", "SD", "Min", "Median", "Max"])
    numeric = df[vars_present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        return pd.DataFrame(columns=["Variable", "N", "Mean", "SD", "Min", "Median", "Max"])
    desc = numeric.describe().T
    desc = desc.rename(columns={"count": "N", "mean": "Mean", "std": "SD", "min": "Min", "50%": "Median", "max": "Max"})
    return desc[["N", "Mean", "SD", "Min", "Median", "Max"]].reset_index().rename(columns={"index": "Variable"})


def correlation_table(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    vars_present = [v for v in variables if v in df.columns]
    if not vars_present:
        return pd.DataFrame()
    numeric = df[vars_present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        return pd.DataFrame()
    return numeric.corr().round(4)


def vif_table(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    vars_present = [v for v in variables if v in df.columns]
    x = df[vars_present].apply(pd.to_numeric, errors="coerce").dropna()
    x = sm.add_constant(x)
    rows = []
    for i, col in enumerate(x.columns):
        if col == "const":
            continue
        rows.append({"variable": col, "VIF": variance_inflation_factor(x.values, i)})
    return pd.DataFrame(rows)
