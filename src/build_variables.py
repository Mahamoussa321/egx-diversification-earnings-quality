from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a numeric series at lower/upper quantiles."""
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile([lower, upper])
    return x.clip(lo, hi)


def winsorize_dataframe(df: pd.DataFrame, columns: Iterable[str], limits=(0.01, 0.99)) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = winsorize_series(out[col], limits[0], limits[1])
    return out


def construct_diversification(df: pd.DataFrame, hhi_col: str | None = None, div_col: str = "DIV", segment_sales_cols: List[str] | None = None) -> pd.DataFrame:
    """Construct diversification.

    If DIV already exists, keep it. If HHI exists, use DIV = 1 - HHI.
    If segment sales columns are supplied, compute HHI from segment shares and then DIV = 1 - HHI.
    """
    out = df.copy()
    if div_col in out.columns:
        out[div_col] = pd.to_numeric(out[div_col], errors="coerce")
        return out

    if segment_sales_cols:
        present = [c for c in segment_sales_cols if c in out.columns]
        if present:
            sales = out[present].apply(pd.to_numeric, errors="coerce").clip(lower=0)
            total = sales.sum(axis=1).replace(0, np.nan)
            shares = sales.div(total, axis=0)
            out["HHI_constructed"] = (shares ** 2).sum(axis=1)
            out[div_col] = 1 - out["HHI_constructed"]
            return out

    if hhi_col and hhi_col in out.columns:
        out[hhi_col] = pd.to_numeric(out[hhi_col], errors="coerce")
        out[div_col] = 1 - out[hhi_col]
        return out

    raise KeyError("Could not construct diversification. Provide DIV, HHI, or segment sales columns.")


def construct_earnings_quality(df: pd.DataFrame, eq_col: str = "EQ", da_col: str | None = None, em_col: str | None = None) -> pd.DataFrame:
    """Construct earnings quality so that higher values mean better quality.

    If EQ already exists, keep it. If DA exists, use EQ = -abs(DA). If an existing earnings-management column
    exists, use EQ = -abs(EM). This avoids accidentally interpreting high earnings management as high quality.
    """
    out = df.copy()
    if eq_col in out.columns:
        out[eq_col] = pd.to_numeric(out[eq_col], errors="coerce")
        return out
    if da_col and da_col in out.columns:
        out[eq_col] = -pd.to_numeric(out[da_col], errors="coerce").abs()
        return out
    if em_col and em_col in out.columns:
        out[eq_col] = -pd.to_numeric(out[em_col], errors="coerce").abs()
        return out
    raise KeyError("Could not construct earnings quality. Provide EQ, DA, or an earnings-management column.")


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd


def construct_governance_index(
    df: pd.DataFrame,
    component_cols: dict[str, str],
    reverse_components: Iterable[str] | None = None,
    index_col: str = "CGQ",
) -> pd.DataFrame:
    """Construct a governance-quality index from standardized components.

    Each component is z-scored. Components in reverse_components are multiplied by -1 so that larger values
    consistently indicate stronger governance.
    """
    out = df.copy()
    if index_col in out.columns:
        out[index_col] = pd.to_numeric(out[index_col], errors="coerce")
        return out

    reverse_components = set(reverse_components or [])
    z_cols = []
    for key, col in component_cols.items():
        if col not in out.columns:
            # Try cleaned column comparison in case the Excel column was normalized.
            cleaned_lookup = {str(c).strip().lower(): c for c in out.columns}
            col_actual = cleaned_lookup.get(str(col).strip().lower())
        else:
            col_actual = col
        if not col_actual or col_actual not in out.columns:
            continue
        z_col = f"z_{key}"
        out[z_col] = zscore(out[col_actual])
        if key in reverse_components:
            out[z_col] = -out[z_col]
        z_cols.append(z_col)

    if not z_cols:
        raise KeyError("Could not construct governance index. No governance component columns found.")

    out[index_col] = out[z_cols].mean(axis=1, skipna=True)
    return out


def construct_amihud_from_daily(
    daily: pd.DataFrame,
    firm_col: str = "firm_id",
    year_col: str = "year",
    return_col: str = "daily_return",
    value_traded_col: str = "daily_value_traded",
    out_col: str = "AMIHUD",
) -> pd.DataFrame:
    """Construct firm-year Amihud illiquidity from daily trading data."""
    x = daily.copy()
    x[return_col] = pd.to_numeric(x[return_col], errors="coerce")
    x[value_traded_col] = pd.to_numeric(x[value_traded_col], errors="coerce")
    x = x.replace({value_traded_col: {0: np.nan}})
    x["_amihud_daily"] = x[return_col].abs() / x[value_traded_col]
    out = (
        x.groupby([firm_col, year_col], dropna=False)["_amihud_daily"]
        .mean()
        .reset_index()
        .rename(columns={"_amihud_daily": out_col})
    )
    return out


def add_interaction(df: pd.DataFrame, x: str, z: str, out_col: str | None = None, center: bool = True) -> pd.DataFrame:
    out = df.copy()
    name = out_col or f"{x}_x_{z}"
    xval = pd.to_numeric(out[x], errors="coerce")
    zval = pd.to_numeric(out[z], errors="coerce")
    if center:
        xval = xval - xval.mean()
        zval = zval - zval.mean()
    out[name] = xval * zval
    return out
