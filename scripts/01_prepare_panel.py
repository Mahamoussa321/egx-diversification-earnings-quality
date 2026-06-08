from __future__ import annotations

import re
import sys
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.build_variables import (
    construct_diversification,
    construct_earnings_quality,
    construct_governance_index,
    winsorize_dataframe,
)
from src.config import load_config, resolve_path
from src.io_utils import clean_column_name, read_all_excel_sheets, read_table


def resolve_col(df: pd.DataFrame, candidates: Iterable[str | None]) -> str | None:
    lookup = {clean_column_name(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if not candidate:
            continue
        key = clean_column_name(candidate).lower()
        if key in lookup:
            return lookup[key]
    return None


def to_numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "#DIV/0!": np.nan}),
        errors="coerce",
    )


def extract_ticker(value: object) -> str | None:
    """Extract an EGX-style ticker from a firm name when available."""
    if not isinstance(value, str):
        return None
    m = re.search(r"CASE:([A-Z0-9]+)", value, flags=re.I)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\(([A-Z0-9]{3,6})\)", value)
    if matches:
        return matches[-1].upper()
    return None


def canonical_company_key(value: object) -> str:
    """Normalize English company names for merging across the accounting, market, and governance files."""
    text = str(value or "").lower()
    replacements = {
        "abou kir": "abu qir",
        "abu quir": "abu qir",
        "alexandria containers and goods": "alexandria container cargo handling",
        "alexandria container&cargo handling": "alexandria container cargo handling",
        "elswedy electrics": "el sewedy electric",
        "elswedy electric": "el sewedy electric",
        "fretilizers": "fertilizer",
        "fertilizers": "fertilizer",
        "misr fertilizers": "misr fertilizer",
        "medinet nasr": "madinet nasr",
        "egypt kuwait": "egyptian kuwaiti",
        "egyptian kuwaiti holding": "egyptian kuwaiti",
        "egypt kuwaiti holding": "egyptian kuwaiti",
        "t m g": "talaat mostafa group",
        "tmg": "talaat mostafa group",
        "palm hills development company": "palm hills developments",
        "orascom hotels and development": "orascom development egypt",
        "ibnsina": "ibn sina",
        "egyptian international pharma": "egyptian international pharmaceutical industries",
        "eipico": "egyptian international pharmaceutical industries",
        "eastern tobacco": "eastern",
        "qalaa": "citadel capital",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\(case:[^)]+\)", " ", text)
    text = re.sub(r"\([A-Za-z0-9 .:&/-]+\)", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stop = {
        "company", "co", "sae", "s", "a", "e", "case", "common", "shares", "holding", "holdings",
        "group", "for", "and", "the", "egypt", "egyptian", "plc", "limited", "sall", "sa",
    }
    words = [w for w in text.split() if w and w not in stop]
    return "".join(words)


def fuzzy_map_keys(source_keys: pd.Series, target_keys: pd.Series, cutoff: float = 0.78) -> dict[str, str]:
    """Map source company keys to target company keys using conservative fuzzy matching."""
    targets = sorted({k for k in target_keys.dropna().astype(str) if k})
    mapping = {}
    for key in sorted({k for k in source_keys.dropna().astype(str) if k}):
        if key in targets:
            mapping[key] = key
            continue
        best = get_close_matches(key, targets, n=1, cutoff=cutoff)
        if best:
            score = SequenceMatcher(None, key, best[0]).ratio()
            if score >= cutoff:
                mapping[key] = best[0]
    return mapping


def parse_volume(value: object) -> float:
    """Parse volumes written as 6.92M, 347.34K, 1.2B, or plain numbers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "-"}:
        return np.nan
    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def canonicalize_main_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create canonical analysis variable names from the uploaded EGX accounting workbook."""
    out = df.copy()
    rename_candidates = {
        "firm_id": ["firm_id", "id", "Firm ID", "Company ID"],
        "firm_name": ["firm_name", "Firm", "Company", "Company Name", "اسم الشركة"],
        "year": ["year", "Year", "السنة", "fiscal_year"],
        "ROA": ["ROA", "معدل العائد على الأصول"],
        "ROE": ["ROE"],
        "size": ["size", "Firm Size", "SIZE"],
        "age": ["age", "Firm Age", "AGE"],
        "Leverage": ["Leverage", "Leverage_", "lev", "leverage_AE"],
        "total_assets": ["total assets", "total_assets", "Total Assets"],
        "ppe": ["PP&E", "PPE", "ppe"],
        "receivables": ["A/R", "AR", "receivables"],
        "total_accruals": ["accrual; liabilities", "accrual liabilities", "accrual_liabilities", "TA"],
        "sales": ["total sales", "total_sales", "Sales", "Revenue"],
        "branches": ["Number of branches", "Number_of_branches", "branches"],
        "production_lines": ["Number of production lines", "Number_of_production_lines", "production_lines"],
    }
    for target, candidates in rename_candidates.items():
        actual = resolve_col(out, candidates)
        if actual and actual != target:
            out = out.rename(columns={actual: target})
    for c in ["year", "ROA", "ROE", "size", "age", "Leverage", "total_assets", "ppe", "receivables", "total_accruals", "sales", "branches", "production_lines"]:
        if c in out.columns:
            out[c] = to_numeric_clean(out[c])
    if "firm_id" in out.columns:
        out["firm_id"] = to_numeric_clean(out["firm_id"]).astype("Int64").astype(str)
    if "firm_name" in out.columns:
        out["firm_ticker"] = out["firm_name"].apply(extract_ticker)
        out["firm_key"] = out["firm_name"].apply(canonical_company_key)
    return out


def add_count_based_diversification_proxy(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Add transparent diversification proxy when segment sales are unavailable."""
    out = df.copy()
    if "DIV" in out.columns:
        return out, None
    if "HHI" in out.columns:
        out["DIV"] = 1 - to_numeric_clean(out["HHI"])
        return out, "Constructed DIV = 1 - HHI."
    if {"branches", "production_lines"}.issubset(out.columns):
        n = to_numeric_clean(out["branches"]).fillna(0) + to_numeric_clean(out["production_lines"]).fillna(0)
        n = n.where(n > 0)
        out["HHI_proxy_count_based"] = 1 / n
        out["DIV"] = 1 - out["HHI_proxy_count_based"]
        return out, "Sales-based HHI was not available; used count-based proxy DIV = 1 - 1/(branches + production_lines). Replace with segment-sales HHI for publication."
    return out, None


def add_modified_jones_eq(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Construct discretionary-accrual earnings quality using a reproducible pooled Modified Jones approximation."""
    out = df.copy()
    if "EQ" in out.columns:
        return out, None
    if "DA" in out.columns:
        out["EQ"] = -to_numeric_clean(out["DA"]).abs()
        return out, "Constructed EQ = -abs(DA)."
    needed = {"firm_id", "year", "total_assets", "sales", "receivables", "ppe", "total_accruals"}
    if not needed.issubset(out.columns):
        return out, None
    out = out.sort_values(["firm_id", "year"]).copy()
    for col in ["total_assets", "sales", "receivables", "ppe", "total_accruals"]:
        out[col] = to_numeric_clean(out[col])
    out["lag_assets"] = out.groupby("firm_id")["total_assets"].shift(1)
    out["delta_sales"] = out.groupby("firm_id")["sales"].diff()
    out["delta_receivables"] = out.groupby("firm_id")["receivables"].diff()
    out["TA_scaled"] = out["total_accruals"] / out["lag_assets"]
    out["inv_lag_assets"] = 1 / out["lag_assets"]
    out["rev_rec_scaled"] = (out["delta_sales"] - out["delta_receivables"]) / out["lag_assets"]
    out["ppe_scaled"] = out["ppe"] / out["lag_assets"]
    reg_cols = ["TA_scaled", "inv_lag_assets", "rev_rec_scaled", "ppe_scaled"]
    reg = out[reg_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(reg) < 10:
        return out, "Modified Jones variables were available, but too few complete rows to estimate discretionary accruals."
    import statsmodels.api as sm
    X = sm.add_constant(reg[["inv_lag_assets", "rev_rec_scaled", "ppe_scaled"]])
    y = reg["TA_scaled"]
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    out.loc[reg.index, "DA"] = y - pred
    out["EQ"] = -out["DA"].abs()
    return out, f"Constructed DA/EQ using pooled Modified Jones approximation; N used for accrual model = {len(reg)}."


def build_governance_panel(path: Path, target_firm_keys: pd.Series) -> tuple[pd.DataFrame, str]:
    gov = read_all_excel_sheets(path)
    if gov.empty:
        return pd.DataFrame(), "Governance workbook was empty."
    gov_rename = {
        "firm_name_gov": ["اسم الشركة SP EGX 30 SEG", "اسم الشركة EGX 30", "اسم_الشركة_SP_EGX_30_SEG", "اسم_الشركة_EGX_30", "Company"],
        "year": ["السنة", "Year", "year"],
        "board_size": ["حجم مجلس الإدارة", "حجم_مجلس_الإدارة"],
        "board_independence": ["استقلالية أعضاء مجلس الإدارة", "استقلالية_أعضاء_مجلس_الإدارة"],
        "ownership_concentration": ["تركز الملكية", "تركز_الملكية"],
        "gender_diversity": ["تنوع نوع أعضاء مجلس الإدارة", "تنوع_نوع_أعضاء_مجلس_الإدارة"],
        "board_meetings": ["اجتماعات مجلس الإدارة", "اجتماعات_مجلس_الإدارة"],
    }
    # Coalesce candidate columns because the 2018 sheet uses a slightly different firm-name header
    # from the 2019-2021 sheets. A simple rename would keep later years blank.
    for target, candidates in gov_rename.items():
        actual_cols = []
        lookup = {clean_column_name(c).lower(): c for c in gov.columns}
        for candidate in candidates:
            key = clean_column_name(candidate).lower()
            if key in lookup and lookup[key] not in actual_cols:
                actual_cols.append(lookup[key])
        if actual_cols:
            gov[target] = gov[actual_cols].bfill(axis=1).iloc[:, 0]
    if "firm_name_gov" not in gov.columns or "year" not in gov.columns:
        return pd.DataFrame(), "Governance workbook found but firm/year identifiers were not identified."
    gov["firm_key_raw"] = gov["firm_name_gov"].apply(canonical_company_key)
    mapping = fuzzy_map_keys(gov["firm_key_raw"], target_firm_keys, cutoff=0.78)
    gov["firm_key"] = gov["firm_key_raw"].map(mapping)
    for c in ["year", "board_size", "board_independence", "ownership_concentration", "gender_diversity", "board_meetings"]:
        if c in gov.columns:
            gov[c] = to_numeric_clean(gov[c])
    component_cols = {k: k for k in ["board_size", "board_independence", "ownership_concentration", "gender_diversity", "board_meetings"] if k in gov.columns}
    if not component_cols:
        return pd.DataFrame(), "Governance workbook found but component columns were not identified."
    gov = construct_governance_index(gov, component_cols=component_cols, reverse_components=[], index_col="CGQ")
    keep = ["firm_key", "year", "CGQ"] + list(component_cols.values())
    gov2 = gov[keep].dropna(subset=["firm_key", "year"]).drop_duplicates(["firm_key", "year"])
    msg = f"Governance workbook processed. Matched {gov2['firm_key'].nunique()} unique accounting firms and {len(gov2)} firm-year rows."
    return gov2, msg


def parse_amihud_wide_excel(path: Path, sheet_name: str | None = "Amihud") -> pd.DataFrame:
    """Parse the wide Amihud workbook: every firm occupies Date/Price/Vol./Change% columns."""
    raw = pd.read_excel(path, sheet_name=sheet_name or "Amihud", header=None, engine="openpyxl")
    rows = []
    ncols = raw.shape[1]
    # The uploaded workbook starts at column B: Date, Price, Vol., Change% for firm in row 1, col C.
    for start in range(1, ncols, 4):
        firm_name = raw.iat[0, start + 1] if start + 1 < ncols else None
        if not isinstance(firm_name, str) or not firm_name.strip():
            continue
        block = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.iloc[2:, start], errors="coerce"),
                "price": pd.to_numeric(raw.iloc[2:, start + 1], errors="coerce") if start + 1 < ncols else np.nan,
                "volume": raw.iloc[2:, start + 2] if start + 2 < ncols else np.nan,
                "daily_return": pd.to_numeric(raw.iloc[2:, start + 3], errors="coerce") if start + 3 < ncols else np.nan,
            }
        )
        block["firm_name_market"] = firm_name.strip()
        block["firm_ticker"] = extract_ticker(firm_name)
        block["firm_key"] = canonical_company_key(firm_name)
        rows.append(block)
    if not rows:
        return pd.DataFrame()
    daily = pd.concat(rows, ignore_index=True)
    daily = daily.dropna(subset=["date", "price", "daily_return"])
    daily["volume_units"] = daily["volume"].map(parse_volume)
    daily["trading_value"] = daily["price"] * daily["volume_units"]
    daily["year"] = daily["date"].dt.year
    daily = daily.replace([np.inf, -np.inf], np.nan)
    daily = daily[(daily["trading_value"] > 0) & daily["daily_return"].notna()].copy()
    daily["amihud_raw_daily"] = daily["daily_return"].abs() / daily["trading_value"]
    out = (
        daily.groupby(["firm_ticker", "firm_key", "firm_name_market", "year"], dropna=False)
        .agg(
            AMIHUD_raw=("amihud_raw_daily", "mean"),
            n_trading_days=("amihud_raw_daily", "count"),
            mean_price=("price", "mean"),
            mean_trading_value=("trading_value", "mean"),
        )
        .reset_index()
    )
    out["AMIHUD"] = out["AMIHUD_raw"] * 1_000_000_000  # scaled Amihud for readable coefficients
    return out


def main() -> None:
    cfg = load_config()
    warnings: list[str] = []

    main_path = resolve_path(cfg["paths"]["diversification_excel"], cfg)
    gov_path = resolve_path(cfg["paths"]["governance_excel"], cfg)
    out_path = resolve_path(cfg["paths"]["processed_panel_csv"], cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_tables = resolve_path(cfg["paths"]["outputs_tables"], cfg)
    output_tables.mkdir(parents=True, exist_ok=True)

    main_sheet = cfg["sheet_names"].get("main_panel")
    df = read_table(main_path, sheet_name=main_sheet)
    df = canonicalize_main_columns(df)

    if "firm_id" not in df.columns:
        if "firm_name" in df.columns:
            df["firm_id"] = df["firm_name"].apply(canonical_company_key)
            warnings.append("Created firm_id from firm_name/company_name.")
        else:
            df["firm_id"] = df.index.astype(str)
            warnings.append("No firm identifier found; created firm_id from row index. Please fix variable_map.json.")
    if "year" not in df.columns:
        warnings.append("No year column found. Fixed-effect analysis requires a year column.")

    # Construct diversification.
    div_cfg = cfg["research_variables"]["diversification"]
    hhi_col = resolve_col(df, [div_cfg.get("hhi_column"), "HHI", "Herfindahl", "Herfindahl Index"])
    div_col = resolve_col(df, [div_cfg.get("preferred_column"), "DIV", "Diversification"])
    try:
        df = construct_diversification(
            df,
            hhi_col=hhi_col,
            div_col=div_col or div_cfg.get("preferred_column", "DIV"),
            segment_sales_cols=div_cfg.get("sales_segment_columns", []),
        )
    except Exception:
        df, note = add_count_based_diversification_proxy(df)
        if note:
            warnings.append(note)
        else:
            warnings.append("Diversification not constructed: Provide DIV, HHI, segment sales columns, or branch/production-line counts.")

    # Construct earnings quality.
    eq_cfg = cfg["research_variables"]["earnings_quality"]
    eq_col = resolve_col(df, [eq_cfg.get("preferred_column"), "EQ", "Earnings Quality"])
    da_col = resolve_col(df, [eq_cfg.get("discretionary_accruals_column"), "DA", "DAC", "AEM"])
    em_col = resolve_col(df, [eq_cfg.get("existing_earnings_management_column"), "EAR", "EM", "Earnings Management"])
    try:
        df = construct_earnings_quality(df, eq_col=eq_col or "EQ", da_col=da_col, em_col=em_col)
    except Exception:
        df, note = add_modified_jones_eq(df)
        if note:
            warnings.append(note)
        else:
            warnings.append("Earnings quality not constructed: Provide EQ/DA/EAR or columns needed for Modified Jones accrual model.")

    # Merge governance variables.
    if gov_path.exists():
        try:
            gov2, gov_msg = build_governance_panel(gov_path, df["firm_key"] if "firm_key" in df.columns else pd.Series(dtype=str))
            warnings.append(gov_msg)
            if not gov2.empty and {"firm_key", "year"}.issubset(df.columns):
                df = df.merge(gov2, how="left", on=["firm_key", "year"], suffixes=("", "_gov"))
                warnings.append(f"Governance values merged into main panel for {int(df['CGQ'].notna().sum())} firm-year observations.")
        except Exception as e:
            warnings.append(f"Governance merge skipped due to error: {e}")
    else:
        warnings.append("Governance workbook not found.")

    # Merge Amihud from wide daily trading file if available.
    daily_path = resolve_path(cfg["paths"].get("daily_trading_excel_or_csv", "data/raw/egx_daily_trading_optional.xlsx"), cfg)
    if daily_path.exists():
        try:
            daily_sheet = cfg["sheet_names"].get("daily_trading") or "Amihud"
            amihud = parse_amihud_wide_excel(daily_path, sheet_name=daily_sheet)
            if amihud.empty:
                warnings.append("Amihud workbook was found, but no daily market observations were parsed.")
            else:
                # Merge by ticker first, then fill missing using a conservative firm-key merge.
                df["AMIHUD"] = np.nan
                df["AMIHUD_raw"] = np.nan
                df["n_trading_days"] = np.nan
                by_ticker = amihud.dropna(subset=["firm_ticker"])[["firm_ticker", "year", "AMIHUD", "AMIHUD_raw", "n_trading_days"]].drop_duplicates(["firm_ticker", "year"])
                if "firm_ticker" in df.columns:
                    df = df.merge(by_ticker, how="left", on=["firm_ticker", "year"], suffixes=("", "_ticker"))
                    for col in ["AMIHUD", "AMIHUD_raw", "n_trading_days"]:
                        if f"{col}_ticker" in df.columns:
                            df[col] = df[col].combine_first(df[f"{col}_ticker"])
                            df = df.drop(columns=[f"{col}_ticker"])
                by_key = amihud[["firm_key", "year", "AMIHUD", "AMIHUD_raw", "n_trading_days"]].drop_duplicates(["firm_key", "year"])
                if "firm_key" in df.columns:
                    df = df.merge(by_key, how="left", on=["firm_key", "year"], suffixes=("", "_key"))
                    for col in ["AMIHUD", "AMIHUD_raw", "n_trading_days"]:
                        if f"{col}_key" in df.columns:
                            df[col] = df[col].combine_first(df[f"{col}_key"])
                            df = df.drop(columns=[f"{col}_key"])
                matched = int(df["AMIHUD"].notna().sum())
                warnings.append(f"Amihud workbook processed. AMIHUD matched for {matched} firm-year observations.")
        except Exception as e:
            warnings.append(f"Amihud merge skipped due to error: {e}")
    else:
        warnings.append("Daily trading data not found; Amihud illiquidity cannot be constructed yet. H2-H6 will be skipped until this file is added.")

    # Construct optional controls.
    if "sales" in df.columns and "firm_id" in df.columns and "year" in df.columns:
        df = df.sort_values(["firm_id", "year"])
        lag_sales = df.groupby("firm_id")["sales"].shift(1)
        df["Sales Growth"] = np.where(lag_sales > 0, (df["sales"] - lag_sales) / lag_sales, np.nan)
        df["Sales Growth"] = pd.to_numeric(df["Sales Growth"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Winsorize numeric analysis columns.
    limits = cfg["analysis_options"].get("winsorize_limits", [0.01, 0.99])
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = winsorize_dataframe(df, numeric_cols, limits=limits)

    # Remove rows without firm/year where possible.
    if "year" in df.columns:
        df = df[df["year"].notna()].copy()
        df["year"] = df["year"].astype(int)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved processed panel: {out_path}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Available analysis columns:", [c for c in ["firm_id", "firm_name", "year", "DIV", "EQ", "DA", "AMIHUD", "CGQ", "ROA", "ROE", "size", "age", "Leverage", "Sales Growth", "n_trading_days"] if c in df.columns])

    warning_path = output_tables / "preparation_warnings.txt"
    warning_path.write_text("\n".join(warnings) if warnings else "No warnings.", encoding="utf-8")
    print(f"Saved warnings: {warning_path}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
