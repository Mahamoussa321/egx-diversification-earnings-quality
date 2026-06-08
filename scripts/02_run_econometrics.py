from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_config, resolve_path
from src.econometrics import correlation_table, descriptive_table, fit_ols_fe, model_stats, tidy_result, vif_table
from src.moderated_mediation import bootstrap_moderated_mediation


def present(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def main() -> None:
    cfg = load_config()
    panel_path = resolve_path(cfg["paths"]["processed_panel_csv"], cfg)
    output_dir = resolve_path(cfg["paths"]["outputs_tables"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not panel_path.exists():
        raise FileNotFoundError(f"Processed panel not found: {panel_path}. Run scripts/01_prepare_panel.py first.")

    df = pd.read_csv(panel_path)

    firm_col = "firm_id" if "firm_id" in df.columns else cfg["id_columns"].get("firm_id", "firm_id")
    year_col = "year" if "year" in df.columns else cfg["id_columns"].get("year", "year")
    industry_col = cfg["id_columns"].get("industry", "industry")

    y = cfg["research_variables"]["earnings_quality"].get("preferred_column", "EQ")
    x = cfg["research_variables"]["diversification"].get("preferred_column", "DIV")
    mediator = cfg["research_variables"]["information_asymmetry"].get("preferred_column", "AMIHUD")
    moderator = cfg["research_variables"]["corporate_governance"].get("preferred_column", "CGQ")

    control_map = cfg["control_variables"]
    controls = present(
        df,
        [
            control_map.get("firm_size", "Firm Size"),
            control_map.get("leverage", "Leverage"),
            control_map.get("firm_age", "Firm Age"),
            control_map.get("roa", "ROA"),
            control_map.get("roe", "ROE"),
            control_map.get("sales_growth", "Sales Growth"),
            control_map.get("cash_flow", "Cash Flow"),
            "size",
            "Leverage",
            "lev",
            "age",
            "ROA",
            "ROE",
        ],
    )
    # Deduplicate controls while preserving order.
    controls = list(dict.fromkeys(controls))

    key_vars = [y, x, mediator, moderator] + controls
    desc = descriptive_table(df, key_vars)
    desc.to_csv(output_dir / "table_01_descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    corr = correlation_table(df, key_vars)
    corr.to_csv(output_dir / "table_02_correlations.csv", encoding="utf-8-sig")

    if len(controls) >= 2:
        try:
            vif = vif_table(df, [x] + controls)
            vif.to_csv(output_dir / "table_03_vif.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            (output_dir / "table_03_vif_error.txt").write_text(str(e), encoding="utf-8")

    rows = []
    stats = []
    notes = []
    cluster = cfg["analysis_options"].get("cluster_by", firm_col)
    if cluster not in df.columns:
        cluster = firm_col if firm_col in df.columns else None

    firm_fe = firm_col if firm_col in df.columns else None
    year_fe = year_col if year_col in df.columns else None
    industry_fe = None if firm_fe else (industry_col if industry_col in df.columns else None)

    # H1 direct effect.
    if y in df.columns and x in df.columns:
        try:
            result, formula, _ = fit_ols_fe(df, y, [x] + controls, "H1_Direct", firm_fe=firm_fe, year_fe=year_fe, industry_fe=industry_fe, cluster_col=cluster)
            rows.append(tidy_result(result, "H1_Direct", keep_vars=[x] + controls))
            stats.append(model_stats(result, "H1_Direct"))
            notes.append(f"H1 formula: {formula}")
        except Exception as e:
            notes.append(f"H1 could not be estimated: {e}")
    else:
        notes.append(f"H1 skipped because required variables are missing: {y}, {x}")

    # H2 first-stage effect.
    if mediator in df.columns and x in df.columns:
        try:
            result, formula, _ = fit_ols_fe(df, mediator, [x] + controls, "H2_FirstStage", firm_fe=firm_fe, year_fe=year_fe, industry_fe=industry_fe, cluster_col=cluster)
            rows.append(tidy_result(result, "H2_FirstStage", keep_vars=[x] + controls))
            stats.append(model_stats(result, "H2_FirstStage"))
            notes.append(f"H2 formula: {formula}")
        except Exception as e:
            notes.append(f"H2 could not be estimated: {e}")
    else:
        notes.append(f"H2 skipped because mediator/information-asymmetry variable is missing: {mediator}")

    # H3-H5 outcome with moderation.
    interaction = f"{mediator}_x_{moderator}"
    if all(v in df.columns for v in [y, x, mediator, moderator]):
        df[interaction] = (df[mediator] - df[mediator].mean()) * (df[moderator] - df[moderator].mean())
        try:
            xvars = [x, mediator, moderator, interaction] + controls
            result, formula, _ = fit_ols_fe(df, y, xvars, "H3_H5_OutcomeModeration", firm_fe=firm_fe, year_fe=year_fe, industry_fe=industry_fe, cluster_col=cluster)
            rows.append(tidy_result(result, "H3_H5_OutcomeModeration", keep_vars=xvars))
            stats.append(model_stats(result, "H3_H5_OutcomeModeration"))
            notes.append(f"H3-H5 formula: {formula}")
        except Exception as e:
            notes.append(f"H3-H5 could not be estimated: {e}")
    else:
        notes.append(f"H3-H5 skipped because one or more required variables are missing: {y}, {x}, {mediator}, {moderator}")

    # Save fitted H1-H5 tables before any optional bootstrap step.
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(output_dir / "table_04_regression_results.csv", index=False, encoding="utf-8-sig")
    if stats:
        pd.concat(stats, ignore_index=True).to_csv(output_dir / "table_04_model_statistics.csv", index=False, encoding="utf-8-sig")

    # H6 conditional indirect effect. Bootstrap can be slow with many fixed effects, so it is optional.
    if all(v in df.columns for v in [y, x, mediator, moderator, firm_col, year_col]):
        n_boot = int(cfg["analysis_options"].get("bootstrap_iterations", 0))
        complete_n = int(df.dropna(subset=[y, x, mediator, moderator, firm_col, year_col] + controls).shape[0])
        if n_boot <= 0:
            notes.append("H6 bootstrap skipped because bootstrap_iterations is set to 0. Use a positive value after confirming the final analysis sample.")
        elif complete_n < 50:
            notes.append(f"H6 bootstrap skipped because complete-case sample is small (N={complete_n}).")
        else:
            try:
                seed = int(cfg["analysis_options"].get("random_seed", 2026))
                summary, draws = bootstrap_moderated_mediation(
                    df,
                    y=y,
                    x=x,
                    mediator=mediator,
                    moderator=moderator,
                    controls=controls,
                    firm_col=firm_col,
                    year_col=year_col,
                    n_boot=n_boot,
                    seed=seed,
                    cluster_col=cluster,
                )
                summary.to_csv(output_dir / "table_05_bootstrap_conditional_indirect_effects.csv", index=False, encoding="utf-8-sig")
                draws.to_csv(output_dir / "bootstrap_draws_conditional_indirect_effects.csv", index=False, encoding="utf-8-sig")
                notes.append("H6 bootstrap conditional indirect effects saved.")
            except Exception as e:
                notes.append(f"H6 bootstrap could not be estimated: {e}")
    else:
        notes.append(f"H6 skipped because mediation/moderation variables or firm/year identifiers are missing.")

    (output_dir / "analysis_notes.txt").write_text("\n".join(notes), encoding="utf-8")
    print("Analysis complete. See outputs/tables/.")
    for note in notes:
        print(f"- {note}")


if __name__ == "__main__":
    main()
