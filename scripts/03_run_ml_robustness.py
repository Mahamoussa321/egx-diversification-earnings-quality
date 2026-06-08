from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_config, resolve_path
from src.ml_robustness import run_ml_regression


def main() -> None:
    cfg = load_config()
    panel_path = resolve_path(cfg["paths"]["processed_panel_csv"], cfg)
    output_dir = resolve_path(cfg["paths"]["outputs_tables"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    y = cfg["research_variables"]["earnings_quality"].get("preferred_column", "EQ")
    core = [
        cfg["research_variables"]["diversification"].get("preferred_column", "DIV"),
        cfg["research_variables"]["information_asymmetry"].get("preferred_column", "AMIHUD"),
        cfg["research_variables"]["corporate_governance"].get("preferred_column", "CGQ"),
    ]
    controls = [v for v in cfg["control_variables"].values() if isinstance(v, str)]
    extra_controls = ["size", "lev", "age", "ROA", "ROE"]
    features = list(dict.fromkeys([f for f in core + controls + extra_controls if f in df.columns]))

    if y not in df.columns:
        raise KeyError(f"Outcome variable {y} not found in processed panel.")
    if len(features) < 2:
        raise KeyError("Not enough features found for ML robustness. Check variable_map.json and processed panel.")

    perf, importance = run_ml_regression(df, y, features, seed=int(cfg["analysis_options"].get("random_seed", 2026)))
    perf.to_csv(output_dir / "table_ml_predictive_performance.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output_dir / "table_ml_permutation_importance.csv", index=False, encoding="utf-8-sig")
    print("ML robustness complete. See outputs/tables/.")


if __name__ == "__main__":
    main()
