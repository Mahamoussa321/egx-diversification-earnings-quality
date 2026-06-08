from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_config, resolve_path
from src.io_utils import excel_inventory


def main() -> None:
    cfg = load_config()
    output_dir = resolve_path(cfg["paths"]["outputs_tables"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        resolve_path(cfg["paths"]["diversification_excel"], cfg),
        resolve_path(cfg["paths"]["governance_excel"], cfg),
    ]

    inventories = []
    for path in files:
        if path.exists():
            print(f"Inspecting: {path}")
            inventories.append(excel_inventory(path))
        else:
            print(f"Missing file: {path}")

    if inventories:
        inv = pd.concat(inventories, ignore_index=True)
        out = output_dir / "excel_inventory.csv"
        inv.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved inventory: {out}")
        print(inv[["file", "sheet", "n_rows", "n_columns"]].to_string(index=False))
    else:
        raise FileNotFoundError("No Excel files found in data/raw.")


if __name__ == "__main__":
    main()
