from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

scripts = [
    "00_inspect_excel.py",
    "01_prepare_panel.py",
    "02_run_econometrics.py",
]

for script in scripts:
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)

print("\nCore pipeline finished. Run scripts/03_run_ml_robustness.py when the processed panel has enough variables.")
