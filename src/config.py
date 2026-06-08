from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    """Return the project root assuming scripts are run from anywhere inside the repo."""
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config" / "variable_map.json").exists():
            return candidate
    raise FileNotFoundError("Could not find config/variable_map.json. Run from inside the project folder.")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    root = project_root()
    path = Path(config_path) if config_path else root / "config" / "variable_map.json"
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_root"] = str(root)
    return cfg


def resolve_path(path_value: str | Path, cfg: Dict[str, Any]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path
