from __future__ import annotations
import os
import yaml
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load main settings.yaml, expanding environment variables."""
    if path is None:
        path = _ROOT / "config" / "settings.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = f.read()

    # Expand ${VAR} env references
    import re
    def _expand(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))
    raw = re.sub(r"\$\{([^}]+)\}", _expand, raw)

    cfg = yaml.safe_load(raw)
    logger.debug("Config loaded from %s", path)
    return cfg


def get_pairs(cfg: dict) -> list[str]:
    return cfg["pairs"]["majors"] + cfg["pairs"].get("emerging", [])
