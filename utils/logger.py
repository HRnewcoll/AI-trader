from __future__ import annotations
import logging
import logging.config
import yaml
from pathlib import Path
import os

_ROOT = Path(__file__).resolve().parent.parent


def setup_logging() -> None:
    log_cfg_path = _ROOT / "config" / "logging_config.yaml"
    os.makedirs(_ROOT / "logs", exist_ok=True)

    if log_cfg_path.exists():
        with open(log_cfg_path) as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    logging.getLogger("trader").info("Logging initialised")
