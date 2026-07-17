"""
Config loading helpers. Wraps OmegaConf so that scripts which don't go
through the full Hydra `@hydra.main` entry point (e.g. inference CLI,
FastAPI startup) can still load and merge the same YAML configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: str = "configs/config.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> DictConfig:
    """Load the master config and resolve its `model`/`dataset`/`training`
    sub-configs manually (mirrors Hydra's `defaults:` composition) for use
    outside of a `@hydra.main` entry point.
    """
    base_dir = Path(config_path).parent
    cfg = OmegaConf.load(config_path)

    for key in ("model", "dataset", "training"):
        default_name = cfg.get("defaults", [{}])
        # Fallback: look for a same-named file under configs/<key>/<key_value>.yaml
        sub_dir = base_dir / key
        selected = None
        for entry in cfg.get("defaults", []):
            if isinstance(entry, dict) and key in entry:
                selected = entry[key]
        if selected is None:
            continue
        sub_path = sub_dir / f"{selected}.yaml"
        if sub_path.exists():
            cfg[key] = OmegaConf.load(sub_path)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def config_to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
