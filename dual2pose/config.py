"""Single entry point for loading the repository's Hydra configuration.

`OmegaConf.load` reads a YAML file verbatim and ignores its `defaults:` list,
so a config loaded that way is missing every group it selects -- most notably
`model`, which holds the model hyperparameters. Scripts that are not themselves
`@hydra.main` entry points must therefore compose through Hydra instead, which
is what `load_config` does.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual2pose.yaml"


def load_config(
    config_path: str | Path = DEFAULT_CONFIG,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose the config at `config_path`, resolving its defaults list."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    # An enclosing @hydra.main has already composed its own config by the time a
    # script body runs, so releasing the global instance here is safe.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.name, overrides=overrides or [])

    # Compose returns a struct config, which rejects new keys. Callers such as
    # the evaluation scripts attach dataset-specific entries (data.left_hip_idx
    # and friends) at runtime, which is what OmegaConf.load used to allow, so
    # keep that behaviour.
    OmegaConf.set_struct(config, False)
    return config
