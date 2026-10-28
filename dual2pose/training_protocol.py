"""Shared provenance rules for the per-frame training protocol."""

from __future__ import annotations

import os
from pathlib import Path
import re


_FOLD_DIRECTORY = Path(
    "index_mapping/use_layer_camera_filter_disabled/"
    "camera_pairs_by_action_folds"
)


def load_fusion_checkpoint(checkpoint_path: Path, *, config=None, device="cpu"):
    """Load a tagged fusion checkpoint, preserving its saved model configuration.

    Older per-frame Lightning files and manual Pocket3 exports both carry
    ``hyper_parameters.hparams``. A caller configuration (or the repository
    default) is only a fallback for tagged files without saved configuration.
    Bare weights and first-frame checkpoints are rejected before construction.
    """
    from copy import deepcopy
    from omegaconf import OmegaConf
    import torch
    from dual2pose.config import load_config
    from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a dictionary: {path}")
    CrossViewFusionTrainer.check_checkpoint_protocol(payload, source=str(path))
    saved_config = payload.get("hyper_parameters", {}).get("hparams")
    selected = saved_config if saved_config is not None else config
    config = deepcopy(selected) if selected is not None else load_config()
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    # Inference must not resolve a training-time ${now:...} expression (Hydra
    # is not initialized in a fresh CLI process) or reuse an old output path.
    config.log_path = str(path.parent)
    model = CrossViewFusionTrainer(config)
    if "state_dict" not in payload:
        raise ValueError(f"Tagged checkpoint has no state_dict: {path}")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def resolve_fold_index_path(data_root: Path, fold: int) -> Path:
    """Resolve one of the two legacy action folds used by the paper."""

    fold = int(fold)
    if fold not in (0, 1):
        raise ValueError(f"Unsupported fold {fold}; available folds are 0 and 1")
    return Path(data_root) / _FOLD_DIRECTORY / f"fold_{fold:02d}.json"


def validate_fold_metadata(index_path: Path, fold: int) -> None:
    """Validate trailing metadata without loading the multi-gigabyte index."""

    index_path = Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Fold index does not exist: {index_path}")
    with index_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 262_144), os.SEEK_SET)
        tail = handle.read().decode("utf-8-sig")
    metadata_start = tail.rfind('"_metadata"')
    if metadata_start < 0:
        raise ValueError(f"Fold index has no trailing _metadata object: {index_path}")
    match = re.search(r'"fold_idx"\s*:\s*(\d+)', tail[metadata_start:])
    if match is None:
        raise ValueError(f"Fold index metadata has no fold_idx: {index_path}")
    metadata_fold = int(match.group(1))
    if metadata_fold != int(fold):
        raise ValueError(
            f"Requested fold {fold}, but index metadata fold {metadata_fold}: {index_path}"
        )


def evaluation_checkpoint_from_env() -> Path:
    raw = os.environ.get("EVAL_CKPT_PATH")
    if not raw:
        raise ValueError("Set EVAL_CKPT_PATH to an explicit current per-frame checkpoint")
    return Path(raw)
