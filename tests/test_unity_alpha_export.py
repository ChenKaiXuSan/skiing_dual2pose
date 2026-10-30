"""The module entrypoints must export gate plots without a top-level map_config."""
import csv

import pytest
import torch

from dual2pose.eval import eval_unity_masking, eval_unity_temporal_offset
from dual2pose.map_config import FILTERED_KPTS_MAPPING


@pytest.mark.parametrize("helpers", [eval_unity_masking, eval_unity_temporal_offset._load_eval_helpers()], ids=["masking", "temporal-offset"])
def test_package_entrypoint_exports_alpha_artifacts(tmp_path, helpers):
    names = list(FILTERED_KPTS_MAPPING.values())
    values = torch.arange(len(names), dtype=torch.float32) / len(names)
    alpha = values[None, None, :, None].expand(2, 3, -1, 1).clone()
    helpers._export_alpha_visualization([{"alpha": alpha}], tmp_path, "clean")

    directory = tmp_path / "alpha_vis"
    with (directory / "alpha_summary_clean.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["joint_name"] for row in rows] == names
    assert [float(row["alpha_mean"]) for row in rows] == pytest.approx(values.tolist())
    assert [float(row["right_mean"]) for row in rows] == pytest.approx((1-values).tolist())
    for name in ("alpha_joint_bar_clean.png", "alpha_time_joint_heatmap_clean.png"):
        assert (directory / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("entrypoint", [eval_unity_masking, eval_unity_temporal_offset], ids=["masking", "temporal-offset"])
def test_summary_csv_declares_direct_per_frame_protocol(tmp_path, monkeypatch, entrypoint):
    """Exercise CSV creation through the entrypoint without loading a dataset."""
    from omegaconf import OmegaConf

    monkeypatch.setenv("EVAL_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(entrypoint, "evaluation_checkpoint_from_env", lambda: tmp_path / "selected.ckpt")
    monkeypatch.setattr(entrypoint, "_build_default_study", lambda **kwargs: [])
    if entrypoint is eval_unity_masking:
        monkeypatch.setattr(entrypoint, "_configure_index_mapping_rewrite", lambda config: None)
    else:
        monkeypatch.setattr(entrypoint, "_patch_index_mapping_path_rewrite", lambda **kwargs: None)
    config = OmegaConf.create({"train": {"gpu": 0, "fold": 0}, "data": {"unity": {"root_path": str(tmp_path)}}})
    entrypoint.init_params.__wrapped__(config)
    summaries = list(tmp_path.glob("*_summary_*.csv"))
    assert len(summaries) == 1
    with summaries[0].open() as handle:
        fields = csv.DictReader(handle).fieldnames
    assert {"canonical_mode", "evaluation_protocol", "output_alignment", "reduction"} <= set(fields)
