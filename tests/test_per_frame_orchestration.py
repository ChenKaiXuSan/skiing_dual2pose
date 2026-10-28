from pathlib import Path
import torch
from dual2pose.eval.temporal_alignment import _canonical_pose
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.eval.run_unity_view_angle_significance import validate_protocol_batch_size
from dual2pose import train_ski_poseptz as ski


def test_sync_descriptor_uses_each_frame_body_axes():
    torch.manual_seed(11)
    pose = torch.randn(2, 5, 15, 3)
    expected, _ = canonicalize_pose_torch(pose, mode="per_frame")
    torch.testing.assert_close(_canonical_pose(pose), expected)


def test_angle_accepts_positive_batch_sizes():
    for size in (1, 256, 4096):
        validate_protocol_batch_size(size)


def test_ski_fixed_budget_checkpoint_and_no_training_as_validation(tmp_path):
    checkpoint = ski.build_fixed_budget_checkpoint(tmp_path)
    assert checkpoint.monitor is None
    assert checkpoint.save_top_k == 0
    assert checkpoint.save_last
    assert ski.SkiPosePTZDataModule.val_dataloader(None) == []
    assert Path(ski.DEFAULT_CONFIG).is_file()
