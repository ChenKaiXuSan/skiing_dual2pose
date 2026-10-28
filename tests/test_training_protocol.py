"""The fixed CanonFuse3D training protocol: per-frame person centering, supervised L1.

Break caught: the trainer canonicalized every batch with the first frame of its
first sequence, so the representation depended on batch composition and carried
the global motion of each sequence; checkpoints of that protocol could be fed a
representation they were never trained on.
"""
import math
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dual2pose.trainer.canonicalize import canonicalize_pose_torch, validate_canonical_mode
from dual2pose.trainer.train_crossview_fusion import CANONICAL_MODE, CrossViewFusionTrainer

LEFT_HIP, RIGHT_HIP, NECK = 6, 7, 14


def _pose_batch(batch=3, frames=8, joints=15, seed=0):
    gen = torch.Generator().manual_seed(seed)
    base = 0.3 * torch.randn(joints, 3, generator=gen)
    base[LEFT_HIP] = torch.tensor([-0.1, 0.0, 0.0])
    base[RIGHT_HIP] = torch.tensor([0.1, 0.0, 0.0])
    base[NECK] = torch.tensor([0.0, 0.5, 0.0])
    base[0] = torch.tensor([-0.03, 0.65, 0.1])
    base[1] = torch.tensor([0.03, 0.65, 0.1])
    return base.expand(batch, frames, joints, 3).clone() + 0.02 * torch.randn(
        batch, frames, joints, 3, generator=gen
    )


def _rigid_per_frame(poses, seed=1):
    """Apply a different rotation and translation to every frame of every sequence."""
    gen = torch.Generator().manual_seed(seed)
    out = poses.clone()
    for b in range(poses.shape[0]):
        for t in range(poses.shape[1]):
            angles = 2 * math.pi * torch.rand(3, generator=gen)
            cx, cy, cz = torch.cos(angles)
            sx, sy, sz = torch.sin(angles)
            rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            rotation = rz @ ry @ rx
            translation = 3.0 * torch.randn(3, generator=gen)
            out[b, t] = poses[b, t] @ rotation.T + translation
    return out


def _config(**arch):
    return OmegaConf.create(
        {
            "model": {
                "backbone": "crossview_fusion",
                "arch": {"hidden_dim": 32, "num_heads": 2, "dropout": 0.0, "temporal_kernel_size": 3, **arch},
                "optim": {"lr": 1e-3, "weight_decay": 0.0},
                "loss": {},
            },
            "log_path": "/tmp/unused",
        }
    )


def _trainer(**arch):
    model = CrossViewFusionTrainer(_config(**arch))
    model.log = lambda *args, **kwargs: None  # no Trainer attached
    return model.eval()


def _batch(left, right, gt=None):
    batch = {"kpt3d_sam": {"cam1": left, "cam2": right}}
    if gt is not None:
        batch["kpt3d_gt"] = gt
    return batch


class CanonicalizeModeTest(unittest.TestCase):
    def test_per_frame_mode_removes_per_frame_rigid_motion(self):
        poses = _pose_batch()
        moved = _rigid_per_frame(poses)
        still, _ = canonicalize_pose_torch(poses, mode="per_frame")
        canon, _ = canonicalize_pose_torch(moved, mode="per_frame")
        torch.testing.assert_close(canon, still, atol=1e-4, rtol=0)
        # the archived first-frame mode keeps the motion relative to its reference frame
        first, _ = canonicalize_pose_torch(moved, mode="first_frame")
        self.assertGreater((first - still).abs().max().item(), 1.0)

    def test_per_frame_mode_does_not_depend_on_the_batch(self):
        poses = _pose_batch()
        together, _ = canonicalize_pose_torch(poses, mode="per_frame")
        alone, _ = canonicalize_pose_torch(poses[1:2], mode="per_frame")
        torch.testing.assert_close(together[1:2], alone, atol=0, rtol=0)
        # archived behaviour, documented: the batch shares its first sequence's first frame
        together_ff, _ = canonicalize_pose_torch(poses, mode="first_frame")
        alone_ff, _ = canonicalize_pose_torch(poses[1:2], mode="first_frame")
        self.assertGreater((together_ff[1:2] - alone_ff).abs().max().item(), 1e-3)

    def test_per_frame_transform_matches_input_layout(self):
        poses = _pose_batch(batch=2, frames=5)
        canon, transform = canonicalize_pose_torch(poses, mode="per_frame")
        self.assertEqual(tuple(transform["pelvis"].shape), (2, 5, 3))
        self.assertEqual(tuple(transform["R"].shape), (2, 5, 3, 3))
        pelvis = 0.5 * (canon[..., LEFT_HIP, :] + canon[..., RIGHT_HIP, :])
        torch.testing.assert_close(pelvis, torch.zeros_like(pelvis), atol=1e-6, rtol=0)
        single, transform_single = canonicalize_pose_torch(poses[0, 0], mode="per_frame")
        self.assertEqual(tuple(single.shape), (15, 3))
        self.assertEqual(tuple(transform_single["R"].shape), (3, 3))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_canonical_mode("per_sequence")
        with self.assertRaises(ValueError):
            canonicalize_pose_torch(_pose_batch(), mode="per_sequence")


class TrainingProtocolTest(unittest.TestCase):
    def test_protocol_is_per_frame_and_not_configurable(self):
        self.assertEqual(CANONICAL_MODE, "per_frame")
        self.assertEqual(_trainer().canonical_mode, "per_frame")
        self.assertEqual(_trainer(canonical_mode="per_frame").canonical_mode, "per_frame")
        with self.assertRaises(ValueError):
            _trainer(canonical_mode="first_frame")

    def test_repository_config_builds_the_protocol(self):
        from dual2pose.config import load_config

        config = load_config()
        self.assertNotIn("canonical_mode", config.model.arch)
        self.assertEqual(CrossViewFusionTrainer(config).canonical_mode, "per_frame")

    def test_training_step_ignores_per_frame_person_motion(self):
        left, right, gt = _pose_batch(seed=3), _pose_batch(seed=4), _pose_batch(seed=5)
        model = _trainer()
        with torch.no_grad():
            batch = _batch(left, right, gt)
            loss = model._shared_step(batch, stage="train")
            fused = batch["_variant_results"]["fused"]
            moved = _batch(_rigid_per_frame(left, 7), _rigid_per_frame(right, 8), _rigid_per_frame(gt, 9))
            loss_moved = model._shared_step(moved, stage="train")
            fused_moved = moved["_variant_results"]["fused"]
        torch.testing.assert_close(loss_moved, loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(fused_moved, fused, atol=1e-3, rtol=1e-3)
        # the supervision target is centred in every frame
        target = batch["_variant_results"]["ground_truth_canonical"]
        target_pelvis = 0.5 * (target[..., LEFT_HIP, :] + target[..., RIGHT_HIP, :])
        torch.testing.assert_close(target_pelvis, torch.zeros_like(target_pelvis), atol=1e-6, rtol=0)

    def test_step_does_not_depend_on_batch_order(self):
        left, right, gt = _pose_batch(seed=3), _pose_batch(seed=4), _pose_batch(seed=5)
        model = _trainer()
        with torch.no_grad():
            batch = _batch(left, right, gt)
            model._shared_step(batch, stage="val")
            flipped = _batch(left.flip(0), right.flip(0), gt.flip(0))
            model._shared_step(flipped, stage="val")
        torch.testing.assert_close(
            flipped["_variant_results"]["fused"].flip(0), batch["_variant_results"]["fused"], atol=1e-5, rtol=1e-5
        )

    def test_training_and_validation_require_ground_truth(self):
        model = _trainer()
        unlabeled = _batch(_pose_batch(seed=3), _pose_batch(seed=4))
        for stage in ("train", "val"):
            with self.assertRaises(RuntimeError):
                model._shared_step(dict(unlabeled), stage=stage)
        # test-time inference on unlabeled recordings still runs
        with torch.no_grad():
            loss = model._shared_step(unlabeled, stage="test")
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(unlabeled["_variant_results"]["ground_truth_canonical"])
        self.assertEqual(tuple(unlabeled["_variant_results"]["fused"].shape), (3, 8, 15, 3))

    def test_canonicalize_method_uses_the_protocol_and_the_model_indices(self):
        poses = _pose_batch()
        expected, _ = canonicalize_pose_torch(poses, left_hip=6, right_hip=7, neck=14, mode="per_frame")
        torch.testing.assert_close(_trainer().canonicalize(poses)[0], expected, atol=0, rtol=0)


class CheckpointProtocolTest(unittest.TestCase):
    def test_saved_checkpoints_record_the_protocol(self):
        model = _trainer()
        checkpoint = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["canonical_mode"], "per_frame")
        model.on_load_checkpoint(checkpoint)  # accepted

    def test_first_frame_checkpoints_are_rejected(self):
        model = _trainer()
        with self.assertRaises(ValueError):
            model.on_load_checkpoint({"state_dict": model.state_dict()})  # archived manuscript checkpoint
        with self.assertRaisesRegex(ValueError, "legacy.ckpt"):
            CrossViewFusionTrainer.check_checkpoint_protocol({"canonical_mode": "first_frame"}, source="legacy.ckpt")

    def test_pocket3_loader_enforces_the_protocol(self):
        from dual2pose.config import load_config
        from dual2pose.eval import eval_pocket3 as ep

        source = CrossViewFusionTrainer(load_config())  # same architecture as load_model builds
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.ckpt"
            torch.save({"state_dict": source.state_dict()}, legacy)
            with self.assertRaises(ValueError):
                ep.load_model(legacy, torch.device("cpu"))
            current = Path(directory) / "per_frame.ckpt"
            torch.save({"state_dict": source.state_dict(), "canonical_mode": "per_frame"}, current)
            model = ep.load_model(current, torch.device("cpu"))
        self.assertEqual(model.canonical_mode, "per_frame")


if __name__ == "__main__":
    unittest.main()
