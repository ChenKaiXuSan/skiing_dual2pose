"""Regressions for the active per-frame training and evaluation pipelines."""
import subprocess
import sys

import numpy as np
import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch.utils.data import DataLoader

from dual2pose.pocket3 import evaluate as ep
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
from tests.test_pocket3 import _FakeFusion
from tests.test_training_protocol import _config, _pose_batch, _rigid_per_frame


def test_default_canonicalization_is_per_frame():
    pose = _pose_batch()
    expected = canonicalize_pose_torch(pose, mode="per_frame")[0]
    torch.testing.assert_close(canonicalize_pose_torch(pose)[0], expected)
    torch.testing.assert_close(canonicalize_pose_torch(_rigid_per_frame(pose))[0], expected, atol=1e-4, rtol=0)


def test_pocket_baselines_do_not_gain_error_from_camera_motion():
    gt = _pose_batch(batch=2, frames=6)
    left, right = _rigid_per_frame(gt, 3), _rigid_per_frame(gt, 4)
    result = ep.run_methods(left.numpy(), right.numpy(), gt.numpy(), {"average": _FakeFusion()}, torch.device("cpu"))
    target = canonicalize_pose_torch(gt, mode="per_frame")[0].numpy()[..., ep.COMMON13, :]
    for name in ("left", "right", "canonical_average", "aligned_average", "canonfuse3d[average]", "gt"):
        np.testing.assert_allclose(result[name], target, atol=1e-4)


def test_pocket_adaptation_uses_the_same_per_frame_baseline():
    from dual2pose.pocket3.adapt import canon

    pose = _rigid_per_frame(_pose_batch())
    torch.testing.assert_close(canon(pose.numpy(), torch.device("cpu")), canonicalize_pose_torch(pose, mode="per_frame")[0])


def test_pocket_baselines_honor_saved_canonicalization_settings():
    from dual2pose.pocket3 import adapt

    class NoFaceAverage(_FakeFusion):
        face_sign = False

        def canonicalize(self, x):
            return canonicalize_pose_torch(x, mode="per_frame", enforce_face_z_positive=False)

        def eval(self):
            return self

    pose = _pose_batch(batch=1, frames=6)
    pose[..., 2] *= -1  # default face-sign correction would reflect the z axis
    model = NoFaceAverage()
    out = ep.run_methods(pose.numpy(), pose.numpy(), pose.numpy(), {"no_face": model}, "cpu")
    expected = ep.comparison_canonical(pose)[0].numpy()[..., ep.COMMON13, :]
    for key in ("canonical_average", "gt", "canonfuse3d[no_face]"):
        np.testing.assert_allclose(out[key], expected, atol=1e-6)
    np.testing.assert_allclose(ep.per_frame_canonical(pose.numpy(), model=model), model.canonicalize(pose)[0].numpy(), atol=1e-6)
    torch.testing.assert_close(adapt.canon(pose.numpy(), "cpu", model=model), model.canonicalize(pose)[0])
    assert model.face_sign is False
    scores = adapt.evaluate(model, pose.numpy(), pose.numpy(), pose.numpy(),
                            np.ones(pose.shape[:-1], dtype=bool), "cpu")
    assert scores["canonical_average"]["per_frame_canonical_mpjpe_mm"] < 1e-3
    assert scores["model"]["per_frame_canonical_mpjpe_mm"] < 1e-3


def test_pocket_rejects_incompatible_checkpoint_coordinate_frames():
    pose = _pose_batch(batch=1).numpy()
    default = CrossViewFusionTrainer(_config()).eval()
    different = CrossViewFusionTrainer(_config(data={"face_sign": False})).eval()
    ski = CrossViewFusionTrainer(_config(data={"left_hip_idx": 4, "right_hip_idx": 5, "neck_idx": 12})).eval()
    ep.run_methods(pose, pose, pose, {"default": default, "different": different}, "cpu")
    for models in ({"ski": ski},):
        with pytest.raises(ValueError, match="canonicalization|skeleton"):
            ep.run_methods(pose, pose, pose, models, "cpu")


def test_pocket_cli_cannot_select_a_first_frame_protocol():
    assert ep.parse_args([]).reference == "per_frame"
    for mode in ("window", "batch", "first_frame"):
        with pytest.raises(SystemExit):
            ep.parse_args(["--reference", mode])


@pytest.mark.parametrize("entrypoint", ["nview", "temporal", "direct", "pocket"])
@pytest.mark.parametrize("recorded", [None, "first_frame"])
def test_active_checkpoint_loaders_reject_legacy_weights(tmp_path, entrypoint, recorded):
    from dual2pose.config import load_config
    from dual2pose.eval.eval_unity_nview import _load_model as nview
    from dual2pose.eval.eval_unity_temporal_alignment import _load_model as temporal
    from dual2pose.realworld.direct import load_model as direct

    config = load_config()
    model = CrossViewFusionTrainer(config)
    checkpoint = {"state_dict": model.state_dict()}
    if recorded is not None:
        checkpoint["canonical_mode"] = recorded
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)
    loaders = {
        "nview": lambda: nview(config, path, torch.device("cpu")),
        "temporal": lambda: temporal(config, path, torch.device("cpu")),
        "direct": lambda: direct(path),
        "pocket": lambda: ep.load_model(path, torch.device("cpu")),
    }
    with pytest.raises(ValueError, match="per-frame protocol"):
        loaders[entrypoint]()


def test_pocket_saved_weights_round_trip_configuration_and_predictions(tmp_path):
    from dual2pose.pocket3.train import save_weights

    model = CrossViewFusionTrainer(_config(data={"face_sign": False})).eval()
    path = tmp_path / "selected.pt"
    save_weights(model, path)
    restored = ep.load_model(path, torch.device("cpu"))
    assert restored.canonical_mode == "per_frame"
    assert restored.face_sign is False
    left, right = _pose_batch(seed=7), _pose_batch(seed=8)
    with torch.no_grad():
        expected = model.models(*ep.model_inputs(model, left, right))[0]
        actual = restored.models(*ep.model_inputs(restored, left, right))[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_lightning_checkpoint_loads_in_a_fresh_process(tmp_path):
    from dual2pose.config import load_config

    config = load_config()  # includes the training-time ${now:...} log directory
    model = CrossViewFusionTrainer(config)
    path = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": model.state_dict(), "canonical_mode": "per_frame",
                "hyper_parameters": {"hparams": config}}, path)
    code = ("import sys; from pathlib import Path; "
            "from dual2pose.pocket3.evaluate import load_model; "
            "m = load_model(Path(sys.argv[1]), 'cpu'); "
            "assert m.canonical_mode == 'per_frame'")
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cosine_scheduler_covers_optimizer_steps_with_accumulation(tmp_path):
    class LearningRates(Callback):
        def __init__(self):
            self.values = []

        def on_train_epoch_end(self, trainer, pl_module):
            self.values.append(trainer.optimizers[0].param_groups[0]["lr"])

    model = CrossViewFusionTrainer(_config())
    poses = _pose_batch(batch=4, frames=4)
    batches = [{"kpt3d_sam": {"cam1": p, "cam2": p}, "kpt3d_gt": p} for p in poses]
    rates = LearningRates()
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=2, accumulate_grad_batches=2,
                      logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, callbacks=[rates], default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=1))
    assert trainer.global_step == 4
    assert rates.values == pytest.approx([model.lr / 2, 0.0], abs=1e-12)
