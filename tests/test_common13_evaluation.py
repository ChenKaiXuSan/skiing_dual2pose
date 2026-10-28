"""The active datasets share common-13 and score outputs without extra alignment."""
import numpy as np
import pytest
import torch

from dual2pose.eval.multiseed_metrics import summarize_test_outputs_by_action
from dual2pose.pocket3 import adapt, evaluate as ep
from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
from tests.test_training_protocol import _config, _pose_batch


class ShiftFusion(torch.nn.Module):
    def __init__(self, eyes_only=False):
        super().__init__()
        self.eyes_only = eyes_only

    def forward(self, left, right, **kwargs):
        out = left.clone()
        if self.eyes_only:
            out[..., :2, 0] += 1.0
        else:
            out[..., 0] += 0.1
        return out, {"alpha": torch.full_like(left[..., :1], 0.5),
                     "output_residual": out - left}


def shifted_model(eyes_only=False):
    model = CrossViewFusionTrainer(_config()).eval()
    model.models = ShiftFusion(eyes_only)
    model.log = lambda *args, **kwargs: None
    return model


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_primary_logged_metric_excludes_eyes(stage):
    model = shifted_model(eyes_only=True)
    pose = _pose_batch(batch=1)
    captured = {}
    model.log = lambda key, value, **kwargs: captured.__setitem__(key, float(value))
    batch = {"kpt3d_sam": {"cam1": pose, "cam2": pose}, "kpt3d_gt": pose}
    model._shared_step(batch, stage)
    assert captured[f"{stage}/mpjpe"] == pytest.approx(0.0, abs=1e-7)
    # Native model outputs remain 15 joints; only the evaluation subset changes.
    assert batch["_variant_results"]["fused"].shape[-2] == 15


def test_primary_report_and_per_action_summary_use_common13(tmp_path):
    model = shifted_model(eyes_only=True)
    model.test_save_dir = tmp_path
    pose = _pose_batch(batch=1)
    batch = {"kpt3d_sam": {"cam1": pose, "cam2": pose}, "kpt3d_gt": pose,
             "meta": {"action_id": ["turn"]}}
    model.on_test_start()
    model.test_step(batch, 0)
    model.on_test_epoch_end()
    text = (tmp_path / "report.txt").read_text()
    assert "joint_subset: common13" in text
    assert "fused_mpjpe_mean: 0.0000" in text
    result = summarize_test_outputs_by_action(model.test_outputs)
    assert result["mpjpe"] == pytest.approx(0.0, abs=1e-7)
    assert result["evaluation_protocol"]["output_alignment"] == "none"


def test_unity_ski_and_pocket_direct_scores_agree():
    from dual2pose.eval.eval_ski_poseptz import _summarize_test_outputs

    gt = torch.zeros(2, 4, 15, 3)
    prediction = gt.clone()
    prediction[..., 0] = 0.1
    prediction[..., :2, 0] = 100.0
    unity = CrossViewFusionTrainer._safe_mpjpe(prediction, gt)[1]
    ski = _summarize_test_outputs([{"fused": prediction[..., 2:, :],
                                    "ground_truth_canonical": gt[..., 2:, :]}])["fused"]["mpjpe"]
    pocket = ep.masked_mpjpe(prediction.numpy(), gt.numpy(), np.ones((2, 4, 15), bool), ep.COMMON13)[0]
    assert unity == pytest.approx(ski, abs=1e-7)
    assert unity == pytest.approx(pocket, abs=1e-7)
    assert unity == pytest.approx(0.1)


def test_pocket_primary_metric_keeps_output_translation():
    model = shifted_model()
    pose = _pose_batch(batch=1).numpy()
    scores = adapt.evaluate(model, pose, pose, pose, np.ones(pose.shape[:-1], bool), "cpu")
    assert scores["model"]["mpjpe_mm"] == pytest.approx(100.0, abs=1e-3)
    assert scores["model"]["recanonicalized_mpjpe_mm"] < 1e-3


def test_pocket_checkpoint_selection_uses_direct_primary_metric(monkeypatch):
    from dual2pose.pocket3.train import train

    class TrainableModel(torch.nn.Module):
        canonical_mode = "per_frame"

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def _shared_step(self, batch, stage):
            return (self.value - 1).square()

    scores = iter([{"mpjpe_mm": 10.0, "per_frame_canonical_mpjpe_mm": 30.0},
                   {"mpjpe_mm": 20.0, "per_frame_canonical_mpjpe_mm": 1.0}])
    monkeypatch.setattr(adapt, "evaluate", lambda *a, **k: {"model": next(scores)})
    from tests.test_training_protocol import _pose_batch
    pose = _pose_batch(batch=1, frames=3).numpy()
    data = (pose, pose, pose, np.ones(pose.shape[:-1], bool))
    _, info = train(TrainableModel(), data, data, "cpu", epochs=2, lr=1e-3, patience=2, seed=0)
    assert info["best_epoch"] == 1
    assert info["best_val_mm"] == 10.0


def test_temporal_metrics_exclude_eye_error_and_keep_output_translation():
    from dual2pose.eval.eval_unity_temporal_alignment import _model_outputs

    pose = _pose_batch(batch=1)
    eyes = _model_outputs(shifted_model(eyes_only=True), pose, pose, pose)
    shifted = _model_outputs(shifted_model(), pose, pose, pose)
    assert eyes["mpjpe"].item() == pytest.approx(0.0, abs=1e-7)
    assert shifted["mpjpe"].item() == pytest.approx(0.1, abs=1e-7)


def test_nview_oracle_selection_uses_common13():
    from dual2pose.eval.eval_unity_nview import evaluate_nview_group
    from tests.test_unity_nview_eval import make_sample

    class PairModel:
        def predict_pair(self, left_camera, right_camera, left_pose, right_pose):
            out = torch.zeros_like(left_pose)
            if (left_camera, right_camera) == ("a", "b"):
                out[..., :2, 0] = 100.0
            else:
                out[..., 0] = 0.1
            return out

    result = evaluate_nview_group(PairModel(), make_sample("a", "b", "c"), 3)
    assert result["pairwise_oracle_pair"] == ("a", "b")


def test_extension_summaries_default_to_common13_with_explicit_all15_diagnostic():
    from dual2pose.eval.eval_unity_masking import _summarize_outputs

    target = torch.zeros(1, 4, 15, 3)
    pred = target.clone()
    pred[..., :2, 0] = 1.0
    flat = {"fused": pred, "left_raw": pred, "right_raw": pred,
            "left_canonical": pred, "right_canonical": pred,
            "ground_truth": target, "ground_truth_canonical": target}
    primary = _summarize_outputs(flat, failure_threshold=0.1)
    assert primary["fused"]["mpjpe"] == 0.0
    assert len(primary["fused"]["per_joint_mpjpe"]) == 13
    auxiliary = _summarize_outputs(flat, failure_threshold=0.1, joint_subset="all15")
    assert auxiliary["fused"]["mpjpe"] == pytest.approx(2 / 15)


def test_frontend_summary_primary_fields_are_common13(tmp_path):
    from dual2pose.eval.eval_unity_frontend_generalization import _summary_row

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test")
    all15 = {"fused": {"mpjpe": 2.0}, "canonical_avg": {"mpjpe": 3.0}}
    common13 = {"fused": {"mpjpe": 0.1}, "canonical_avg": {"mpjpe": 0.2}}
    row = _summary_row("test", None, all15, common13, {}, 2, checkpoint, 0, 42)
    assert row["joint_subset"] == "common13"
    assert row["fused_mpjpe"] == 0.1
    assert row["fusion_gain_mpjpe"] == pytest.approx(0.1)
    assert row["all15_fused_mpjpe"] == 2.0


def test_view_angle_summary_excludes_eye_error():
    from dual2pose.eval.extension_experiment_utils import summarize_outputs_by_angle

    target = torch.zeros(1, 4, 15, 3)
    pred = target.clone()
    pred[..., :2, 0] = 1.0
    output = {"fused": pred, "left_canonical": pred, "right_canonical": pred,
              "ground_truth_canonical": target,
              "meta": {"cam1_id": ["capture_L0_A000"], "cam2_id": ["capture_L0_A090"]}}
    rows = summarize_outputs_by_angle([output], failure_threshold=0.1, bin_edges=[0, 180])
    assert rows[0]["fused_mpjpe"] == 0.0


def test_pocket_report_and_plot_use_direct_primary_metric(tmp_path, monkeypatch):
    model = shifted_model()
    pose = _pose_batch(batch=1, frames=4).numpy()[0]
    right = pose.copy()
    right[:, 12, 0] += 0.01  # a non-zero averaging baseline for gain reporting
    gt = {"frame": np.arange(4), "utc": np.arange(4) / ep.FPS,
          "joints": pose, "reliable": np.ones((4, 15), bool)}
    bbox = np.tile(np.array([0, 0, 1000, 1080], np.float32), (4, 1))
    monkeypatch.setattr(ep, "load_gt", lambda path: gt)
    monkeypatch.setattr(ep, "load_sam3d_dir", lambda path, n: (
        pose if path.name == "pocket_1" else right, np.ones(4, bool), bbox))
    monkeypatch.setattr(ep, "load_model", lambda *args: model)
    monkeypatch.setattr(ep, "_md5", lambda path: "fixture")
    plotted = {}
    monkeypatch.setattr(ep, "write_figure", lambda path, series, *args: plotted.update(series))
    args = ep.parse_args(["--dataset-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                          "--window", "4", "--stride", "4", "--gpu", "-1", "--ckpt", "test=unused.ckpt"])
    summary = ep.evaluate(args)
    assert summary["primary_metric"] == "mpjpe_m"
    assert summary["evaluation_protocol"]["output_alignment"] == "none"
    metrics = summary["metrics"]["canonfuse3d[test]"]
    assert metrics["mpjpe_m"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["recanonicalized_mpjpe_m"] < 1e-6
    assert np.mean(plotted["canonfuse3d[test]"]) == pytest.approx(0.1, abs=1e-6)
