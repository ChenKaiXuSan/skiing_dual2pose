import numpy as np
import pytest
import torch

from dual2pose.pocket3 import evaluate as ep
from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
from tests.test_training_protocol import _config, _pose_batch


def test_default_outputs_use_fresh_per_frame_namespace():
    first, second = ep.parse_args([]).output_dir, ep.parse_args([]).output_dir
    assert first != second
    assert first.parent == ep.REPO_ROOT / 'logs/per_frame_common13_v1/pocket3/evaluation'


def test_unreliable_nan_target_has_identical_loss_and_gradients():
    model = CrossViewFusionTrainer(_config()).eval()
    model.log = lambda *a, **k: None
    pose = _pose_batch(batch=1, frames=5)
    reliable = torch.ones(pose.shape[:-1], dtype=torch.bool)
    reliable[..., 4] = False
    def run(value):
        gt = pose.clone(); gt[..., 4, :] = value
        batch = {'kpt3d_sam': {'cam1': pose, 'cam2': pose}, 'kpt3d_gt': gt,
                 'kpt3d_gt_reliable': reliable}
        model.zero_grad()
        loss = model._shared_step(batch, 'train'); loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]
    expected, eg = run(99.)
    actual, ag = run(float('nan'))
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(ag, eg):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b)


def test_unreliable_anchor_is_rejected_before_canonicalization():
    model = CrossViewFusionTrainer(_config()).eval()
    model.log = lambda *a, **k: None
    pose = _pose_batch(batch=1, frames=5)
    reliable = torch.ones(pose.shape[:-1], dtype=torch.bool)
    reliable[:, 2, ep.NECK] = False
    with pytest.raises(ValueError, match='anchor'):
        model._shared_step({'kpt3d_sam': {'cam1': pose, 'cam2': pose},
                            'kpt3d_gt': pose, 'kpt3d_gt_reliable': reliable}, 'train')


@pytest.mark.parametrize('failure', ['empty', 'nonfinite'])
def test_primary_metric_rejects_invalid_scores(failure):
    pose = _pose_batch(batch=1, frames=5).numpy()
    reliable = np.ones(pose.shape[:-1], bool)
    if failure == 'empty':
        reliable[:] = False
    else:
        pose[..., 4, :] = np.nan
    with pytest.raises(ValueError, match='valid|finite'):
        ep.masked_mpjpe(pose, pose, reliable, ep.COMMON13)


def test_per_frame_metric_masks_nan_instead_of_multiplying():
    gt = _pose_batch(batch=1, frames=5).numpy()
    pred = gt.copy(); pred[..., 4, :] = np.nan
    reliable = np.ones(gt.shape[:-1], bool); reliable[..., 4] = False
    np.testing.assert_array_equal(ep.per_frame_mpjpe(pred, gt, reliable, ep.COMMON13), 0)


def test_native_ski_adapter_preserves_declared_comparison_axes():
    model = CrossViewFusionTrainer(_config(data={'left_hip_idx': 4, 'right_hip_idx': 5,
                                               'neck_idx': 12, 'face_sign': False})).eval()
    pose = _pose_batch(batch=1, frames=5)
    pose[..., 2] *= -1
    native, transform = model.canonicalize(ep.native_pose(model, pose))
    converted = ep.to_comparison_frame(native, transform, pose)
    expected = ep.comparison_canonical(pose)[0][..., ep.COMMON13, :]
    torch.testing.assert_close(converted, expected)
    assert not model.face_sign


def test_native_adapter_keeps_output_translation_under_camera_motion():
    from tests.test_training_protocol import _rigid_per_frame
    model = CrossViewFusionTrainer(_config(data={'left_hip_idx': 4, 'right_hip_idx': 5,
                                               'neck_idx': 12, 'face_sign': False})).eval()
    pose = _rigid_per_frame(_pose_batch(batch=2, frames=5), 91)
    pose[..., 2] *= -1
    native, transform = model.canonicalize(ep.native_pose(model, pose))
    shift = torch.tensor([0.2, -0.1, 0.3])
    converted = ep.to_comparison_frame(native + shift, transform, pose)
    gt = ep.comparison_canonical(pose)[0][..., ep.COMMON13, :]
    errors = torch.linalg.vector_norm(converted - gt, dim=-1)
    torch.testing.assert_close(errors, torch.full_like(errors, shift.norm()), atol=1e-5, rtol=0)


@pytest.mark.parametrize('native13', [False, True])
def test_synthetic_training_checkpoint_evaluation_roundtrip(tmp_path, native13):
    from dual2pose.pocket3 import train, adapt
    data = {'left_hip_idx': 4, 'right_hip_idx': 5, 'neck_idx': 12, 'face_sign': False} if native13 else {}
    model = CrossViewFusionTrainer(_config(data=data)).eval()
    pose = _pose_batch(batch=1, frames=5).numpy()
    gt = pose.copy(); gt[..., 4, :] = np.nan
    rel = np.ones(pose.shape[:-1], bool); rel[..., 4] = False
    batch = (pose, pose, gt, rel)
    trained, info = train.train(model, batch, batch, 'cpu', epochs=1, lr=1e-4, patience=1, seed=3)
    assert info['best_epoch'] == 1 and np.isfinite(info['best_val_mm'])
    path = tmp_path / 'masked.ckpt'
    train.save_weights(trained, path)
    restored = ep.load_model(path, 'cpu')
    before = adapt.evaluate(trained, *batch, 'cpu')
    after = adapt.evaluate(restored, *batch, 'cpu')
    assert before['model']['mpjpe_mm'] == after['model']['mpjpe_mm']
    assert restored.face_sign == model.face_sign
    adapted = adapt.adapt(restored, *batch, 'cpu', 'full', epochs=1, lr=1e-4, seed=3)
    assert np.isfinite(adapt.evaluate(adapted, *batch, 'cpu')['model']['mpjpe_mm'])


def test_windows_cannot_cross_frozen_block_boundaries():
    from dual2pose.pocket3.train import windows_in, gather
    usable = np.ones(100, bool)
    for lo, hi in ((0, 40), (40, 70), (70, 100)):
        starts = windows_in(usable, usable, lo, hi, window=20, stride=5)
        assert starts and all(lo <= s and s + 20 <= hi for s in starts)
    with pytest.raises(ValueError, match='no valid'):
        gather(None, None, None, None, [], 30)
