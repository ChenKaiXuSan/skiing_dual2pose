"""Unit tests for the hand-held dual-Pocket evaluation (no GPU, no checkpoint)."""
import numpy as np
import pytest

from dual2pose.pocket3 import evaluate as ep
from dual2pose.map_config import FILTERED_KPTS_MAPPING


def _coco_pose(f=4):
    rng = np.random.default_rng(0)
    return rng.normal(size=(f, 17, 3)).astype(np.float32)


def test_coco_mapping_matches_model_skeleton():
    names = {v: k for k, v in FILTERED_KPTS_MAPPING.items()}
    j = _coco_pose()
    out, rel = ep.map_coco17_to_model15(j, np.ones((4, 17), bool), ep.COCO17)
    assert out.shape == (4, 15, 3) and rel.all()
    # left/right consistency with the model skeleton names
    assert np.allclose(out[:, names["Upperarm_L"]], j[:, ep.COCO17.index("l_shoulder")])
    assert np.allclose(out[:, names["Hand_R"]], j[:, ep.COCO17.index("r_wrist")])
    assert np.allclose(out[:, names["Foot_L"]], j[:, ep.COCO17.index("l_ankle")])
    mid = 0.5 * (j[:, 5] + j[:, 6])
    assert np.allclose(out[:, names["neck_01"]], mid)


def test_neck_reliability_requires_both_shoulders():
    rel = np.ones((2, 17), bool)
    rel[0, 6] = False
    _, r15 = ep.map_coco17_to_model15(_coco_pose(2), rel, ep.COCO17)
    assert not r15[0, ep.NECK] and r15[1, ep.NECK]
    assert not r15[0, 3] and r15[0, 2]  # right shoulder unreliable, left fine


def test_mapping_rejects_wrong_joint_order():
    with pytest.raises(ValueError):
        ep.map_coco17_to_model15(_coco_pose(1), np.ones((1, 17), bool), list(reversed(ep.COCO17)))


def test_pair_by_utc_constant_offset():
    t0 = 1.0e9
    utc_left = t0 + np.arange(100) / ep.FPS
    utc_right = t0 - 0.879 + np.arange(140) / ep.FPS   # right started 0.879 s earlier
    j, resid = ep.pair_by_utc(utc_left, utc_right)
    assert (j >= 0).all()
    assert np.allclose(j, np.round(0.879 * ep.FPS + np.arange(100)))
    assert np.abs(resid).max() <= 0.5 / ep.FPS


def test_pair_by_utc_rejects_gaps():
    utc_left = np.arange(10) / ep.FPS
    utc_right = np.array([0.0, 5.0])
    j, _ = ep.pair_by_utc(utc_left, utc_right)
    assert j[0] == 0 and (j[1:] == -1).all()


def test_build_windows_skips_unusable_frames():
    usable = np.ones(100, bool)
    usable[35] = False
    starts = ep.build_windows(usable, window=30, stride=30)
    assert starts == [0, 36, 66]
    anchor = np.ones(100, bool)
    anchor[36] = False          # first frame of the second window lacks a reliable anchor
    assert ep.build_windows(usable, 30, 30, anchor) == [0, 37, 67]


def test_masked_metrics():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(2, 30, 15, 3))
    rel = np.ones((2, 30, 15), bool)
    pred = gt.copy()
    pred[..., 2, :] += [0.01, 0, 0]           # 1 cm error on joint 2 only
    rel[0, :, 3] = False                        # joint 3 unreliable in window 0
    mp, n = ep.masked_mpjpe(pred, gt, rel, ep.COMMON13)
    assert n == 2 * 30 * 13 - 30
    assert np.isclose(mp, 0.01 * 60 / n)
    pa, nf, nd = ep.masked_pa_mpjpe(pred, gt, rel, ep.COMMON13)
    assert nf == 30 and nd == 0 and pa < 0.01
    ac, na = ep.masked_accel_error(pred, gt, rel, ep.COMMON13)
    assert np.isclose(ac, 0.0) and na == 2 * 28 * 13 - 28
    assert ep.masked_mpjpe(gt, gt, rel, ep.COMMON12_NO_NECK)[0] == 0.0


def test_subject_in_view_rule():
    valid = np.ones(5, bool)
    bbox = np.zeros((5, 4), np.float32)
    bbox[:, 3] = [300.0, 700.0, 377.0, 378.0, 900.0]
    valid[4] = False
    flags = ep.subject_in_view(bbox, valid, 1080.0, min_frac=0.35)
    assert flags.tolist() == [False, True, False, True, False]
    bbox[1] = np.nan
    assert not ep.subject_in_view(bbox, valid, 1080.0)[1]


def test_sequence_aligned_mpjpe_removes_global_similarity():
    rng = np.random.default_rng(3)
    gt = rng.normal(size=(2, 30, 15, 3))
    rel = np.ones((2, 30, 15), bool)
    rel[1, :5, 4] = False
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    pred = 1.3 * gt @ rot.T + np.array([0.5, -0.2, 2.0])
    raw, _ = ep.masked_mpjpe(pred, gt, rel, ep.COMMON13)
    aligned, n = ep.sequence_aligned_mpjpe(pred, gt, rel, ep.COMMON13)
    assert raw > 0.5 and aligned < 1e-9
    assert n == 2 * 30 * 13 - 5
    per_frame = ep.sequence_aligned_errors(pred, gt, rel, ep.COMMON13)
    assert per_frame.shape == (2, 30) and np.nanmax(per_frame) < 1e-9
    s_, r_, t_ = ep.umeyama_similarity(gt[0, :, 2], pred[0, :, 2])
    assert np.isclose(s_, 1.3) and np.allclose(r_, rot.T) and np.allclose(t_, [0.5, -0.2, 2.0])


def test_per_frame_canonical_removes_global_motion():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(1, 1, 15, 3)).astype(np.float32)
    base[0, 0, ep.LEFT_HIP] = [-0.1, 0, 0]; base[0, 0, ep.RIGHT_HIP] = [0.1, 0, 0]
    base[0, 0, ep.NECK] = [0, 0.5, 0]; base[0, 0, 0] = [-0.03, 0.65, 0.1]; base[0, 0, 1] = [0.03, 0.65, 0.1]
    # same pose, translated and rotated differently in every frame
    T = 6
    poses = np.repeat(base, T, axis=1)
    for t in range(T):
        th = 0.3 * t
        rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)
        poses[0, t] = poses[0, t] @ rot.T + np.array([t, 0.2 * t, -0.5 * t], np.float32)
    rel = np.ones((1, T, 15), bool)
    raw = ep.masked_mpjpe(poses, np.repeat(base, T, axis=1), rel, ep.COMMON13)[0]
    canon = ep.masked_mpjpe(ep.per_frame_canonical(poses), ep.per_frame_canonical(np.repeat(base, T, axis=1)), rel, ep.COMMON13)[0]
    assert raw > 1.0 and canon < 1e-4
    rr = ep.root_relative(poses)
    assert np.allclose(rr[..., [ep.LEFT_HIP, ep.RIGHT_HIP], :].mean(-2), 0, atol=1e-6)



class _FakeFusion:
    """Stands in for CrossViewFusionTrainer: canonical average in the per-frame body frame."""

    canonical_mode = "per_frame"

    def canonicalize(self, x):
        from dual2pose.trainer.canonicalize import canonicalize_pose_torch
        return canonicalize_pose_torch(x, left_hip=ep.LEFT_HIP, right_hip=ep.RIGHT_HIP, neck=ep.NECK, mode="per_frame")

    def models(self, left, right, return_components=True):
        import torch
        fused = 0.5 * (left + right)
        return fused, {"output_residual": torch.zeros_like(fused), "alpha": torch.full(left.shape[:2] + (1,), 0.5)}


def test_run_methods_feeds_the_model_the_per_frame_body_frame():
    import torch
    rng = np.random.default_rng(11)
    W, T = 2, 6
    base = np.zeros((1, 1, 15, 3), np.float32)
    base[0, 0, ep.LEFT_HIP] = [-0.1, 0, 0]; base[0, 0, ep.RIGHT_HIP] = [0.1, 0, 0]
    base[0, 0, ep.NECK] = [0, 0.5, 0]; base[0, 0, 0] = [-0.03, 0.65, 0.1]; base[0, 0, 1] = [0.03, 0.65, 0.1]
    base[0, 0, 2:14] = rng.normal(scale=0.3, size=(12, 3))
    L = np.repeat(np.repeat(base, T, axis=1), W, axis=0) + rng.normal(scale=0.02, size=(W, T, 15, 3)).astype(np.float32)
    R = L + rng.normal(scale=0.02, size=L.shape).astype(np.float32)
    G = L + rng.normal(scale=0.01, size=L.shape).astype(np.float32)
    for t in range(T):  # translate every frame somewhere else
        L[:, t] += np.array([t, -t, 2 * t], np.float32); R[:, t] += np.array([-t, 0, t], np.float32)
    out = ep.run_methods(L, R, G, {"pf": _FakeFusion()}, torch.device("cpu"))
    # the non-learned baselines keep the first-frame window protocol
    assert not np.allclose(out["canonical_average"][..., [ep.LEFT_HIP, ep.RIGHT_HIP], :].mean(-2), 0, atol=1e-2)
    # the model receives per-frame inputs, so its output is a per-frame body-frame pose
    expected = 0.5 * (ep.per_frame_canonical(L) + ep.per_frame_canonical(R))
    assert np.allclose(out["canonfuse3d[pf]"], expected, atol=1e-5)
    assert np.allclose(out["canonfuse3d[pf]"][..., [ep.LEFT_HIP, ep.RIGHT_HIP], :].mean(-2), 0, atol=1e-5)
    # and the matching reference is exposed, centred in every frame
    assert np.allclose(out["gt_per_frame"], ep.per_frame_canonical(G), atol=1e-5)
    assert np.allclose(out["gt_per_frame"][..., [ep.LEFT_HIP, ep.RIGHT_HIP], :].mean(-2), 0, atol=1e-6)
    assert out["gt"].shape == out["gt_per_frame"].shape == (W, T, 15, 3)
