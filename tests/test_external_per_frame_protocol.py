"""Behavioral regressions for the current external-method coordinate contract."""
import json

import numpy as np
import pytest
import torch

from dual2pose.eval.evaluate_metapose import canonical_prediction_and_control
from dual2pose.eval.stride_external import canonical_average17
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator


def moving_pose(joints=15):
    rng = np.random.default_rng(91)
    base = rng.normal(size=(joints, 3)).astype(np.float32)
    lh, rh, neck = (6, 7, 14) if joints == 15 else (4, 5, 12)
    base[lh], base[rh], base[neck] = [-1, 0, 0], [1, 0, 0], [0, 2, 0]
    base[:2] = [0, 2, 1]
    angles = np.arange(90).reshape(3, 30) * .17
    rotations = np.zeros((3, 30, 3, 3), np.float32)
    rotations[..., 0, 0] = rotations[..., 1, 1] = np.cos(angles)
    rotations[..., 0, 1] = -np.sin(angles)
    rotations[..., 1, 0] = np.sin(angles)
    rotations[..., 2, 2] = 1
    translation = rng.normal(size=(3, 30, 1, 3)).astype(np.float32) * 2
    native = base @ rotations + translation
    return native, base, rotations, translation


@pytest.mark.parametrize('dataset,joints', [('unity', 15), ('ski', 13)])
def test_native_prediction_uses_each_input_frame_and_keeps_output_translation(dataset, joints):
    native, base, rotations, translation = moving_pose(joints)
    # The output-only shift is expressed in the input's camera frame. It must
    # remain a 0.5 m error after conversion, including at different B,T frames.
    offset = np.array([.3, .4, 0], np.float32) @ rotations
    prediction = native[..., -13:, :] + offset[..., None, :]
    mapped, average = canonical_prediction_and_control(prediction, native, native, dataset)
    expected = np.broadcast_to(base[-13:] + [.3, .4, 0], mapped.shape)
    np.testing.assert_allclose(mapped, expected, atol=2e-6)
    acc = PAMetricAccumulator(); acc.update(mapped, average)
    assert acc.result()['mpjpe'] == pytest.approx(.5, abs=1e-6)
    reverse, _ = canonical_prediction_and_control(prediction[::-1].copy(), native[::-1].copy(), native[::-1].copy(), dataset)
    torch.testing.assert_close(reverse.flip(0), mapped)
    singles = torch.cat([canonical_prediction_and_control(prediction[i:i+1], native[i:i+1], native[i:i+1], dataset)[0] for i in range(3)])
    torch.testing.assert_close(singles, mapped)


@pytest.mark.parametrize('dataset,joints', [('unity', 15), ('ski', 13)])
def test_stride_extra_bones_follow_input_frame_and_are_batch_invariant(dataset, joints):
    native, base, rotations, translation = moving_pose(joints)
    bones_base = np.ones((127, 3), np.float32) * [3, 4, 5]
    bones = bones_base @ rotations + translation
    output = canonical_average17(native, native, bones, bones, dataset)
    np.testing.assert_allclose(output[..., 0, :], np.broadcast_to([3, 4, 5], (3, 30, 3)), atol=3e-6)
    np.testing.assert_allclose(output[..., 4, :], np.broadcast_to([-1, 0, 0], (3, 30, 3)), atol=3e-6)
    reverse = canonical_average17(native[::-1].copy(), native[::-1].copy(), bones[::-1].copy(), bones[::-1].copy(), dataset)
    np.testing.assert_allclose(reverse[::-1], output, atol=1e-6)


def test_deciwatch_preparation_is_per_frame_and_old_prepared_cache_is_rejected(tmp_path, monkeypatch):
    from dual2pose.experiments.run_deciwatch import _canonical_average, WindowDataset
    from dual2pose.eval.main_baseline_data import sha256
    native, base, _, _ = moving_pose()
    cache = tmp_path/'raw'; cache.mkdir()
    prepared = tmp_path/'prepared'; prepared.mkdir()
    for name in ('left', 'right', 'target'):
        np.save(cache/f'{name}.npy', native)
    np.save(cache/'frames.npy', np.tile(np.arange(30), (3, 1)))
    (cache/'samples.jsonl').write_text(''.join(json.dumps({'index': i})+'\n' for i in range(3)))
    manifest = dict(dataset='unity', split='train', sample_count=3, time_window=30,
                    representation='raw_pre_batch_canonicalization',
                    files={p.name: sha256(p) for p in cache.iterdir()})
    (cache/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setitem(__import__('dual2pose.experiments.run_deciwatch', fromlist=['EXPECTED']).EXPECTED, 'unity', {'train': 3})
    _canonical_average(cache, prepared, True)
    np.testing.assert_allclose(np.load(prepared/'input.npy'), np.broadcast_to(base[2:], (3, 30, 13, 3)), atol=2e-6)
    x, y = WindowDataset(prepared/'input.npy', prepared/'target.npy')[1]
    torch.testing.assert_close(x, y)
    # An old canonical array cannot enter training merely because it exists.
    legacy = tmp_path/'legacy'; legacy.mkdir()
    np.save(legacy/'input.npy', native[..., 2:, :])
    with pytest.raises(ValueError, match='manifest|protocol'):
        WindowDataset(legacy/'input.npy')
    # Hash validation prevents replacing prepared arrays under a current label.
    np.save(prepared/'input.npy', native[..., 2:, :])
    with pytest.raises(ValueError, match='checksum'):
        WindowDataset(prepared/'input.npy')


def test_current_control_rejects_legacy_or_label_only_provenance(tmp_path):
    from dual2pose.eval.external_protocol import check_current_control, protocol_metadata
    path = tmp_path/'control.json'
    path.write_text(json.dumps({'average_control': {'mpjpe': 0}}))
    with pytest.raises(ValueError, match='protocol'):
        check_current_control(path, dataset='ski', cache_manifest_sha256='raw-cache')
    path.write_text(json.dumps(dict(**protocol_metadata(), average_control={'mpjpe': 0})))
    with pytest.raises(ValueError, match='provenance'):
        check_current_control(path, dataset='ski', cache_manifest_sha256='raw-cache')


def test_stride_scorer_retains_output_translation(tmp_path):
    from dual2pose.experiments.run_stride_external import evaluate_predictions
    from dual2pose.eval.stride_external import COMMON
    from dual2pose.eval.main_baseline_data import sha256
    native, base, rotations, translation = moving_pose()
    np.save(tmp_path/'target.npy', native)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        files={'target.npy': sha256(tmp_path/'target.npy')})))
    bones = np.ones((127, 3), np.float32) @ rotations + translation
    avg = canonical_average17(native, native, bones, bones, 'unity')
    prediction = avg[..., COMMON, :] + [.3, .4, 0]
    result = evaluate_predictions(tmp_path, prediction, avg, 'unity')
    assert result['average_control']['mpjpe'] == pytest.approx(0, abs=1e-6)
    assert result['stride_adapted']['mpjpe'] == pytest.approx(.5, abs=1e-6)
    assert result['stride_adapted']['point_count'] == 3*30*13
