"""Contracts reject historical, partial, duplicated, and nonfinite frozen results."""
import csv
import importlib
import json
from pathlib import Path

import pytest


def module():
    return importlib.import_module('dual2pose.experiments.workflow_evaluations')


def context(tmp_path):
    return {key: str(tmp_path / key) for key in ('repo', 'root', 'python', 'main_root', 'core_root', 'raw_cache')}


def test_builder_covers_approved_frozen_matrix_and_dependencies(tmp_path):
    jobs = module().build_jobs(context(tmp_path))
    ids = [job['id'] for job in jobs]
    assert len(ids) == len(set(ids))
    for prefix, count in [('eval_mask_', 45), ('eval_rgb_', 18), ('eval_offset_', 7), ('eval_frontend_', 32)]:
        assert sum(name.startswith(prefix) for name in ids) == count
    by_id = {job['id']: job for job in jobs}
    assert by_id['eval_sampling']['validation']['rows'] == 7
    assert by_id['eval_sync']['validation']['rows'] == 33
    assert 'eval_sync_calibration' in by_id['eval_sync']['deps']
    assert by_id['eval_nview']['resource'] == 'exclusive_gpu'
    assert 'dlt_residual_unity' in by_id['eval_c07']['deps']
    assert by_id['eval_c07']['validation']['actual_cells'] == 77
    for job in jobs:
        assert job['family'] == 'evaluations'
        assert 'CUDA_VISIBLE_DEVICES' not in job['env']
        assert all('run_frontend_adaptation_matrix' not in ' '.join(cmd) for cmd in job['commands'])


def csv_job(tmp_path):
    out = tmp_path / 'result.csv'
    job = {'id': 'test', 'validation': {'kind': 'csv', 'path': str(out), 'rows': 1,
          'samples': 64440, 'metric': 'fused_mpjpe', 'conditions': {'offset_frames': [0]}}}
    return job, out


def write_csv(path, **changes):
    row = {'evaluation_protocol': 'common13_direct_per_frame_v1', 'joint_subset': 'common13',
           'canonical_mode': 'per_frame', 'output_alignment': 'none', 'sample_count': 64440,
           'fused_mpjpe': .2, 'offset_frames': 0}
    row.update(changes)
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


@pytest.mark.parametrize('changes', [{'evaluation_protocol': 'old'}, {'sample_count': 64400},
                                     {'fused_mpjpe': 'nan'}, {'offset_frames': 1}])
def test_validation_rejects_bad_primary_artifacts(tmp_path, changes):
    job, path = csv_job(tmp_path)
    write_csv(path, **changes)
    with pytest.raises(ValueError):
        module().validate(job, {})


def test_validation_returns_hashed_verified_artifact(tmp_path):
    job, path = csv_job(tmp_path)
    write_csv(path)
    result = module().validate(job, {})
    assert result['rows'] == 1
    assert len(result['artifacts'][str(path)]) == 64


def test_calibration_rejects_smoke_and_nonvalidation_selection(tmp_path):
    path = tmp_path / 'validation_threshold.json'
    path.write_text(json.dumps({'descriptor_protocol': 'per_frame_body_velocity_v1', 'split': 'val',
        'limited_smoke_test': True, 'validation_sample_count': 64440, 'calibration_row_count': 708840,
        'selected_threshold': .3, 'injected_offsets': list(range(11))}))
    with pytest.raises(ValueError):
        module().validate({'validation': {'kind': 'calibration', 'path': str(path)}}, {})


def test_preparations_do_not_require_unavailable_main_checkpoint(tmp_path):
    for job in module().build_jobs(context(tmp_path)):
        if not job['deps']:
            assert '{unity_checkpoint}' not in json.dumps(job)


def test_sync_offsets_match_authorized_fractional_grid():
    assert module().SYNC_OFFSETS == (-5, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 5)


def test_calibration_accepts_complete_actual_validation_contract(tmp_path):
    path = tmp_path / 'validation_threshold.json'
    path.write_text(json.dumps({'descriptor_protocol': 'per_frame_body_velocity_v1', 'split': 'val',
        'limited_smoke_test': False, 'validation_sample_count': 128880, 'calibration_row_count': 1417680,
        'selected_threshold': .3, 'injected_offsets': [-5,-3,-2,-1,-.5,0,.5,1,2,3,5]}))
    assert module().validate({'validation': {'kind': 'calibration', 'path': str(path)}}, {})['threshold'] == .3


def test_csv_rejects_result_from_another_checkpoint(tmp_path):
    job, path = csv_job(tmp_path)
    checkpoint = tmp_path / 'current.ckpt'
    checkpoint.write_bytes(b'current-weights')
    job['env'] = {'EVAL_CKPT_PATH': str(checkpoint)}
    write_csv(path, checkpoint=str(tmp_path / 'old.ckpt'), checkpoint_sha256='a' * 64)
    with pytest.raises(ValueError, match='checkpoint'):
        module().validate(job, {})


def test_significance_rejects_nonfinite_primary_metrics(tmp_path):
    path = tmp_path / 'view_angle_statistics_last.json'
    payload = {'provenance': {'evaluation_protocol': module().PROTOCOL},
        'action_pair_row_count': 64440, 'cluster_count': 16110, 'bootstrap_resamples': 10000,
        'within_bin': [{'fused_mpjpe_mean': float('nan'), 'canonical_avg_mpjpe_mean': .2,
                        'p_raw': .1, 'p_holm': .3, 'mean_gain_ci95_low': -.1, 'mean_gain_ci95_high': .1}] * 6,
        'omnibus': {'significant_0_05': False, 'p_value': .3}, 'pairwise_contrasts': []}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='fused_mpjpe_mean'):
        module().validate({'validation': {'kind': 'significance', 'path': str(path)}}, {})


def test_native_mask_entrypoint_honors_single_condition_environment(tmp_path, monkeypatch):
    from dual2pose.eval.eval_unity_masking import _build_default_study
    jobs = module().build_jobs(context(tmp_path))
    job = next(job for job in jobs if job['id'] == 'eval_mask_right_distal_0p5')
    for name, value in job['env'].items():
        monkeypatch.setenv(name, value)
    settings = _build_default_study()
    assert len(settings) == 1
    assert (settings[0].view_mode, settings[0].pattern, settings[0].ratio) == ('right', 'distal', .5)


def test_native_offset_entrypoint_accepts_single_nonzero_condition():
    from dual2pose.eval.eval_unity_temporal_offset import _parse_offsets, _build_default_study
    for offset in module().OFFSETS:
        settings = _build_default_study(view_mode='right', offsets=_parse_offsets(str(offset)))
        assert len(settings) == 1
        assert settings[0].offset_frames == offset
