import json
from pathlib import Path

import pytest

from dual2pose.experiments.workflow_training import build_jobs, validate, validate_pocket_record


def context(tmp_path):
    return {k: str(tmp_path / k) for k in ('repo', 'root', 'python', 'main_root', 'core_root', 'raw_cache')}


def test_complete_training_matrix_and_dependencies(tmp_path):
    jobs = {j['id']: j for j in build_jobs(context(tmp_path))}
    fits = [j for j in jobs.values() if j['validation'].get('training')]
    assert len(fits) == 28  # remaining 36 minus main2, core6
    assert jobs['dlt_residual_unity']['deps'] == ['geometry_unity']
    assert jobs['frontend_mixed_full']['deps'] == ['main_unity', 'frontend_assets']
    assert jobs['pocket_reverse_random_seed2']['validation']['epochs'] == 300
    assert jobs['pocket_reverse_random_seed2']['validation']['patience'] == 40
    assert jobs['pocket_adapt_forward_heads_only']['validation']['epochs'] == 20
    assert all(j['family'] == 'training' for j in jobs.values())
    assert len(jobs) == len(set(jobs))
    assert not (tmp_path / 'root').exists()


def test_missing_artifact_does_not_release_dependency(tmp_path):
    ctx = context(tmp_path)
    job = next(j for j in build_jobs(ctx) if j['id'] == 'gate_dynamic_residual')
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(job, ctx)


def pocket_record():
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    return {'evaluation_protocol': dict(EVALUATION_PROTOCOL), 'epochs': 300, 'patience': 40,
            'epochs_run': 43, 'best_epoch': 3, 'selection': 'minimum_validation_common13',
            'history': [{'epoch': i, 'val_common13_mpjpe_mm': 1 if i == 3 else 2,
                         'train_loss': .1} for i in range(1, 44)],
            'blocks': {'train': [0, 100], 'val': [100, 160], 'test': [160, 220]},
            'window_starts': {'train': [0, 30, 60], 'val': [100, 130], 'test': [160, 190]},
            'valid_test_points': 780, 'metrics': {'model': {'mpjpe_mm': 20., 'pa_mpjpe_mm': 10., 'accel_mm': 1.}, 'gate_only': {'mpjpe_mm': 22.}}}


def test_pocket_budget_early_stopping_and_mask_coverage():
    record = pocket_record()
    validate_pocket_record(record, {'kind': 'pocket_train', 'epochs': 300, 'patience': 40})
    record['epochs_run'] = 10
    with pytest.raises(ValueError, match='history|budget|early'):
        validate_pocket_record(record, {'kind': 'pocket_train', 'epochs': 300, 'patience': 40})


def test_pocket_cross_boundary_windows_and_nonfinite_rejected():
    record = pocket_record()
    record['window_starts']['train'] = [90]
    with pytest.raises(ValueError, match='boundary'):
        validate_pocket_record(record, {'kind': 'pocket_train', 'epochs': 300, 'patience': 40})
    record = pocket_record()
    record['metrics']['model']['mpjpe_mm'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        validate_pocket_record(record, {'kind': 'pocket_train', 'epochs': 300, 'patience': 40})


def test_gate_requires_actual_full_budget_selection_and_checkpoint_hash(tmp_path):
    import hashlib
    import torch
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    ctx = context(tmp_path)
    job = next(j for j in build_jobs(ctx) if j['id'] == 'gate_dynamic_residual')
    out = Path(job['validation']['output'])
    out.mkdir(parents=True)
    config = {'evaluation_protocol': dict(EVALUATION_PROTOCOL), 'epochs': 100, 'smoke_steps': 0}
    rows = [{'epoch': e, 'train_samples': 192512, 'validation': {'common13': {'mpjpe': .1 + e}}}
            for e in range(100)]
    (out / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    torch.save({'canonical_mode': 'per_frame', 'evaluation_protocol': dict(EVALUATION_PROTOCOL),
                'state_dict': {'weight': torch.zeros(1)}, 'epoch': 0}, out / 'best.pt')
    digest = hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest()
    record = {'configuration': config, 'selected_epoch': 0, 'checkpoint_sha256': digest,
              'metrics': {'common13': {'mpjpe': .1, 'sample_count': 64440,
              'point_count': 64440 * 30 * 13, 'acceleration_point_count': 64440 * 28 * 13}}}
    (out / 'metrics.json').write_text(json.dumps(record))
    assert validate(job, ctx)['checkpoint'] == str(out / 'best.pt')
    record['checkpoint_sha256'] = 'corrupt'
    (out / 'metrics.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='hash'):
        validate(job, ctx)
    record['checkpoint_sha256'] = digest
    (out / 'metrics.json').write_text(json.dumps(record))
    (out / 'history.jsonl').write_text(json.dumps(rows[-1]) + '\n')
    with pytest.raises(ValueError, match='budget/history'):
        validate(job, ctx)


def test_pocket_old_protocol_is_rejected():
    record = pocket_record()
    record['evaluation_protocol']['canonical_mode'] = 'first_frame'
    with pytest.raises(ValueError, match='protocol'):
        validate_pocket_record(record, {'kind': 'pocket_train', 'epochs': 300, 'patience': 40})


def test_pocket_adaptation_wrapper_uses_existing_apis_and_exports_valid_result(tmp_path, monkeypatch):
    """Execute the worker wrapper with tiny arrays and signature-checked test doubles."""
    import inspect
    import numpy as np
    import torch
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    from dual2pose.pocket3 import evaluate as ep, adapt as ad, train as tr
    from dual2pose.experiments.workflow_training import run_pocket
    original_adapt, original_evaluate, original_save = ad.adapt, ad.evaluate, tr.save_weights
    source = tmp_path / 'source.pt'
    payload = {'canonical_mode': 'per_frame', 'evaluation_protocol': dict(EVALUATION_PROTOCOL),
               'state_dict': {'weight': torch.zeros(1)}}
    torch.save(payload, source)
    fake_model = type('TestModel', (), {})()
    monkeypatch.setattr(ep, 'load_model', lambda path, device: fake_model)
    poses = np.zeros((100, 15, 3), dtype=np.float32)
    reliable = np.ones((100, 15), dtype=bool)
    monkeypatch.setattr(tr, 'load_frames', lambda args: (poses, poses, np.arange(100),
        {'joints': poses, 'reliable': reliable}, np.ones(100, bool), np.ones(100, bool)))
    called = []

    def fake_adapt(*args, **kwargs):
        bound = inspect.signature(original_adapt).bind(*args, **kwargs)
        assert bound.arguments['epochs'] == 20
        assert bound.arguments['scope'] == 'heads_only'
        called.append('adapt')
        return fake_model

    def fake_evaluate(*args, **kwargs):
        inspect.signature(original_evaluate).bind(*args, **kwargs)
        called.append('evaluate')
        return {name: {'mpjpe_mm': 1., 'pa_mpjpe_mm': .5, 'accel_mm': .1}
                for name in ('model', 'gate_only', 'canonical_average', 'aligned_average')}

    def fake_save(*args, **kwargs):
        bound = inspect.signature(original_save).bind(*args, **kwargs)
        torch.save(payload, bound.arguments['path'])

    monkeypatch.setattr(ad, 'adapt', fake_adapt)
    monkeypatch.setattr(ad, 'evaluate', fake_evaluate)
    monkeypatch.setattr(tr, 'save_weights', fake_save)
    monkeypatch.setattr(ep, 'DEFAULT_ROOT', tmp_path / 'data')
    for cam in ('pocket_1', 'pocket_2'):
        path = ep.DEFAULT_ROOT / 'gt/pocket_aligned' / f'chen_4_{cam}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, joints=poses)
    spec = {'output': str(tmp_path / 'run'), 'checkpoint': str(source), 'direction': 'forward',
            'scope': 'heads_only', 'seed': 0, 'epochs': 20}
    run_pocket('pocket_adapt', spec)
    result = validate({'validation': {'kind': 'pocket_adapt', **spec}}, {})
    assert Path(result['checkpoint']).is_file()
    assert result['valid_test_points'] == 30 * 13
    assert called == ['adapt', 'evaluate']
    with pytest.raises(FileExistsError):
        run_pocket('pocket_adapt', spec)
