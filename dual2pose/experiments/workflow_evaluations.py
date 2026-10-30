"""Frozen-study DAG and strict artifact gates for the approved per-frame rerun.

Builders are side-effect free. Preparation commands only write inside the new run
namespace; historical raw detections are checked and hashed, never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from dual2pose.runtime_paths import input_path
from typing import Any

PROTOCOL = {'name': 'common13_direct_per_frame_v1', 'joint_subset': 'common13',
            'canonical_mode': 'per_frame', 'output_alignment': 'none',
            'reduction': 'mean_joint_frame_distance'}
SAMPLES = 64440
FRONTENDS = ('videopose3d', 'poseformer', 'motionbert')
OFFSETS = (-2, -1, -.5, 0, .5, 1, 2)
SYNC_OFFSETS = (-5, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 5)
RATES = (-.02, -.01, -.005, 0, .005, .01, .02)
GATE_CONDITIONS = ('clean', 'left_zero', 'right_zero', 'switch_left_to_right', 'switch_right_to_left')


def _slug(value):
    return str(value).replace('-', 'm').replace('.', 'p')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def build_jobs(ctx: dict) -> list[dict]:
    repo, root, python = Path(ctx['repo']), Path(ctx['root']) / 'evaluations', ctx['python']
    jobs: list[dict] = []
    hydra = ['train.gpu=0', 'train.fold=0', 'train.seed=42', 'data.batch_size=256',
             'data.num_workers=4', 'data.load_frames=false', 'data.load_2d_kpt=false',
             'data.load_3d_kpt=true', 'data.load_mask=false']

    def command(module, *args):
        return [python, '-u', '-m', module, *map(str, args)]

    def add(name, commands, validation, *, deps=('main_unity',), env=None, resource='gpu', out=None):
        directory = out or root / name
        jobs.append({'id': name, 'deps': list(deps), 'commands': commands, 'resource': resource,
                     'family': 'evaluations', 'env': {**({'EVAL_CKPT_PATH': '{unity_checkpoint}'} if 'main_unity' in deps else {}),
                     'EVAL_OUTPUT_ROOT': str(directory), 'EVAL_SEED': '42', 'OMP_NUM_THREADS': '4',
                     'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'ALIGN_LIMIT_BATCHES': '0',
                     'NVIEW_LIMIT_GROUPS': '0', **(env or {})}, 'validation': validation})
        return jobs[-1]

    def hjob(name, module, validation, **kwargs):
        out = kwargs.get('out') or root / name
        return add(name, [command('dual2pose.eval.' + module, *hydra, f'log_path={out}/hydra')], validation, **kwargs)

    def csv_contract(name, pattern, rows=1, conditions=None, **extra):
        return {'kind': 'csv', 'path': str(root / name / pattern), 'rows': rows, 'samples': SAMPLES,
                'metric': 'fused_mpjpe', 'conditions': conditions or {}, **extra}

    mask_jobs = []
    for view, pattern, ratio in itertools.product(('left', 'right', 'both'), ('random', 'distal', 'temporal'), (0, .25, .5, .75, 1)):
        name = f'eval_mask_{view}_{pattern}_{_slug(ratio)}'
        mask_jobs.append(name)
        hjob(name, 'eval_unity_masking', csv_contract(name, 'occlusion_summary_*.csv',
            conditions={'view_mode': [view], 'pattern': [pattern], 'ratio': [ratio]}),
            env={'MASK_VIEW_MODES': view, 'MASK_PATTERNS': pattern, 'MASK_RATIOS': str(ratio)})
    merged = root / 'pose_masking.csv'
    add('eval_merge_masking', [command(__name__, 'merge-masking', '--root', root)],
        {'kind': 'csv', 'path': str(merged), 'rows': 45, 'samples': SAMPLES, 'metric': 'fused_mpjpe',
         'conditions': {'view_mode': ['left', 'right', 'both'], 'pattern': ['random', 'distal', 'temporal'],
                        'ratio': [0, .25, .5, .75, 1]}}, deps=mask_jobs, resource='cpu')

    rgb_root = Path(ctx.get('rgb_manifest_root', repo / 'logs/ivc_mmsports_extension/image_occlusion/inference'))
    rgb_audit = root / 'rgb_assets.json'
    add('eval_prepare_rgb', [command(__name__, 'audit-rgb', '--source', rgb_root, '--output', rgb_audit)],
        {'kind': 'asset_audit', 'path': str(rgb_audit), 'manifests': 6}, deps=(), resource='cpu')
    for view, pattern, ratio in itertools.product(('left', 'right', 'both'), ('random', 'distal', 'temporal'), (.5, 1.)):
        name = f'eval_rgb_{view}_{pattern}_{_slug(ratio)}'
        cell = f'{view}_{pattern}_r{ratio:.2f}'.replace('.', 'p')
        add(name, [command('dual2pose.eval.eval_unity_image_occlusion', '--manifest-root', rgb_root,
            '--output-root', root / name, '--checkpoint', '{unity_checkpoint}', '--pose-occlusion-csv', merged,
            '--gpu', 0, '--batch-size', 256, '--num-workers', 4, '--cell', cell)],
            csv_contract(name, 'image_occlusion_summary_last.csv', conditions={'view_mode': [view], 'pattern': [pattern], 'ratio': [ratio]}),
            deps=('main_unity', 'eval_prepare_rgb', 'eval_merge_masking'))

    for offset in OFFSETS:
        name = 'eval_offset_' + _slug(offset)
        hjob(name, 'eval_unity_temporal_offset', csv_contract(name, 'temporal_offset_summary_*.csv',
             conditions={'offset_frames': [offset]}), env={'TEMPORAL_OFFSETS': str(offset), 'TEMPORAL_OFFSET_VIEW': 'right'})
    hjob('eval_sampling', 'eval_unity_sampling_rate', csv_contract('eval_sampling', 'sampling_rate_summary_*.csv', 7,
         conditions={'rate_error_fraction': list(RATES)}),
         env={'SAMPLING_RATE_ERRORS': ','.join(map(str, RATES)), 'SAMPLING_RATE_VIEW': 'right', 'SAMPLING_RATE_ANCHOR': 'center'})
    hjob('eval_angles', 'eval_unity_view_angle', csv_contract('eval_angles', 'view_angle_summary_*.csv', 6, conditions={'angle_bin': ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180']}, sample_mode='sum'))
    hjob('eval_angle_significance', 'run_unity_view_angle_significance',
         {'kind': 'significance', 'path': str(root / 'eval_angle_significance/view_angle_statistics_last.json')},
         env={'BOOTSTRAP_RESAMPLES': '10000'})

    sync_root = root / 'sync'
    hjob('eval_sync_calibration', 'calibrate_temporal_alignment',
         {'kind': 'calibration', 'path': str(sync_root / 'validation_threshold.json'), 'samples': 128880}, deps=(), out=sync_root)
    hjob('eval_sync', 'eval_unity_temporal_alignment',
         {'kind': 'sync', 'path': str(sync_root / 'temporal_alignment_summary.csv'), 'rows': 33,
          'samples': SAMPLES, 'metric': 'mpjpe_mean'}, deps=('main_unity', 'eval_sync_calibration'), out=sync_root)
    hjob('eval_nview', 'eval_unity_nview', {'kind': 'nview', 'path': str(root / 'eval_nview'),
         'rows': 10, 'groups': 44}, resource='exclusive_gpu')
    add('eval_gate_interventions', [command('dual2pose.eval.evaluate_gate_interventions', '--cache-root', ctx['raw_cache'],
         '--dataset', 'unity', '--split', 'test', '--output', root / 'eval_gate_interventions',
         '--checkpoint', '{unity_checkpoint}', '--device', 'cuda:0')],
        {'kind': 'gate', 'path': str(root / 'eval_gate_interventions'), 'samples': SAMPLES, 'conditions': 5})

    for frontend in FRONTENDS:
        manifest = repo / f'logs/eval_unity_frontend_generalization/predictions/{frontend}/{frontend}_manifest.json'
        audit = root / f'frontend_{frontend}_assets.json'
        add(f'eval_prepare_frontend_{frontend}', [command(__name__, 'audit-frontend', '--source', manifest, '--output', audit)],
            {'kind': 'asset_audit', 'path': str(audit), 'manifests': 1}, deps=(), resource='cpu')
    models = ['frozen'] + [f'{frontend}_{scope}' for frontend in FRONTENDS for scope in ('heads_only', 'full')] + ['mixed_full']
    for model, frontend in itertools.product(models, ('sam3d', *FRONTENDS)):
        name = f'eval_frontend_{model}__on__{frontend}'
        model_id = 'main_unity' if model == 'frozen' else f'frontend_{model}'
        checkpoint = '{unity_checkpoint}' if model == 'frozen' else '{checkpoint:' + model_id + '}'
        env = {'EVAL_CKPT_PATH': checkpoint}
        deps = [model_id]
        if frontend != 'sam3d':
            env['FRONTEND_MANIFEST'] = str(repo / f'logs/eval_unity_frontend_generalization/predictions/{frontend}/{frontend}_manifest.json')
            deps.append(f'eval_prepare_frontend_{frontend}')
        else:
            env['FRONTEND_MANIFEST'] = ''
        hjob(name, 'eval_unity_frontend_generalization', csv_contract(name, 'frontend_generalization_summary_*.csv',
             conditions={'frontend_name': [frontend]}, allow_missing_canonical_mode=True), deps=deps, env=env)

    c07_frontend = Path(ctx.get('c07_frontend_root', repo / 'logs/ivc_review_revision_20260916/occluded_geometry_full/frontend'))
    c07_baseline = root / 'c07_baseline'
    add('eval_prepare_c07', [command(__name__, 'prepare-c07', '--source', c07_frontend, '--output', c07_baseline,
         '--raw-cache', ctx['raw_cache'], '--unity-checkpoint', '{unity_checkpoint}',
         '--residual-checkpoint', '{checkpoint:dlt_residual_unity}')],
        {'kind': 'asset_audit', 'path': str(c07_baseline / 'assets.json'), 'manifests': 3},
        deps=('main_unity', 'dlt_residual_unity'), resource='cpu')
    add('eval_c07', [command('dual2pose.eval.evaluate_occluded_geometry', '--frontend', c07_frontend,
        '--output', root / 'eval_c07', '--baseline', c07_baseline, '--device', 'cuda:0')],
        {'kind': 'c07', 'path': str(root / 'eval_c07'), 'actual_cells': 77, 'display_cells': 105, 'samples': SAMPLES, 'residual_checkpoint': '{checkpoint:dlt_residual_unity}'},
        deps=('main_unity', 'dlt_residual_unity', 'eval_prepare_c07'))
    return jobs


def _protocol(record, *, allow_missing_canonical_mode=False):
    value = record.get('evaluation_protocol')
    if isinstance(value, dict):
        if value != PROTOCOL:
            raise ValueError(f'Unexpected evaluation protocol: {value}')
    else:
        if value != PROTOCOL['name'] or record.get('joint_subset') != 'common13' or record.get('output_alignment') != 'none':
            raise ValueError(f'Missing or historical primary protocol: {record}')
        if record.get('canonical_mode') != 'per_frame' and not (allow_missing_canonical_mode and 'canonical_mode' not in record):
            raise ValueError('Missing or historical canonical mode')


def _finite(value, label, *, nonnegative=True):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'Invalid {label}: {value}')
    return value


def _one(pattern):
    path = Path(pattern)
    paths = list(path.parent.glob(path.name))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one artifact {pattern}; found {len(paths)}')
    return paths[0]


def _rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def _conditions(rows, conditions):
    if not conditions:
        return
    keys = list(conditions)
    def normalize(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    expected = set(itertools.product(*[list(map(normalize, conditions[key])) for key in keys]))
    actual = [tuple(normalize(row.get(key)) for key in keys) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f'Condition coverage mismatch: {keys}: {actual}')


def _metric_counts(metrics, samples, *, frames=30):
    for key in ('mpjpe', 'pa_mpjpe', 'acceleration_error'):
        _finite(metrics[key], key)
    for key, expected in {'sample_count': samples, 'point_count': samples * frames * 13,
                          'pa_point_count': samples * frames * 13,
                          'acceleration_point_count': samples * (frames - 2) * 13}.items():
        if metrics[key] != expected:
            raise ValueError(f'{key}: {metrics[key]} != {expected}')


def validate(job: dict, ctx: dict) -> dict:
    spec = job['validation']
    kind, path = spec['kind'], Path(spec['path'])
    artifacts: list[Path] = []
    result: dict[str, Any] = {'evaluation_protocol': PROTOCOL}
    expected_checkpoint = job.get('env', {}).get('EVAL_CKPT_PATH')
    expected_hash = _sha(expected_checkpoint) if expected_checkpoint else None

    def check_checkpoint(record):
        if expected_checkpoint and (Path(record.get('checkpoint', '')).resolve() != Path(expected_checkpoint).resolve()
                                    or record.get('checkpoint_sha256') != expected_hash):
            raise ValueError('Result checkpoint does not match the verified upstream checkpoint')

    if kind == 'csv':
        path = _one(str(path))
        rows = _rows(path)
        if len(rows) != spec['rows']:
            raise ValueError(f'Incomplete rows: {len(rows)} != {spec["rows"]}: {path}')
        for row in rows:
            _protocol(row, allow_missing_canonical_mode=spec.get('allow_missing_canonical_mode', False))
            check_checkpoint(row)
            _finite(row[spec['metric']], spec['metric'])
            if spec.get('sample_mode') != 'sum' and int(row['sample_count']) != spec['samples']:
                raise ValueError(f'Incomplete sample coverage: {row["sample_count"]}')
        if spec.get('sample_mode') == 'sum' and sum(int(row['sample_count']) for row in rows) != spec['samples']:
            raise ValueError('Incomplete angle-bin sample coverage')
        _conditions(rows, spec.get('conditions', {}))
        result['rows'] = len(rows)
        artifacts.append(path)
    elif kind == 'calibration':
        data = _json(path)
        if data.get('descriptor_protocol') != 'per_frame_body_velocity_v1' or data.get('split') != 'val' or data.get('limited_smoke_test') is not False:
            raise ValueError('Calibration must be complete validation with per-frame descriptors')
        expected = spec.get('samples', 128880)
        if data['validation_sample_count'] != expected or data['calibration_row_count'] != 11 * expected:
            raise ValueError('Incomplete calibration coverage')
        if tuple(data['injected_offsets']) != SYNC_OFFSETS:
            raise ValueError('Wrong calibration offsets')
        _finite(data['selected_threshold'], 'selected_threshold')
        result['threshold'] = data['selected_threshold']
        artifacts.append(path)
    elif kind == 'sync':
        provenance_path = path.parent / 'temporal_alignment_provenance.json'
        provenance = _json(provenance_path)
        _protocol(provenance)
        check_checkpoint(provenance)
        if provenance['limited_smoke_test'] or provenance['evaluated_samples'] != SAMPLES or provenance['metric_frame_count'] != 20:
            raise ValueError('Incomplete synchronization or wrong central scoring window')
        if _sha(provenance['threshold_file']) != provenance['threshold_sha256']:
            raise ValueError('Calibration artifact changed')
        rows = _rows(path)
        if len(rows) != 33:
            raise ValueError('Synchronization requires 33 cells')
        _conditions(rows, {'injected_offset': list(SYNC_OFFSETS), 'condition': ['uncorrected', 'automatic', 'oracle']})
        for row in rows:
            if int(row['sample_count']) != SAMPLES:
                raise ValueError('Incomplete synchronization sample coverage')
            _finite(row['mpjpe_mean'], 'mpjpe_mean')
        artifacts.extend([path, provenance_path])
        result['rows'] = len(rows)
    elif kind == 'nview':
        manifest = _json(path / 'nview_group_manifest.json')
        _protocol(manifest)
        if manifest['limited_smoke_test'] or manifest['proposed_group_count'] != 180 or manifest['evaluated_group_count'] != 180:
            raise ValueError('Incomplete N-view eligibility scan')
        if manifest['accepted_group_count'] != spec['groups'] or len(set(manifest['accepted_group_ids'])) != spec['groups']:
            raise ValueError('N-view cohort changed; inspect eligibility rather than silently changing the cohort')
        rows = _rows(path / 'nview_summary.csv')
        combinations = {(1, 'single_view')} | {(n, method) for n in (2, 3, 4) for method in
                        ('nview_canonical_mean', 'pairwise_canonfuse_mean', 'pairwise_oracle_select')}
        if {(int(row['n_views']), row['method']) for row in rows} != combinations:
            raise ValueError('Wrong N-view method/view combinations')
        if len(rows) != 10:
            raise ValueError('N-view requires ten result combinations')
        for row in rows:
            if int(row['group_count']) != spec['groups']:
                raise ValueError('Incomplete N-view groups')
            _finite(row['mpjpe_mean'], 'mpjpe_mean')
        per_group = _rows(path / 'nview_per_group.csv')
        actual_groups = [(row['group_id'], int(row['n_views']), row['method']) for row in per_group]
        expected_groups = {(group, n, method) for group in manifest['accepted_group_ids'] for n, method in combinations}
        if len(actual_groups) != len(set(actual_groups)) or set(actual_groups) != expected_groups:
            raise ValueError('Duplicate or missing N-view group results')
        for row in per_group:
            _finite(row['mpjpe'], 'group mpjpe')
        if len(per_group) != spec['groups'] * 10:
            raise ValueError('Incomplete N-view per-group artifact')
        efficiency = _rows(path / 'nview_efficiency.csv')
        if len(efficiency) != 3:
            raise ValueError('Missing N-view latency')
        if {int(row['n_views']) for row in efficiency} != {2, 3, 4}:
            raise ValueError('Wrong N-view latency conditions')
        for row in efficiency:
            _finite(row['latency_mean_ms'], 'latency_mean_ms')
        _protocol(_json(path / 'nview_provenance.json'))
        check_checkpoint(_json(path / 'nview_provenance.json'))
        artifacts.extend(path / name for name in ('nview_group_manifest.json', 'nview_summary.csv', 'nview_per_group.csv', 'nview_efficiency.csv', 'nview_provenance.json'))
        result.update(rows=10, groups=spec['groups'])
    elif kind == 'gate':
        config, progress = _json(path / 'config.json'), _json(path / 'progress.json')
        _protocol(config)
        check_checkpoint(config)
        if progress['status'] != 'complete_full_test' or progress['samples'] != SAMPLES or tuple(progress['conditions']) != GATE_CONDITIONS:
            raise ValueError('Incomplete frozen gate intervention study')
        for condition in GATE_CONDITIONS:
            p = path / f'{condition}.json'
            data = _json(p)
            _protocol(data)
            if data['condition'] != condition:
                raise ValueError('Wrong gate intervention condition')
            expected_methods = {f'{gate}_{residual}' for gate in ('dynamic', 'mean', 'left', 'right')
                                for residual in ('residual', 'no_residual')} | {'canonical_average'}
            if set(data['metrics']) != expected_methods:
                raise ValueError('Missing gate diagnostic reconstructions')
            for metrics in data['metrics'].values():
                _metric_counts(metrics['common13'], SAMPLES)
            if data['gate_diagnostics']['joint_frame_count'] != SAMPLES * 30 * 13:
                raise ValueError('Incomplete gate diagnostics')
            artifacts.append(p)
        artifacts.extend([path / 'config.json', path / 'progress.json'])
        result['conditions'] = 5
    elif kind == 'significance':
        data = _json(path)
        _protocol(data['provenance'])
        check_checkpoint(data['provenance'])
        if data['action_pair_row_count'] != SAMPLES or data['cluster_count'] != 16110 or len(data['within_bin']) != 6:
            raise ValueError('Incomplete significance analysis')
        if data['bootstrap_resamples'] != 10000:
            raise ValueError('Incorrect bootstrap budget')
        for row in data['within_bin']:
            for key in ('fused_mpjpe_mean', 'canonical_avg_mpjpe_mean', 'p_raw', 'p_holm'):
                _finite(row[key], key)
            for key in ('mean_gain_ci95_low', 'mean_gain_ci95_high'):
                _finite(row[key], key, nonnegative=False)
            if row['mean_gain_ci95_low'] > row['mean_gain_ci95_high']:
                raise ValueError('Reversed bootstrap interval')
        _finite(data['omnibus']['p_value'], 'omnibus p value')
        expected = 15 if data['omnibus']['significant_0_05'] else 0
        if len(data['pairwise_contrasts']) != expected:
            raise ValueError('Wrong conditional posthoc contrast coverage')
        pairs_path = path.parent / 'view_angle_per_pair_last.csv'
        if len(_rows(pairs_path)) != SAMPLES:
            raise ValueError('Incomplete significance per-pair exports')
        artifacts.extend([path, pairs_path, path.parent / 'view_angle_significance_last.csv', path.parent / 'view_angle_pairwise_contrasts_last.csv'])
    elif kind == 'c07':
        from dual2pose.eval.evaluate_occluded_geometry import EVALUATION_CONDITIONS, validate_condition_result
        protocol_path = path / 'protocol.json'
        protocol = _json(protocol_path)
        _protocol(protocol)
        check_checkpoint(protocol['frozen_models']['canonfuse3d'])
        residual = protocol['frozen_models']['dlt_mlp']
        if residual['checkpoint'] != spec['residual_checkpoint'] or residual['checkpoint_sha256'] != _sha(spec['residual_checkpoint']):
            raise ValueError('C07 residual checkpoint mismatch')
        progress = _json(path / 'progress.json')
        expected = {'status': 'complete', 'conditions': 7, 'unique_metric_cells': 77,
                    'display_cells_including_camera_invariant_reuse': 105, 'sample_count_per_cell': SAMPLES, 'split': 'test'}
        if any(progress.get(key) != value for key, value in expected.items()):
            raise ValueError('C07 completion coverage mismatch')
        for condition, _, _ in EVALUATION_CONDITIONS:
            condition_path = path / f'{condition}.json'
            if not validate_condition_result(_json(condition_path), _sha(protocol_path), condition, SAMPLES):
                raise ValueError(f'Invalid C07 condition: {condition_path}')
            artifacts.append(condition_path)
        artifacts.extend([protocol_path, path / 'progress.json'])
        result.update(actual_cells=77, display_cells=105)
    elif kind == 'asset_audit':
        data = _json(path)
        if data.get('status') != 'verified' or len(data.get('manifests', {})) != spec['manifests'] or not data.get('files'):
            raise ValueError(f'Incomplete raw asset audit: {path}')
        for filename, expected in {**data['manifests'], **data['files']}.items():
            if _sha(filename) != expected:
                raise ValueError(f'Raw asset changed after audit: {filename}')
        artifacts.append(path)
        result['files'] = len(data['files'])
    else:
        raise ValueError(f'Unknown validation kind: {kind}')
    result['artifacts'] = {str(p): _sha(p) for p in artifacts}
    return result


def _audit_manifest(manifest_path, *, rgb=False):
    import numpy as np
    manifest_path = Path(manifest_path)
    payload = _json(manifest_path)
    metadata = payload['metadata']
    entries = payload['entries']
    if len(entries) != 720 or metadata.get('limited_smoke_test') is not False:
        raise ValueError(f'Incomplete frontend: {manifest_path}')
    if rgb and (not metadata.get('complete') or metadata['completed_frame_count'] != 17089):
        raise ValueError(f'Incomplete RGB frontend: {manifest_path}')
    if not rgb and metadata.get('split') != 'test':
        raise ValueError(f'Frontend split must be test: {manifest_path}')
    keys, files, total_frames = set(), {}, 0
    for entry in entries:
        key = tuple(entry[k] for k in ('person_id', 'action_id', 'camera_id'))
        if key in keys:
            raise ValueError(f'Duplicate raw stream: {key}')
        keys.add(key)
        pose_path = input_path(entry['pose_path'])
        if not pose_path.is_absolute():
            pose_path = manifest_path.parent / pose_path
        with np.load(pose_path, allow_pickle=False) as arrays:
            pose, frames = arrays['pose'], arrays['frame_indices']
            if pose.shape != (len(frames), 15, 3) or not np.isfinite(pose).all() or len(np.unique(frames)) != len(frames):
                raise ValueError(f'Invalid raw pose or frame IDs: {pose_path}')
            if rgb:
                failed = arrays['detection_failed']
                if failed.dtype != np.bool_ or failed.shape != frames.shape or not (pose[failed] == 0).all():
                    raise ValueError(f'Invalid detection failure policy: {pose_path}')
            total_frames += len(frames)
        files[str(pose_path.resolve())] = _sha(pose_path)
    if rgb and total_frames != 17089:
        raise ValueError(f'RGB source frame coverage changed: {total_frames}')
    return files


def _audit(source, output, *, rgb):
    source = Path(source)
    manifests, files = {}, {}
    paths = [source / pattern / ratio / 'frontend_manifest.json' for pattern in ('random', 'distal', 'temporal') for ratio in ('0p5', '1')] if rgb else [source]
    for path in paths:
        files.update(_audit_manifest(path, rgb=rgb))
        manifests[str(path.resolve())] = _sha(path)
        if rgb:
            required = path.parent / 'required_frames_manifest.json'
            payload = _json(required)
            if payload['test_pair_count'] != SAMPLES or payload['stream_count'] != 720 or payload['unique_required_frame_count'] != 17089:
                raise ValueError(f'Wrong required frame cohort: {required}')
            files[str(required.resolve())] = _sha(required)
    _write(output, {'status': 'verified', 'representation': 'raw', 'manifests': manifests, 'files': files})


def _prepare_c07(args):
    import torch
    from dual2pose.eval.main_baseline_data import require_current_protocol
    from dual2pose.eval.evaluate_occluded_geometry import load_inputs
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'cache'
    target = Path(args.raw_cache).resolve()
    if cache.exists():
        if cache.resolve() != target:
            raise ValueError(f'Refuse to replace existing cache: {cache}')
    else:
        cache.symlink_to(target, target_is_directory=True)
    for name, checkpoint in [('canonfuse3d', args.unity_checkpoint), ('dlt_mlp', args.residual_checkpoint)]:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        # Fusion protocol lives inside hyperparameters; residual checkpoints put it at top level.
        if name == 'dlt_mlp':
            require_current_protocol(saved)
        else:
            from dual2pose.training_protocol import load_fusion_checkpoint
            load_fusion_checkpoint(checkpoint, device='cpu')
        _write(output / f'results/unity/{name}.json', {'checkpoint': str(Path(checkpoint).resolve()),
            'checkpoint_sha256': _sha(checkpoint), 'selected_epoch': saved['epoch'],
            'provenance': {'canonical_mode': 'per_frame', 'evaluation_protocol': PROTOCOL}})
    contract, pairs, stores, target_array, cached, frontend_manifests = load_inputs(Path(args.source), output)
    if contract['split'] != 'test' or len(pairs) != SAMPLES or contract['max_pairs'] is not None:
        raise ValueError('C07 requires complete fixed test cohort')
    manifests, files = {}, {}
    for condition, manifest in frontend_manifests.items():
        p = Path(args.source) / condition / 'frontend_manifest.json'
        manifests[str(p.resolve())] = _sha(p)
        for entry in manifest['entries']:
            files[str(input_path(entry['path']).resolve())] = entry['sha256']
    for filename in ('contract.json', 'required_frames_manifest.json'):
        p = Path(args.source) / filename
        files[str(p.resolve())] = _sha(p)
    _write(output / 'assets.json', {'status': 'verified', 'representation': 'raw', 'manifests': manifests, 'files': files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit-rgb', 'audit-frontend', 'merge-masking', 'prepare-c07'))
    for key in ('source', 'output', 'root', 'raw-cache', 'unity-checkpoint', 'residual-checkpoint'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.action in ('audit-rgb', 'audit-frontend'):
        _audit(args.source, args.output, rgb=args.action == 'audit-rgb')
    elif args.action == 'prepare-c07':
        _prepare_c07(args)
    else:
        paths = sorted(Path(args.root).glob('eval_mask_*/occlusion_summary_*.csv'))
        if len(paths) != 45:
            raise ValueError(f'Expected 45 completed masking cells, found {len(paths)}')
        rows = [row for path in paths for row in _rows(path)]
        output = Path(args.root) / 'pose_masking.csv'
        with output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
