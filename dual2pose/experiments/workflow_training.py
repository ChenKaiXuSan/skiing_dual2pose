"""Dependency jobs for the approved per-frame training matrix.

Builders are read-only. Worker entry points prepare new namespaces and call the
existing scientific implementations; they never launch another GPU scheduler.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from dual2pose.runtime_paths import input_path
from types import SimpleNamespace

FRONTENDS = ('videopose3d', 'poseformer', 'motionbert')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _protocol(record):
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    if record.get('evaluation_protocol') != EVALUATION_PROTOCOL:
        raise ValueError('Missing or incompatible per-frame/common13 protocol')


def _finite(value):
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    elif isinstance(value, list):
        for v in value:
            _finite(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite result')


def _ckpt(path):
    import torch
    from dual2pose.eval.main_baseline_data import require_current_protocol
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require_current_protocol(saved)
    if not saved.get('state_dict'):
        raise ValueError('Checkpoint lacks model weights')
    if any(not torch.isfinite(v).all() for v in saved['state_dict'].values() if torch.is_tensor(v)):
        raise ValueError('Nonfinite checkpoint weights')
    return saved, {'checkpoint': str(path.resolve()), 'checkpoint_sha256': _sha(path)}


def frontend_manifests(repo):
    repo = Path(repo)
    return {f: {s: str(repo / (f'logs/eval_unity_frontend_generalization/predictions/{f}/{f}_manifest.json'
            if s == 'test' else f'logs/ivc_p1/frontend_adaptation/predictions/{f}/{s}/{f}_manifest.json'))
            for s in ('train', 'val', 'test')} for f in FRONTENDS}


def build_jobs(ctx):
    repo, root = Path(ctx['repo']), Path(ctx['root'])
    python = ctx['python']
    shared = root / 'main_baselines'
    jobs = []

    def add(id, deps, commands, kind, *, resource='gpu', training=False, **spec):
        jobs.append({'id': id, 'deps': deps, 'commands': commands, 'env': {},
                     'resource': resource, 'family': 'training',
                     'validation': {'kind': kind, 'training': training, **spec}})

    def wrapper(kind, **spec):
        return [python, '-u', '-m', 'dual2pose.experiments.workflow_training',
                '--task', kind, '--spec', json.dumps(spec)]

    for dataset in ('unity', 'ski'):
        geom_spec = {'dataset': dataset, 'root': str(shared),
                     'source': str(Path(ctx['raw_cache']).parent / 'geometry' / dataset)}
        command = [python, '-u', '-m', 'dual2pose.experiments.run_geometry_baselines',
                   '--root', str(shared), '--dataset', dataset, '--device', 'cuda:0']
        add('geometry_' + dataset, ['external_prepare_cache'],
            [wrapper('prepare_geometry', **geom_spec), command + ['--mode', 'evaluate']],
            'geometry', **geom_spec)
        add('dlt_residual_' + dataset, ['geometry_' + dataset],
            [command + ['--mode', 'train', '--epochs', '100', '--seed', '42']],
            'residual', training=True, epochs=100, dataset=dataset, root=str(shared))
    for label, variant in [('dynamic_residual', 'full'), ('fixed_residual', 'fixed_gate'),
                           ('dynamic_no_residual', 'no_output_residual')]:
        out = root / 'gates' / label
        add('gate_' + label, ['main_unity'], [[python, '-u', '-m',
            'dual2pose.experiments.run_gate_ablation', '--cache-root', ctx['raw_cache'],
            '--output', str(out), '--variant', variant, '--device', 'cuda:0',
            '--epochs', '100', '--seed', '42', '--batch-size', '4096']],
            'gate', training=True, epochs=100, output=str(out / variant), variant=variant)
    manifests = frontend_manifests(repo)
    assets = root / 'frontend' / 'assets.json'
    add('frontend_assets', [], [wrapper('frontend_assets', manifests=manifests, output=str(assets))],
        'assets', resource='cpu', output=str(assets))
    for frontend, scope in [(f, s) for f in FRONTENDS for s in ('heads_only', 'full')] + [('mixed', 'full')]:
        out = root / 'frontend' / 'training' / f'{frontend}_{scope}'
        command = [python, '-u', '-m', 'dual2pose.train_frontend_adaptation', 'train.gpu=0',
                   'train.fold=0', 'train.seed=42', f'adaptation.frontend={frontend}',
                   f'adaptation.scope={scope}', 'adaptation.epochs=20',
                   'adaptation.learning_rate=0.0001', 'adaptation.source_checkpoint={unity_checkpoint}',
                   'data.unity.root_path=/home/kaixu_chen/skiing/data/skiing_unity_dataset',
                   'data.num_workers=16', 'data.batch_size=4096', f'log_path={out}']
        for f, splits in manifests.items():
            command.extend(f'adaptation.manifests.{f}.{s}={p}' for s, p in splits.items())
        add(f'frontend_{frontend}_{scope}', ['main_unity', 'frontend_assets'], [wrapper('new_output', output=str(out)), command],
            'frontend', training=True, epochs=20, output=str(out), frontend=frontend, scope=scope)
    for direction in ('forward', 'reverse'):
        for initialization in ('unity', 'random'):
            for seed in (0, 1, 2):
                id = f'pocket_{direction}_{initialization}_seed{seed}'
                spec = dict(output=str(root / 'pocket' / id), direction=direction,
                            initialization=initialization, seed=seed, epochs=300, patience=40,
                            checkpoint='{unity_checkpoint}')
                add(id, ['main_unity'], [wrapper('pocket_train', **spec)], 'pocket_train', training=True, **spec)
        for scope in ('heads_only', 'full'):
            id = f'pocket_adapt_{direction}_{scope}'
            spec = dict(output=str(root / 'pocket' / id), direction=direction, scope=scope,
                        seed=0, epochs=20, checkpoint='{unity_checkpoint}')
            add(id, ['main_unity'], [wrapper('pocket_adapt', **spec)], 'pocket_adapt', training=True, **spec)
    for dataset in ('unity', 'ski'):
        id = 'pocket_frozen_' + dataset
        spec = dict(output=str(root / 'pocket' / id), checkpoint='{' + dataset + '_checkpoint}', dataset=dataset)
        add(id, ['main_' + dataset], [wrapper('pocket_frozen', **spec)], 'pocket_frozen', **spec)
    return jobs


def _metric(record, samples):
    _finite(record)
    if not isinstance(record.get('mpjpe'), (float, int)) or record['mpjpe'] < 0:
        raise ValueError('Missing direct MPJPE')
    if record.get('sample_count') != samples or record.get('point_count') != samples * 30 * 13:
        raise ValueError('Incomplete common13 coverage')
    if record.get('acceleration_point_count') != samples * 28 * 13:
        raise ValueError('Incomplete acceleration coverage')


def _history(path, budget, key):
    rows = [_read_line for line in Path(path).read_text().splitlines() if line
            for _read_line in [json.loads(line)]]
    if [r['epoch'] for r in rows] != list(range(budget)):
        raise ValueError('Training epoch budget/history incomplete')
    for row in rows:
        _finite(row)
        if row[key] <= 0:
            raise ValueError('Empty training coverage')
    return rows


def validate_pocket_record(record, spec):
    _protocol(record)
    _finite(record)
    if record.get('epochs') != spec['epochs']:
        raise ValueError('Pocket training budget mismatch')
    for name, starts in record['window_starts'].items():
        lo, hi = record['blocks'][name]
        if not starts or any(s < lo or s + 30 > hi for s in starts):
            raise ValueError('Empty windows or temporal boundary crossing')
    blocks = sorted(record['blocks'].values())
    if any(a[1] > b[0] for a, b in zip(blocks, blocks[1:])):
        raise ValueError('Temporal blocks overlap')
    if record.get('valid_test_points', 0) <= 0:
        raise ValueError('No valid common13 test points')
    for method in ('model', 'gate_only'):
        if not isinstance(record['metrics'][method].get('mpjpe_mm'), (int, float)):
            raise ValueError('Missing Pocket direct common13 score')
    if spec['kind'] == 'pocket_train':
        history = record['history']
        ran, best, patience = record['epochs_run'], record['best_epoch'], spec['patience']
        if record.get('patience') != patience or not 1 <= best <= ran <= spec['epochs']:
            raise ValueError('Invalid Pocket training budget/selection')
        if [x['epoch'] for x in history] != list(range(1, ran + 1)):
            raise ValueError('Incomplete Pocket epoch history')
        if ran < spec['epochs'] and ran - best != patience:
            raise ValueError('Pocket early stopping did not exhaust patience')
        # Reproduce the existing training loop improvement tolerance, including ties.
        selected, value, bad = 0, math.inf, 0
        for row in history:
            if row['val_common13_mpjpe_mm'] < value - 1e-3:
                selected, value, bad = row['epoch'], row['val_common13_mpjpe_mm'], 0
            else:
                bad += 1
            if bad >= patience and row['epoch'] != ran:
                raise ValueError('Pocket history continued after early stopping')
        if selected != best or record['selection'] != 'minimum_validation_common13':
            raise ValueError('Pocket selected checkpoint differs from validation history')
    elif record.get('epochs_run') != 20 or record.get('selection') != 'fixed_final_epoch':
        raise ValueError('Pocket adaptation must use fixed final epoch20')


def validate(job, ctx):
    spec = job['validation']
    kind = spec['kind']
    metadata = {}
    if kind == 'assets':
        data = _read(spec['output'])
        if data.get('verified_manifests') != 9 or data.get('streams', 0) <= 0:
            raise ValueError('Incomplete frontend raw asset validation')
        metadata['manifest_sha256'] = _sha(spec['output'])
    elif kind in ('geometry', 'residual'):
        dataset, root = spec['dataset'], Path(spec['root'])
        samples = 64440 if dataset == 'unity' else 30
        methods = ('dlt', 'robust_dlt') if kind == 'geometry' else ('dlt_mlp',)
        for method in methods:
            record = _read(root / 'results' / dataset / (method + '.json'))
            _protocol(record['provenance'])
            _metric(record['metrics']['common13'], samples)
            if kind == 'residual':
                if record['training']['epochs'] != 100:
                    raise ValueError('Residual budget mismatch')
                name = 'dlt_mlp' if dataset == 'unity' else 'dlt_mlp_root_relative'
                rows = _history(root / 'training' / dataset / name / 'history.jsonl', 100, 'training_samples')
                saved, metadata = _ckpt(record['checkpoint'])
                if metadata['checkpoint_sha256'] != record['checkpoint_sha256']:
                    raise ValueError('Residual checkpoint hash mismatch')
                expected_epoch = min(rows, key=lambda r: r['validation']['common13']['mpjpe'])['epoch'] if dataset == 'unity' else 99
                if saved['epoch'] != expected_epoch or record['selected_epoch'] != expected_epoch:
                    raise ValueError('Residual checkpoint selection mismatch')
    elif kind == 'gate':
        output = Path(spec['output'])
        record = _read(output / 'metrics.json')
        config = record['configuration']
        _protocol(config)
        if config['epochs'] != 100 or config.get('smoke_steps') != 0:
            raise ValueError('Gate budget/smoke result mismatch')
        rows = _history(output / 'history.jsonl', 100, 'train_samples')
        if any(row['train_samples'] != 192512 for row in rows):
            raise ValueError('Incomplete gate training coverage')
        _metric(record['metrics']['common13'], 64440)
        saved, metadata = _ckpt(output / 'best.pt')
        expected = min(rows, key=lambda r: r['validation']['common13']['mpjpe'])['epoch']
        if saved['epoch'] != expected or record['selected_epoch'] != expected:
            raise ValueError('Gate validation selection mismatch')
        if record['checkpoint_sha256'] != metadata['checkpoint_sha256']:
            raise ValueError('Gate checkpoint hash mismatch')
    elif kind == 'frontend':
        output = Path(spec['output'])
        record = _read(output / 'adaptation_complete.json')
        init = _read(output / 'adaptation_init.json')
        _protocol(record); _protocol(init)
        if init['epochs'] != 20 or init['frontend'] != spec['frontend'] or init['scope'] != spec['scope']:
            raise ValueError('Frontend adaptation budget/configuration mismatch')
        _finite(record)
        if record.get('best_model_score') is None or record.get('trainer_global_step', 0) <= 0:
            raise ValueError('Frontend adaptation lacks score/optimizer steps')
        rows = []
        for file in output.glob('csv_logs/**/metrics.csv'):
            with file.open() as handle:
                rows.extend(csv.DictReader(handle))
        epochs = {int(float(r['epoch'])) for r in rows if r.get('epoch') and r.get('val/common13_mpjpe')}
        if epochs != set(range(20)):
            raise ValueError('Frontend full 20-epoch validation history missing')
        saved, metadata = _ckpt(record['best_model_path'])
        last, _ = _ckpt(record['last_model_path'])
        if last.get('global_step') != record['trainer_global_step'] or last.get('epoch') not in (19, 20):
            raise ValueError('Frontend terminal checkpoint budget incomplete')
        validation_rows = [r for r in rows if r.get('val/common13_mpjpe') and r.get('epoch')]
        best = min(validation_rows, key=lambda r: float(r['val/common13_mpjpe']))
        if saved['epoch'] != int(float(best['epoch'])) or not math.isclose(record['best_model_score'], float(best['val/common13_mpjpe']), rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError('Frontend checkpoint differs from minimum validation score')
        metadata['best_epoch'] = saved['epoch']
    elif kind in ('pocket_train', 'pocket_adapt'):
        record = _read(Path(spec['output']) / 'summary.json')
        validate_pocket_record(record, spec)
        saved, metadata = _ckpt(record['checkpoint'])
        contract = saved.get('workflow_training', {})
        if contract.get('epochs_run') != record['epochs_run'] or contract.get('selected_epoch') != record['best_epoch'] or contract.get('budget') != spec['epochs']:
            raise ValueError('Pocket checkpoint budget/selection metadata mismatch')
        if metadata['checkpoint_sha256'] != record['checkpoint_sha256']:
            raise ValueError('Pocket checkpoint hash mismatch')
        metadata['valid_test_points'] = record['valid_test_points']
    elif kind == 'pocket_frozen':
        output = Path(spec['output'])
        full, blocks = _read(output / 'full' / 'summary.json'), _read(output / 'blocks.json')
        _protocol(full); _protocol(blocks)
        _finite(full['metrics']); _finite(blocks)
        if full['counts']['windows'] <= 0 or set(blocks['directions']) != {'forward', 'reverse'}:
            raise ValueError('Incomplete Pocket frozen coverage')
        for block in blocks['directions'].values():
            if not block['window_starts'] or block['valid_test_points'] <= 0:
                raise ValueError('Empty Pocket test block')
    else:
        raise ValueError('Unknown training validation kind: ' + kind)
    return metadata


def prepare_geometry(spec):
    from dual2pose.experiments.run_geometry_baselines import _load_verified_split
    root, dataset = Path(spec['root']), spec['dataset']
    target, source = root / 'geometry' / dataset, Path(spec['source']).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Raw calibrated geometry unavailable: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError('Existing geometry namespace points to a different source')
    elif target.exists():
        raise FileExistsError(f'Preserving existing geometry namespace: {target}')
    else:
        target.symlink_to(source, target_is_directory=True)
    for split in (('train', 'val', 'test') if dataset == 'unity' else ('train', 'test')):
        _load_verified_split(root, dataset, split)


def verify_frontend_assets(spec):
    import numpy as np
    records, streams = {}, 0
    for frontend, splits in spec['manifests'].items():
        seen = set()
        for split, path in splits.items():
            path = Path(path)
            data = _read(path)
            entries = data.get('entries', [])
            if len(entries) != {'train': 2160, 'val': 1440, 'test': 720}[split] or data.get('frontend_name') != frontend or data['metadata'].get('limited_smoke_test'):
                raise ValueError(f'Incomplete frontend manifest: {path}')
            digest = hashlib.sha256()
            for entry in entries:
                key = (entry['person_id'], entry['action_id'], entry['camera_id'])
                if key in seen:
                    raise ValueError('Frontend streams overlap across train/val/test')
                seen.add(key)
                pose_path = input_path(entry['pose_path'])
                if not pose_path.is_absolute():
                    pose_path = path.parent / pose_path
                with np.load(pose_path, allow_pickle=False) as arrays:
                    if not {'pose', 'frame_indices'}.issubset(arrays.files):
                        raise ValueError(f'Missing frontend poses/frame indices: {pose_path}')
                    pose, frames = arrays['pose'], arrays['frame_indices']
                    if pose.shape != (len(frames), 15, 3) or len(frames) == 0 or len(np.unique(frames)) != len(frames):
                        raise ValueError(f'Invalid frontend stream coverage: {pose_path}')
                    for name in arrays.files:
                        a = arrays[name]
                        if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                            raise ValueError(f'Nonfinite frontend raw file: {pose_path}')
                digest.update(str(pose_path).encode()); digest.update(_sha(pose_path).encode())
            streams += len(entries)
            records[str(path)] = {'sha256': _sha(path), 'streams': len(entries), 'pose_set_sha256': digest.hexdigest()}
    _write(spec['output'], {'verified_manifests': len(records), 'streams': streams, 'manifests': records})


def _pocket_data():
    from dual2pose.pocket3 import evaluate as ep
    from dual2pose.pocket3 import train as tr
    args = SimpleNamespace(dataset_root=ep.DEFAULT_ROOT, session='chen_4', frame_height=1080., window=30)
    return tr.load_frames(args)


def _pocket_blocks(data, direction, adaptation=False):
    import numpy as np
    L, R, j, gt, usable, anchor = data
    F = len(usable)
    if adaptation:
        cut = int(round(.6 * F))
        return {'train': (0, cut), 'test': (cut, F)} if direction == 'forward' else {'train': (F-cut, F), 'test': (0, F-cut)}
    indices = np.where(usable)[0]
    if len(indices) < 3:
        raise ValueError('Pocket recording lacks usable train/val/test frames')
    cut1, cut2 = int(indices[int(.6 * len(indices))]), int(indices[int(.72 * len(indices))])
    return ({'train': (0, cut1), 'val': (cut1, cut2), 'test': (cut2, F)} if direction == 'forward'
            else {'train': (F-cut1, F), 'val': (F-cut2, F-cut1), 'test': (0, F-cut2)})


def _pocket_sets(data, blocks, adaptation=False):
    from dual2pose.pocket3 import train as tr
    L, R, j, gt, usable, anchor = data
    starts = {name: tr.windows_in(usable, anchor, *block, 30, 5 if name == 'train' and not adaptation else 30)
              for name, block in blocks.items()}
    sets = {name: tr.gather(L, R, j, gt, s, 30) for name, s in starts.items()}
    return starts, sets


def run_pocket(kind, spec):
    import numpy as np
    import torch
    from dual2pose.pocket3 import evaluate as ep, train as tr, adapt as ad
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    source = Path(spec['checkpoint'])
    _, source_meta = _ckpt(source)
    base = ep.load_model(source, device)
    base.log = lambda *a, **k: None
    data = _pocket_data()
    if kind == 'pocket_frozen':
        ep.main(['--ckpt', spec['dataset'] + '=' + str(source), '--output-dir', str(output / 'full'), '--gpu', '0'])
        directions = {}
        for direction in ('forward', 'reverse'):
            blocks = _pocket_blocks(data, direction)
            starts, sets = _pocket_sets(data, {'test': blocks['test']})
            directions[direction] = {'block': list(blocks['test']), 'window_starts': starts['test'],
                'valid_test_points': int(sets['test'][3][..., ep.COMMON13].sum()),
                'metrics': ad.evaluate(base, *sets['test'], device)}
        _write(output / 'blocks.json', {'evaluation_protocol': dict(EVALUATION_PROTOCOL),
               'source': source_meta, 'directions': directions})
        return
    adaptation = kind == 'pocket_adapt'
    blocks = _pocket_blocks(data, spec['direction'], adaptation)
    starts, sets = _pocket_sets(data, blocks, adaptation)
    seed = spec['seed']
    torch.manual_seed(seed); np.random.seed(seed)
    if adaptation:
        model = ad.adapt(base, *sets['train'], device, spec['scope'], 20, 1e-4, seed)
        info = {'epochs_run': 20, 'best_epoch': 20, 'selection': 'fixed_final_epoch'}
    else:
        initial = base if spec['initialization'] == 'unity' else tr.fresh_model(device)
        model, info = tr.train(initial, sets['train'], sets['val'], device, 300,
                               1e-4 if spec['initialization'] == 'unity' else 1e-3, 40, seed)
        info['selection'] = 'minimum_validation_common13'
    checkpoint = output / 'selected.pt'
    tr.save_weights(model, checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    payload['workflow_training'] = {'epochs_run': info['epochs_run'], 'selected_epoch': info['best_epoch'],
        'budget': spec['epochs'], 'selection': info['selection'], 'seed': seed, 'direction': spec['direction']}
    torch.save(payload, checkpoint)
    # The selected weights produce all test metrics together after selection.
    metrics = ad.evaluate(model, *sets['test'], device)
    result = {**spec, **info, 'evaluation_protocol': dict(EVALUATION_PROTOCOL),
              'blocks': {k: list(v) for k, v in blocks.items()}, 'window_starts': starts,
              'valid_test_points': int(sets['test'][3][..., ep.COMMON13].sum()),
              'metrics': metrics, 'test_cohort_sha256': hashlib.sha256(np.asarray(starts['test'], dtype='<i8').tobytes() + sets['test'][3].tobytes()).hexdigest(),
              'source_sha256': {str(Path(m.__file__)): _sha(m.__file__) for m in (ep, tr, ad)},
              'gt_sha256': {cam: _sha(ep.DEFAULT_ROOT / 'gt/pocket_aligned' / f'chen_4_{cam}.npz') for cam in ('pocket_1', 'pocket_2')},
              'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha(checkpoint),
              'source_checkpoint': source_meta, 'primary_metric': 'mpjpe_mm'}
    validate_pocket_record(result, {'kind': kind, **spec})
    _write(output / 'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec)
    if args.task == 'prepare_geometry':
        prepare_geometry(spec)
    elif args.task == 'new_output':
        Path(spec['output']).mkdir(parents=True, exist_ok=False)
    elif args.task == 'frontend_assets':
        verify_frontend_assets(spec)
    elif args.task in ('pocket_train', 'pocket_adapt', 'pocket_frozen'):
        run_pocket(args.task, spec)
    else:
        parser.error('Unknown task')


if __name__ == '__main__':
    main()
