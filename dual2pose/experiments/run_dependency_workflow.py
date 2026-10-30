"""Persistent, artifact-gated two-GPU execution of the approved per-frame study."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

FAMILIES = ('training', 'evaluations', 'external')
SUCCESS = 'complete'
TERMINAL = {'complete', 'failed', 'blocked'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_graph(jobs):
    by_id = {j['id']: j for j in jobs}
    if len(by_id) != len(jobs):
        raise ValueError('Duplicate job ids')
    visited, active = set(), set()
    def visit(key):
        if key not in by_id: raise ValueError(f'Missing dependency: {key}')
        if key in active: raise ValueError(f'Dependency cycle at {key}')
        if key in visited: return
        active.add(key)
        for dep in by_id[key].get('deps', []): visit(dep)
        active.remove(key); visited.add(key)
    for key in by_id: visit(key)


def dependency_blocks(jobs, states):
    blocked = {k for k, v in states.items() if v['status'] in {'failed', 'blocked'}}
    newly = {}
    while True:
        count = len(blocked)
        for job in jobs:
            key = job['id']
            if states[key]['status'] != 'pending' or key in blocked: continue
            bad = [d for d in job.get('deps', []) if d in blocked]
            if bad: blocked.add(key); newly[key] = bad
        if len(blocked) == count: return newly


def ready_assignments(jobs, states, free_gpus, cpu_available, all_gpus=(0, 1)):
    ready = [j for j in jobs if states[j['id']]['status'] == 'pending'
             and all(states[d]['status'] == SUCCESS for d in j.get('deps', []))]
    ready.sort(key=lambda j: (j.get('resource') == 'exclusive_gpu', -j.get('priority', 0)))
    free = list(free_gpus); result = []
    for job in ready:
        resource = job.get('resource', 'gpu')
        if resource == 'external': continue
        if resource == 'cpu':
            if cpu_available:
                result.append((job['id'], None)); cpu_available = False
        elif resource == 'exclusive_gpu':
            if set(free) == set(all_gpus) and not result and cpu_available:
                return [(job['id'], free[0])]
        elif free:
            result.append((job['id'], free.pop(0)))
    return result


def resolve_tokens(value, metadata):
    if isinstance(value, list): return [resolve_tokens(v, metadata) for v in value]
    if isinstance(value, dict): return {k: resolve_tokens(v, metadata) for k, v in value.items()}
    if not isinstance(value, str): return value
    def replace(match):
        name = match.group(1)
        key = {'unity_checkpoint': 'main_unity', 'ski_checkpoint': 'main_ski'}.get(name)
        if key is None and name.startswith('checkpoint:'): key = name.split(':', 1)[1]
        checkpoint = metadata.get(key, {}).get('checkpoint')
        if not checkpoint: raise ValueError(f'Missing verified checkpoint: {name}')
        return str(checkpoint)
    return re.sub(r'\{(unity_checkpoint|ski_checkpoint|checkpoint:[^}]+)\}', replace, value)


def verify_checkpoint_tokens(job, metadata):
    tokens = re.findall(r'\{(unity_checkpoint|ski_checkpoint|checkpoint:[^}]+)\}', json.dumps(job))
    for token in set(tokens):
        key = {'unity_checkpoint': 'main_unity', 'ski_checkpoint': 'main_ski'}.get(token)
        key = key or token.split(':', 1)[1]
        record = metadata.get(key, {})
        checkpoint = record.get('checkpoint')
        if not checkpoint or not record.get('checkpoint_sha256') or sha(checkpoint) != record['checkpoint_sha256']:
            raise ValueError(f'Upstream checkpoint hash changed or missing: {key}')


def verify_budget(saved, rows, epochs, steps):
    logged = {int(r['epoch']) for r in rows if r.get('train/loss_epoch')}
    processed = saved['loops']['fit_loop']['epoch_progress']['total']['processed']
    if logged != set(range(epochs)) or processed != epochs or saved['global_step'] != steps:
        raise ValueError(f'Incomplete training budget: logged={len(logged)}, processed={processed}, steps={saved["global_step"]}')
    if saved['epoch'] not in (epochs - 1, epochs):
        raise ValueError('Unexpected terminal checkpoint epoch')


def family_module(name):
    return importlib.import_module('dual2pose.experiments.workflow_' + name)


def verify_main(job, context):
    import torch
    dataset = job['validation']['dataset']; root = Path(context['main_root'])
    status = json.loads((root / f'{dataset}_status.json').read_text())
    if status.get('return_code') != 0: raise ValueError(f'{dataset} training did not exit successfully')
    last = root / dataset / 'checkpoints/last.ckpt'
    saved = torch.load(last, map_location='cpu', weights_only=False)
    rows = list(csv.DictReader((root / dataset / 'csv_logs/train/version_0/metrics.csv').open()))
    if dataset != 'unity':
        verify_budget(saved, rows, 100, 3600)
    protocol = 'common13_direct_per_frame_v1'
    if saved.get('evaluation_protocol', {}).get('name') != protocol: raise ValueError('Main checkpoint protocol mismatch')
    test = [r for r in rows if r.get('test/mpjpe')]
    if not test or not math.isfinite(float(test[-1]['test/mpjpe'])): raise ValueError('Missing finite final test result')
    checkpoint = last
    if dataset == 'unity':
        # Some Lightning versions refresh save_last only when top-k changes.
        # Prove the completed fit from every logged epoch/step, then verify the
        # validation-selected weights independently; never relabel stale weights.
        train = [r for r in rows if r.get('train/loss_epoch')]
        val = [r for r in rows if r.get('val/mpjpe')]
        for history, metric in ((train, 'train/loss_epoch'), (val, 'val/mpjpe')):
            if [int(r['epoch']) for r in history] != list(range(100)):
                raise ValueError('Incomplete Unity epoch history')
            if any(int(r['step']) != (int(r['epoch']) + 1) * 47 - 1 or
                   not math.isfinite(float(r[metric])) for r in history):
                raise ValueError('Unity step budget or finite metrics mismatch')
        if int(test[-1]['epoch']) != 100 or int(test[-1]['step']) != 4700:
            raise ValueError('Unity test did not follow the full training budget')
        if '`max_epochs=100` reached.' not in (root / 'unity.log').read_text():
            raise ValueError('Missing Unity fit completion evidence')
        summary = json.loads((root / dataset / 'test_metrics_by_action.json').read_text())
        if summary['test_checkpoint_policy'] != 'best': raise ValueError('Unity must select validation-best')
        checkpoint = Path(summary['best_model_path'])
        selected = torch.load(checkpoint, map_location='cpu', weights_only=False)
        best = min(val, key=lambda r: float(r['val/mpjpe']))
        if (selected['epoch'] != int(best['epoch']) or
                selected['global_step'] != (int(best['epoch']) + 1) * 47 or
                not math.isclose(summary['best_model_score'], float(best['val/mpjpe']), rel_tol=1e-6)):
            raise ValueError('Unity checkpoint differs from validation minimum')
        from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
        if (selected.get('canonical_mode') != 'per_frame' or
                selected.get('evaluation_protocol') != EVALUATION_PROTOCOL or
                summary.get('evaluation_protocol') != EVALUATION_PROTOCOL):
            raise ValueError('Unity selected checkpoint/test protocol mismatch')
        if not selected.get('state_dict') or any(not torch.isfinite(v).all() for v in selected['state_dict'].values() if torch.is_tensor(v)):
            raise ValueError('Missing or nonfinite selected Unity weights')
        if (summary.get('sample_count') != 64440 or
                sum(v['sample_count'] for v in summary['per_action'].values()) != 64440 or
                not math.isclose(summary['mpjpe'], float(test[-1]['test/mpjpe']), rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError('Unity final-test coverage/metric mismatch')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'completed_epochs': 100, 'test_mpjpe': float(test[-1]['test/mpjpe']), 'verified_at': now(),
            'terminal_checkpoint_epoch': saved['epoch'],
            'metrics_csv_sha256': sha(root / dataset / 'csv_logs/train/version_0/metrics.csv')}


def verify_core(job, context):
    import torch
    root = Path(context['core_root']); spec = job['validation']; kind = spec['kind']; dataset = spec.get('dataset', 'ski')
    count = 64440 if dataset == 'unity' else 30
    if kind in ('local', 'reference'):
        method = spec.get('method', 'canonfuse3d')
        record = json.loads((root / 'results' / dataset / f'{method}.json').read_text())
        metric = record['metrics']['common13']; protocol = record['provenance']['evaluation_protocol']['name']
        if kind == 'local':
            last = torch.load(root / 'training' / dataset / method / 'last.pt', map_location='cpu', weights_only=False)
            history = [json.loads(line) for line in (root / 'training' / dataset / method / 'history.jsonl').read_text().splitlines()]
            if last['epoch'] != 99 or [r['epoch'] for r in history] != list(range(100)):
                raise ValueError('Core training budget mismatch')
            if not all(math.isfinite(r['training_loss']) for r in history): raise ValueError('Nonfinite training loss')
        checkpoint = record.get('checkpoint')
        if checkpoint and sha(checkpoint) != record['checkpoint_sha256']: raise ValueError('Core checkpoint hash mismatch')
    else:
        record = json.loads((root / 'deciwatch' / dataset / 'report.json').read_text())
        if record['status'] != 'complete': raise ValueError('DeciWatch incomplete')
        metric = record['metrics']['deciwatch']; protocol = record['evaluation_protocol']['name']
        history = [json.loads(line) for line in (root / 'deciwatch' / dataset / 'epochs.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in history] != list(range(1, 71)): raise ValueError('DeciWatch budget mismatch')
        checkpoint = str(root / 'deciwatch' / dataset / 'best.pth')
        if sha(checkpoint) != record['checkpoint_sha256']: raise ValueError('DeciWatch checkpoint hash mismatch')
    if protocol != 'common13_direct_per_frame_v1' or metric['sample_count'] != count or metric['point_count'] != count * 30 * 13:
        raise ValueError('Core metric protocol/coverage mismatch')
    if not all(math.isfinite(metric[k]) for k in ('mpjpe', 'acceleration_error')): raise ValueError('Nonfinite core metrics')
    return {'checkpoint': checkpoint, 'checkpoint_sha256': sha(checkpoint) if checkpoint else None, 'primary_mpjpe': metric['mpjpe'], 'sample_count': count, 'verified_at': now()}


def pid_token(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError): return None


def alive(pid, token=None):
    if not pid: return False
    current = pid_token(pid)
    return current is not None and (token is None or current == token)


def gpu_inventory():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    inventory = {}
    uuids = {}
    for line in text.strip().splitlines():
        idx, uuid, free = [v.strip() for v in line.split(',')]
        inventory[int(idx)] = {'free_mb': int(free), 'processes': []}; uuids[uuid] = int(idx)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    for line in apps.strip().splitlines():
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid in uuids: inventory[uuids[uuid]]['processes'].append(int(pid))
    return inventory


def imported_jobs(context):
    jobs = []
    for dataset in ('unity', 'ski'):
        jobs.append({'id': 'main_' + dataset, 'deps': [], 'commands': [], 'resource': 'external', 'family': 'main', 'validation': {'dataset': dataset}})
    queue = json.loads((Path(context['core_root']) / 'queue.json').read_text())
    for record in queue['jobs']:
        jobs.append({'id': 'core_' + record['name'], 'deps': ['main_ski'] if record['kind'] == 'reference' else [],
                     'commands': [record['command']], 'env': {}, 'resource': 'gpu', 'family': 'core',
                     'validation': {'kind': record['kind'], 'dataset': record.get('dataset', 'ski'), 'method': record.get('method', 'canonfuse3d')},
                     'external_name': record['name']})
    return jobs


def write_event(root, payload):
    with (root / 'events.jsonl').open('a') as handle: handle.write(json.dumps({'time': now(), **payload}) + '\n')


def build(context):
    root = Path(context['root'])
    if (root / 'manifest.json').exists(): raise FileExistsError('Workflow already exists; use run to resume')
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'context.json', context)
    jobs = imported_jobs(context)
    for family in FAMILIES: jobs.extend(family_module(family).build_jobs(context))
    validate_graph(jobs)
    for job in jobs:
        # Start the longest independent TTA workload early to reduce the tail.
        if job['id'] == 'external_stride_unity': job['priority'] = 60
        elif job['resource'] == 'cpu': job['priority'] = 20
    atomic_json(root / 'manifest.json', {'created_at': now(), 'protocol': 'common13_direct_per_frame_v1', 'jobs': jobs})
    states = {j['id']: ({'status': 'blocked', 'error': j['blocked_reason']} if j.get('blocked_reason') else {'status': 'external' if j['family'] in ('main', 'core') else 'pending'}) for j in jobs}
    atomic_json(root / 'state.json', {'created_at': now(), 'jobs': states, 'metadata': {}})
    source = root / 'source'; source.mkdir()
    repo = Path(context['repo'])
    for directory in ('dual2pose', 'configs'):
        shutil.copytree(repo / directory, source / directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for directory in ('logs', 'ckpt', '.cache', 'data'):
        if (repo / directory).exists(): (source / directory).symlink_to(repo / directory, target_is_directory=True)
    hashes = {str(p.relative_to(source)): sha(p) for directory in ('dual2pose', 'configs') for p in (source / directory).rglob('*') if p.is_file()}
    atomic_json(root / 'source_hashes.json', hashes)
    print(json.dumps({'root': str(root), 'jobs': len(jobs), 'by_family': {f: sum(j['family'] == f for j in jobs) for f in ('main', 'core') + FAMILIES}}, indent=2))


def worker(root, key, gpu):
    root = Path(root); context = json.loads((root / 'context.json').read_text())
    job = next(j for j in json.loads((root / 'manifest.json').read_text())['jobs'] if j['id'] == key)
    state = json.loads((root / 'state.json').read_text()); started = now()
    output = root / 'tasks' / key; output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'completion.json'
    if result_path.exists(): raise FileExistsError(result_path)
    try:
        verify_checkpoint_tokens(job, state['metadata'])
        resolved = resolve_tokens(job, state['metadata'])
        env = os.environ.copy(); env.update({k: str(v) for k, v in resolved.get('env', {}).items()})
        env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
        # Freeze Python/config code while reusing unchanged native source data.
        source = root / 'source'; env['PYTHONPATH'] = str(source)
        commands = [[str(v).replace(context['repo'] + '/dual2pose/', str(source / 'dual2pose') + '/') for v in command] for command in resolved['commands']]
        atomic_json(output / 'invocation.json', {'commands': commands, 'environment': {**resolved.get('env', {}), 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES']}, 'started_at': started})
        with (output / 'run.log').open('x') as log:
            for command in commands:
                execution = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                if execution.returncode: raise RuntimeError(f'Command exited {execution.returncode}: {command}')
        atomic_json(output / 'process.json', {'return_code': 0, 'status': 'verifying', 'time': now()})
        if job['family'] == 'main': metadata = verify_main(resolved, context)
        elif job['family'] == 'core': metadata = verify_core(resolved, context)
        else: metadata = family_module(job['family']).validate(resolved, context)
        atomic_json(result_path, {'status': 'complete', 'started_at': started, 'finished_at': now(), 'metadata': metadata or {}})
    except Exception:
        atomic_json(result_path, {'status': 'failed', 'started_at': started, 'finished_at': now(), 'error': traceback.format_exc()})
        raise


def observe_external(jobs, states, metadata, context):
    reservations = {}
    main = Path(context['main_root']); core = Path(context['core_root'])
    queue = json.loads((core / 'queue.json').read_text())
    if queue.get('status') == 'running' and alive(queue.get('pid')): reservations[1] = 'existing_core_queue'
    for dataset, gpu in (('unity', 0), ('ski', 1)):
        status = json.loads((main / f'{dataset}_status.json').read_text())
        if 'return_code' not in status and alive(status.get('launcher_pid')): reservations[gpu] = 'existing_' + dataset
    for job in jobs:
        key = job['id']
        if states[key]['status'] != 'external': continue
        try:
            if job['family'] == 'main':
                raw = json.loads((main / f'{job["validation"]["dataset"]}_status.json').read_text())
                if 'return_code' not in raw and alive(raw.get('launcher_pid')): continue
                if 'return_code' not in raw: raise RuntimeError('Main launcher exited without a process result')
                result = verify_main(job, context)
            else:
                record = next(r for r in queue['jobs'] if r['name'] == job['external_name'])
                if record['status'] == 'artifacts_verified': result = verify_core(job, context)
                elif record['status'] == 'pending' and not alive(queue.get('pid')):
                    # Only never-started jobs are adopted; failed outputs are preserved.
                    states[key] = {'status': 'pending', 'adopted_at': now()}; job['resource'] = 'gpu'; continue
                elif record['status'] in ('failed',) or (record['status'] == 'running' and not alive(queue.get('pid'))):
                    raise RuntimeError(queue.get('error', 'Existing core launcher stopped before verification'))
                else: continue
            states[key] = {'status': 'complete', 'verified_at': now(), 'imported': True}; metadata[key] = result
        except Exception:
            states[key] = {'status': 'failed', 'finished_at': now(), 'error': traceback.format_exc(), 'imported': True}
    return reservations


def run(root, poll_seconds=10):
    root = Path(root); lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / 'manifest.json').read_text()); jobs = manifest['jobs']
    context = json.loads((root / 'context.json').read_text()); state = json.loads((root / 'state.json').read_text())
    states, metadata = state['jobs'], state['metadata']; handles = {}
    while True:
        reservations = observe_external(jobs, states, metadata, context)
        for job in jobs:
            key = job['id']; record = states[key]
            if record['status'] not in ('running', 'verifying'): continue
            completion = root / 'tasks' / key / 'completion.json'
            # Probe liveness first: a worker can publish completion and exit
            # between an earlier file probe and a later PID probe.
            worker_alive = alive(record.get('pid'), record.get('pid_token'))
            if not worker_alive and completion.exists():
                finished = json.loads(completion.read_text()); states[key] = {**record, **finished}
                if finished['status'] == 'complete': metadata[key] = finished.get('metadata', {})
                write_event(root, {'event': 'finished', 'job': key, 'status': finished['status']})
                if key in handles:
                    handles.pop(key).wait()
            elif not worker_alive:
                states[key] = {**record, 'status': 'failed', 'error': 'Worker disappeared without validated completion', 'finished_at': now()}
            elif (root / 'tasks' / key / 'process.json').exists(): record['status'] = 'verifying'
        for key, causes in dependency_blocks(jobs, states).items(): states[key] = {'status': 'blocked', 'failed_dependencies': causes}
        try:
            inventory = gpu_inventory(); state.pop('gpu_probe_error', None)
        except Exception:
            inventory = {}; state['gpu_probe_error'] = traceback.format_exc()
        active = [j for j in jobs if states[j['id']]['status'] in ('running', 'verifying')]
        occupied = set(reservations)
        for j in active:
            if j['resource'] == 'exclusive_gpu': occupied.update((0, 1))
            elif states[j['id']].get('gpu') is not None: occupied.add(states[j['id']]['gpu'])
        free = [g for g in (0, 1) if g not in occupied and g in inventory and not inventory[g]['processes'] and inventory[g]['free_mb'] >= 40000]
        cpu_available = not any(j['resource'] in ('cpu', 'exclusive_gpu') for j in active)
        state.update(updated_at=now(), scheduler_pid=os.getpid(), reservations=reservations, gpu_inventory=inventory, counts={s: sum(v['status'] == s for v in states.values()) for s in ('pending', 'external', 'running', 'verifying', 'complete', 'failed', 'blocked')})
        atomic_json(root / 'state.json', state)
        if not (root / 'PAUSE').exists():
            for key, gpu in ready_assignments(jobs, states, free, cpu_available):
                output = root / 'tasks' / key
                if output.exists() and any(output.iterdir()):
                    states[key] = {'status': 'failed', 'error': 'Refusing existing task directory; inspect before retry'}; continue
                output.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy(); env.update(PYTHONPATH=str(root / 'source'), PYTHONDONTWRITEBYTECODE='1')
                command = [context['python'], '-m', 'dual2pose.experiments.run_dependency_workflow', 'worker', '--root', str(root), '--job', key]
                if gpu is not None: command += ['--gpu', str(gpu)]
                with (output / 'worker.log').open('x') as log:
                    process = subprocess.Popen(command, cwd=root / 'source', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                handles[key] = process
                states[key] = {'status': 'running', 'gpu': gpu, 'pid': process.pid, 'pid_token': pid_token(process.pid), 'started_at': now()}
                atomic_json(root / 'state.json', state)
                write_event(root, {'event': 'started', 'job': key, 'gpu': gpu, 'pid': process.pid})
        if all(v['status'] in TERMINAL for v in states.values()):
            state['finished_at'] = now(); state['status'] = 'complete' if all(v['status'] == SUCCESS for v in states.values()) else 'finished_with_failures'
            atomic_json(root / 'state.json', state); return
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('build', 'run', 'worker'))
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--repo', type=Path, default=Path.cwd()); parser.add_argument('--main-root', type=Path); parser.add_argument('--core-root', type=Path); parser.add_argument('--raw-cache', type=Path)
    parser.add_argument('--job'); parser.add_argument('--gpu', type=int); parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'build':
        if not all((args.main_root, args.core_root, args.raw_cache)): parser.error('build requires main-root/core-root/raw-cache')
        build({'repo': str(args.repo.resolve()), 'root': str(root), 'python': sys.executable, 'main_root': str(args.main_root.resolve()), 'core_root': str(args.core_root.resolve()), 'raw_cache': str(args.raw_cache.resolve())})
    elif args.action == 'worker': worker(root, args.job, args.gpu)
    else: run(root, args.poll_seconds)


if __name__ == '__main__': main()
