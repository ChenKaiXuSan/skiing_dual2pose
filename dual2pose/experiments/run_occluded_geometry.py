"""Run approved full C07 stages after a verified validation smoke; no retries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dual2pose.eval.evaluate_occluded_geometry import EVALUATION_CONDITIONS, validate_condition_result
from dual2pose.eval.main_baseline_data import sha256, write_json
from dual2pose.eval.run_matched_occlusion_frontend import REPO_ROOT


def verify_smoke(root):
    protocol_path = root/'evaluation/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    progress = json.loads((root/'evaluation/progress.json').read_text())
    if progress['status'] != 'complete' or progress['split'] != 'val':
        raise RuntimeError('A completed validation-only smoke is required')
    frontend = protocol['frontend_contract']
    if frontend['split'] != 'val' or not frontend['max_pairs']:
        raise RuntimeError('Invalid validation smoke contract')
    for hashes in (protocol['source_sha256'], frontend['source_sha256']):
        for name, expected in hashes.items():
            if sha256(REPO_ROOT/name) != expected:
                raise RuntimeError(f'Code changed since the smoke: {name}')
    for name, expected in frontend['auxiliary_files_sha256'].items():
        if sha256(name) != expected:
            raise RuntimeError(f'Auxiliary frontend asset changed: {name}')
    for condition, _, _ in EVALUATION_CONDITIONS:
        saved = json.loads((root/f'evaluation/{condition}.json').read_text())
        if not validate_condition_result(saved, sha256(protocol_path), condition, frontend['max_pairs']):
            raise RuntimeError(f'Invalid smoke output: {condition}')
    return {'root': str(root.resolve()), 'protocol_sha256': sha256(protocol_path),
            'sample_count': frontend['max_pairs'], 'all_seven_conditions_valid': True}


def run(args):
    smoke = verify_smoke(args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUDA_VISIBLE_DEVICES=str(args.gpu))
    stages = [
        ('frontend', [sys.executable, '-u', '-m', 'dual2pose.eval.run_matched_occlusion_frontend',
            '--split', 'test', '--output', str(args.output/'frontend'), '--gpu', str(args.gpu)]),
        ('evaluation', [sys.executable, '-u', '-m', 'dual2pose.eval.evaluate_occluded_geometry',
            '--frontend', str(args.output/'frontend'), '--output', str(args.output/'evaluation'), '--device', 'cuda:0']),
    ]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record = {'status': 'running', 'pid': os.getpid(), 'started_utc': stamp, 'gpu': args.gpu,
              'timeout_seconds_per_stage': args.timeout_hours*3600, 'smoke': smoke, 'stages': []}
    started = time.monotonic()
    try:
        for name, command in stages:
            log_path = args.output/f'{name}_{stamp}.log'
            stage = {'name': name, 'command': command, 'log': str(log_path), 'status': 'running'}
            record['stages'].append(stage)
            write_json(args.output/'runner_status.json', record)
            print(json.dumps(stage), flush=True)
            with log_path.open('x') as log:
                process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=args.timeout_hours*3600, check=False)
            stage['returncode'] = process.returncode
            if process.returncode:
                stage['status'] = 'failed'
                raise RuntimeError(f'{name} exited {process.returncode}; no automatic retry; see {log_path}')
            progress = json.loads((args.output/name/'progress.json').read_text())
            if progress.get('status') != 'complete':
                raise RuntimeError(f'{name} exit did not produce a completion artifact')
            stage['status'] = 'complete'
            stage['progress'] = progress
            write_json(args.output/'runner_status.json', record)
        record['status'] = 'complete'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        write_json(args.output/'runner_status.json', record)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timeout-hours', type=float, default=48.)
    args = parser.parse_args()
    if args.timeout_hours <= 0:
        parser.error('--timeout-hours must be positive')
    run(args)


if __name__ == '__main__':
    main()
