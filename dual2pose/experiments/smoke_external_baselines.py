"""CPU-only external component diagnostics, never publication evaluation.

STRIDE: official DSTformer, random weights and an explicit 13-joint adaptation.
MetaPose: official stage-1 initialization only, optionally in a separate Python.
Neither path implements/claims the complete published method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def _npz_new(path, **values):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **values)


def _source(repo, relative_files):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True,
                                       timeout=30).strip()
    return {'repository': str(repo), 'commit': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain')),
            'file_sha256': {name: _sha256(repo / name) for name in relative_files}}


def stride_backbone_probe(repo, average, output):
    import torch
    import yaml
    from dual2pose.eval.frontend_lifters import _load_python_file, _temporary_import_root

    repo = Path(repo).resolve()
    provenance = _source(repo, ['stride/lib/model/DSTformer.py', 'stride/lib/model/drop.py',
                                'stride/configs/pose3d/MB_ft_h36m.yaml'])
    cfg = yaml.safe_load((repo / 'stride/configs/pose3d/MB_ft_h36m.yaml').read_text())
    if average.ndim != 4 or average.shape[1:] != (30, 13, 3):
        raise ValueError('Expected N,30,13,3 canonical average')
    before = average.copy()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with _temporary_import_root(repo / 'stride', 'lib'):
        module = _load_python_file('_ivc_stride_dstformer', repo / 'stride/lib/model/DSTformer.py')
        model = module.DSTformer(dim_in=3, dim_out=3, dim_feat=cfg['dim_feat'],
                                 dim_rep=cfg['dim_rep'], depth=cfg['depth'],
                                 num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'],
                                 num_joints=13, maxlen=cfg['maxlen'], att_fuse=cfg['att_fuse']).cpu().eval()
        tensor = torch.from_numpy(average.copy())
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(tensor).numpy()
        elapsed = time.perf_counter() - started
    if prediction.shape != average.shape or not np.isfinite(prediction).all():
        raise ValueError('STRIDE backbone returned invalid output')
    np.testing.assert_array_equal(average, before)
    _npz_new(output / 'stride_untrained_common13_diagnostic.npz', prediction=prediction)
    return {'status': 'passed_component_only', 'publication_eligible': False,
            'component': 'official_DSTformer_forward_only', 'source': provenance,
            'weights': 'random_initialization_seed42', 'checkpoint_loaded': False,
            'official_joint_count': cfg['num_joints'], 'diagnostic_joint_count': 13,
            'adaptation_steps': 0, 'input': 'canonical_average_xyz',
            'input_shape': list(average.shape), 'output_shape': list(prediction.shape),
            'finite': True, 'input_unchanged': True, 'parameters': sum(p.numel() for p in model.parameters()),
            'cpu_forward_seconds_diagnostic_only': elapsed, 'torch_version': torch.__version__,
            'remaining': ['Resolve native-to-H36M17 semantics or predeclare retraining',
                          'Obtain and strictly load the specified motion prior',
                          'Implement and validate official per-video adaptation and reset']}


def metapose_stage1_worker(repo, inputs, output):
    """Standalone worker: deliberately has no PyTorch/project-data dependency."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    repo = Path(repo).resolve()
    provenance = _source(repo, ['metapose/inference_time_optimization.py'])
    source = repo / 'metapose/inference_time_optimization.py'
    spec = importlib.util.spec_from_file_location('_ivc_metapose_opt', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(inputs, allow_pickle=False) as archive:
        raw = archive['raw_views'].copy()
    if raw.ndim != 5 or raw.shape[1:] != (30, 2, 13, 3) or not 1 <= raw.shape[0] <= 4:
        raise ValueError('Expected bounded N,30,2,13,3 raw view inputs')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite raw view inputs')
    before = raw.copy()
    poses, rotations, scales, shifts = [], [], [], []
    started = time.perf_counter()
    for views in raw.reshape(-1, 2, 13, 3):
        pose, (rotation, scale, shift) = module.initial_epi_estimate(tf.convert_to_tensor(views))
        poses.append(pose.numpy())
        rotations.append(rotation.numpy())
        scales.append(scale.numpy())
        shifts.append(shift.numpy())
    values = {'pose': np.stack(poses).reshape(*raw.shape[:2], 13, 3),
              'rotation': np.stack(rotations).reshape(*raw.shape[:2], 2, 3, 3),
              'scale': np.stack(scales).reshape(*raw.shape[:2], 2),
              'shift': np.stack(shifts).reshape(*raw.shape[:2], 2, 3)}
    elapsed = time.perf_counter() - started
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Nonfinite MetaPose stage-1 result')
    np.testing.assert_array_equal(raw, before)
    _npz_new(output / 'metapose_stage1_only.npz', **values)
    report = {'status': 'passed_component_only', 'publication_eligible': False,
              'component': 'official_initial_epi_estimate_only', 'source': provenance,
              'inputs_sha256': _sha256(inputs), 'checkpoint_loaded': False,
              'input': 'raw_per_camera_xyz_common13', 'input_shape': list(raw.shape),
              'output_shapes': {key: list(value.shape) for key, value in values.items()},
              'finite': True, 'input_unchanged': True, 'supplied_calibration': False,
              'ground_truth_loaded': False, 'estimated_2d_used_in_stage1': False,
              'learned_refinement_executed': False, 'iterative_refinement_executed': False,
              'cpu_initialization_seconds_diagnostic_only': elapsed,
              'python': sys.executable, 'tensorflow': tf.__version__, 'tfp': tfp.__version__,
              'remaining': ['Resolve pixel/3D coordinate normalization and uncertainty input',
                            'Adapt learned-stage skeleton and obtain/train compatible weights',
                            'Evaluate complete method, not S1 alone']}
    _json_new(output / 'metapose_stage1.json', report)
    return report


def run_probe(cache_dir, dataset_root, output_dir, count=2, stride_repo=None,
              metapose_repo=None, metapose_python=None):
    from dual2pose.eval.external_baseline_inputs import load_estimated_2d, load_pose_probe
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output directory: {output}')
    started = time.perf_counter()
    probe = load_pose_probe(cache_dir, count)
    observed, observations_provenance = load_estimated_2d(probe, dataset_root)
    output.mkdir(parents=True, exist_ok=False)
    _npz_new(output / 'inputs.npz', raw_views=probe.raw_views,
             canonical_views=probe.canonical_views, average=probe.average,
             frame_ids=probe.frame_ids, estimated_2d_px=observed)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'inputs_validated', 'publication_eligible': False,
              'purpose': 'component_smoke_only_no_accuracy_evaluation',
              'command': [sys.executable, *sys.argv], 'python_version': platform.python_version(),
              'numpy_version': np.__version__, 'device': 'cpu', 'seed': 42,
              'runner_sha256': _sha256(__file__), 'inputs': probe.provenance,
              'observations': observations_provenance,
              'components': {name: {'status': 'not_requested'} for name in ('stride', 'metapose')}}
    if stride_repo:
        try:
            report['components']['stride'] = stride_backbone_probe(stride_repo, probe.average, output)
        except Exception as exc:
            report['components']['stride'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    if metapose_repo:
        command = [str(metapose_python or sys.executable), str(Path(__file__).resolve()),
                   '--metapose-worker', '--metapose-repo', str(Path(metapose_repo).resolve()),
                   '--worker-inputs', str(output / 'inputs.npz'), '--output-dir', str(output)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'TF_CPP_MIN_LOG_LEVEL': '2',
               'OMP_NUM_THREADS': '2', 'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2'}
        try:
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
            (output / 'metapose_worker.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                report['components']['metapose'] = {'status': 'failed', 'returncode': result.returncode,
                                                   'error': result.stderr[-4000:], 'command': command}
            else:
                report['components']['metapose'] = json.loads((output / 'metapose_stage1.json').read_text())
                report['components']['metapose']['command'] = command
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['components']['metapose'] = {'status': 'failed', 'error': str(exc), 'command': command}
    report['elapsed_seconds'] = time.perf_counter() - started
    report['artifact_sha256'] = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    if any(c['status'] == 'failed' for c in report['components'].values()):
        report['status'] = 'component_failure'
    _json_new(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--stride-repo', type=Path)
    parser.add_argument('--metapose-repo', type=Path)
    parser.add_argument('--metapose-python', type=Path)
    parser.add_argument('--metapose-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-inputs', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.metapose_worker:
        if not args.metapose_repo or not args.worker_inputs:
            parser.error('worker requires source and input paths')
        metapose_stage1_worker(args.metapose_repo, args.worker_inputs, args.output_dir)
        return
    if not args.cache_dir or not args.dataset_root:
        parser.error('--cache-dir and --dataset-root are required')
    report = run_probe(args.cache_dir, args.dataset_root, args.output_dir, args.count,
                       args.stride_repo, args.metapose_repo, args.metapose_python)
    print(json.dumps({'report': str(args.output_dir / 'report.json'), 'status': report['status'],
                      'publication_eligible': False,
                      'components': {name: row['status'] for name, row in report['components'].items()}}))
    if report['status'] == 'component_failure':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
