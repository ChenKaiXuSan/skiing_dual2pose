"""Re-evaluate selected IVC main-table models with PA-MPJPE; never retrain."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from dual2pose.eval.main_baseline_data import ROOT, batch_slices, canonical_batch, load_cache, sha256, write_json
from dual2pose.eval.pa_mpjpe import PA_PROTOCOL, PAMetricAccumulator
from dual2pose.eval.render_main_baseline_tables import DIRECT, GEOMETRY
from dual2pose.experiments.run_main_baselines import FunctionModel, ReferenceModel
from dual2pose.experiments.run_geometry_baselines import _load_verified_split, _prediction_reference, _tensor
from dual2pose.models.main_baselines import aligned_average, quality_weighted_fusion, build_baseline
from dual2pose.models.dlt_residual_baseline import DLTResidualMLP


def selected_model(method, record, joints, device):
    checkpoint = record.get('checkpoint')
    if checkpoint and sha256(checkpoint) != record['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint checksum mismatch: {checkpoint}')
    functions = {'left_canonical': lambda l, r: l, 'right_canonical': lambda l, r: r,
                 'canonical_avg': lambda l, r: (l+r)/2, 'aligned_average': aligned_average,
                 'quality_weighted': quality_weighted_fusion}
    if method in functions:
        model = FunctionModel(functions[method])
    elif method == 'canonfuse3d':
        model = ReferenceModel(checkpoint)
    elif method in {'dlt', 'robust_dlt'}:
        return None
    else:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['epoch'] != record['selected_epoch']:
            raise RuntimeError(f'Selected epoch mismatch: {method}')
        model = DLTResidualMLP(joints) if method == 'dlt_mlp' else build_baseline(method, num_joints=joints, time_window=30)
        model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval()


def check_previous_metrics(previous, current):
    differences = {}
    for subset, old in previous.items():
        new = current[subset]
        for key in ['sample_count', 'point_count', 'acceleration_point_count']:
            if old[key] != new[key]:
                raise RuntimeError(f'Evaluation coverage changed: {subset}/{key}')
        differences[subset] = {}
        for key in ['mpjpe', 'acceleration_error']:
            delta = abs(old[key] - new[key])
            differences[subset][key] = delta
            if not np.isfinite(delta) or delta > 5e-7:
                raise RuntimeError(f'Existing metric reproduction failed: {subset}/{key}: {delta}')
    return differences


@torch.no_grad()
def run(args):
    args.root = args.root.resolve(); args.output = args.output.resolve()
    paths = [args.output/'results'/args.dataset/f'{name}.json' for name in DIRECT+GEOMETRY]
    for path in paths:
        if path.exists():
            raise FileExistsError(f'Preserving existing PA result: {path}')
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    arrays, cache_manifest = load_cache(args.root/'cache'/args.dataset/'test', args.device)
    inputs = _load_verified_split(args.root, args.dataset, 'test')
    subsets = ['all15', 'common13'] if args.dataset == 'unity' else ['common13']
    count, _, joints, _ = arrays[0].shape
    batch_size = 256 if args.dataset == 'unity' else 4
    source_files = [Path(__file__), ROOT/'dual2pose/eval/pa_mpjpe.py',
                    ROOT/'dual2pose/eval/main_baseline_data.py',
                    ROOT/'dual2pose/experiments/run_main_baselines.py',
                    ROOT/'dual2pose/experiments/run_geometry_baselines.py',
                    ROOT/'dual2pose/models/main_baselines.py',
                    ROOT/'dual2pose/models/dlt_residual_baseline.py',
                    ROOT/'dual2pose/models/crossview_fusion.py']
    source_hashes = {str(p): sha256(p) for p in source_files}
    started = time.monotonic()
    for method, output_path in zip(DIRECT+GEOMETRY, paths):
        original_path = args.root/'results'/args.dataset/f'{method}.json'
        original = json.loads(original_path.read_text())
        if args.dataset == 'ski' and method in GEOMETRY and not original['provenance'].get('ski_prediction_root_centered'):
            raise RuntimeError('Ski geometric reference must be pelvis-relative')
        model = selected_model(method, original, joints, args.device)
        accumulators = {subset: PAMetricAccumulator() for subset in subsets}
        method_start = time.monotonic()
        for batch, (start, stop) in enumerate(batch_slices(count, batch_size)):
            if method in DIRECT:
                left, right, target = canonical_batch(*(a[start:stop] for a in arrays), dataset=args.dataset)
                prediction = model(left, right)
            else:
                raw = inputs.robust_dlt if method == 'robust_dlt' else inputs.dlt
                raw_prediction = _prediction_reference(_tensor(raw[start:stop], args.device), args.dataset)
                prediction, _, target = canonical_batch(raw_prediction, raw_prediction,
                    _tensor(inputs.target[start:stop], args.device), dataset=args.dataset)
                if method == 'dlt_mlp':
                    prediction = model(prediction, _tensor(inputs.reprojection_error_px[start:stop], args.device),
                                       _tensor(inputs.dlt_valid[start:stop], args.device))
            for subset, metric in accumulators.items():
                idx = slice(2, None) if subset == 'common13' and args.dataset == 'unity' else slice(None)
                metric.update(prediction[:, :, idx], target[:, :, idx])
            if batch % 32 == 0 or stop == count:
                progress = {'pid': os.getpid(), 'dataset': args.dataset, 'method': method,
                            'processed_samples': stop, 'total_samples': count,
                            'method_seconds': time.monotonic()-method_start, 'status': 'evaluating'}
                write_json(args.output/f'progress_{args.dataset}.json', progress)
                print(json.dumps(progress), flush=True)
        metrics = {name: accumulator.result() for name, accumulator in accumulators.items()}
        differences = check_previous_metrics(original['metrics'], metrics)
        result = copy.deepcopy(original)
        result['metrics'] = metrics
        result['pa_evaluation'] = {'protocol': PA_PROTOCOL, 'original_result': str(original_path),
            'original_result_sha256': sha256(original_path), 'original_metric_absolute_differences': differences,
            'cache_manifest_sha256': sha256(args.root/'cache'/args.dataset/'test/manifest.json'),
            'geometry_manifest_sha256': sha256(args.root/'geometry'/args.dataset/'test/manifest.json'),
            'source_sha256': source_hashes, 'command': sys.argv, 'device': args.device,
            'torch_version': torch.__version__, 'numpy_version': np.__version__,
            'precision': 'FP32 inference; TF32 disabled; float64 Procrustes',
            'elapsed_seconds': time.monotonic()-method_start, 'retrained': False}
        write_json(output_path, result)
        print(json.dumps({'dataset': args.dataset, 'method': method, 'metrics': metrics,
                          'original_metric_absolute_differences': differences}), flush=True)
        del model
    write_json(args.output/f'progress_{args.dataset}.json', {'pid': os.getpid(), 'status': 'complete',
        'dataset': args.dataset, 'method_count': len(paths), 'elapsed_seconds': time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['unity', 'ski'], required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == '__main__': main()
