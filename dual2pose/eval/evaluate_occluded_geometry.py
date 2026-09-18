"""Matched C07 RGB occlusion/calibration evaluation; no test-time tuning.

All five methods retain the original ordered cohort. Missing DLT points use
an explicit zero-vector fallback before the unchanged canonicalization; the
residual MLP receives the missingness mask. This is a disclosed deployment
fallback, not a claim that failed triangulations were recovered.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np

from dual2pose.eval.main_baseline_geometry import triangulate_sequence, gate_dlt_result, unity_projection_matrix
from dual2pose.eval.run_matched_occlusion_frontend import (
    CONDITIONS, DEFAULT_DATA_ROOT, REPO_ROOT, validate_matched_stream, stream_output_path,
)
from dual2pose.eval.main_baseline_data import sha256, write_json

DEFAULT_BASELINE = REPO_ROOT/'logs/ivc_mmsports_extension/main_baselines/20260906'
ANGLES = (0., 1., 3.)
METHODS = ('canonfuse3d', 'canonical_avg', 'dlt', 'robust_dlt', 'dlt_mlp')
EVALUATION_CONDITIONS = [('clean', 'clean', 'clean')] + [
    (f'{condition}_{side}', condition if side in ('left', 'both') else 'clean',
     condition if side in ('right', 'both') else 'clean')
    for condition in ('random_0p5', 'random_1') for side in ('left', 'right', 'both')]


def select_frames(data, requested):
    ids = np.asarray(data['frame_indices'])
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate cache frame IDs')
    lookup = {int(frame): index for index, frame in enumerate(ids)}
    try:
        positions = [lookup[int(frame)] for frame in requested]
    except KeyError as error:
        raise ValueError(f'Requested frame missing: {error}') from error
    return {key: np.asarray(data[key])[positions] for key in
            ('pose', 'keypoints_2d', 'valid_2d', 'detection_failed')}


def rotate_projection(projection, intrinsics, degrees):
    """Positive right-camera local-y extrinsic rotation, fixed camera center."""
    p, k = np.asarray(projection, np.float64), np.asarray(intrinsics, np.float64)
    if not np.isfinite(degrees) or p.shape != (3, 4) or k.shape != (3, 3):
        raise ValueError('Invalid camera rotation inputs')
    if degrees == 0:
        return p.copy()
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    q = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return k @ q @ np.linalg.solve(k, p)


def geometry_predictions(p_left, p_right, uv_left, uv_right):
    """No GT argument: triangulation and the archived fixed 25-pixel gate."""
    dlt = triangulate_sequence(p_left, p_right, uv_left, uv_right)
    robust = gate_dlt_result(dlt, threshold_px=25.)
    finite_error = np.isfinite(dlt.reprojection_error_px)
    return {'dlt': np.where(dlt.valid_mask[..., None], dlt.prediction, 0.).astype(np.float32),
        'robust_dlt': np.where(robust.valid_mask[..., None], robust.prediction, 0.).astype(np.float32),
        'dlt_valid': dlt.valid_mask, 'robust_valid': robust.valid_mask,
        'reprojection_error_px': np.where(finite_error, dlt.reprojection_error_px, 0.).astype(np.float32),
        'counts': {'point_count_all15': int(dlt.valid_mask.size),
            'dlt_invalid_points': int((~dlt.valid_mask).sum()),
            'robust_invalid_points': int((~robust.valid_mask).sum()),
            'dlt_invalid_points_common13': int((~dlt.valid_mask[..., 2:]).sum()),
            'robust_invalid_points_common13': int((~robust.valid_mask[..., 2:]).sum()),
            'gate_accepted_points': int(robust.gate_accepted.sum()),
            'gate_interpolated_points': int(robust.interpolated_mask.sum()),
            'gate_ungated_fallback_points': int(robust.fallback_mask.sum()),
            'nonfinite_reprojection_points': int((~finite_error).sum())}}


def load_inputs(frontend, baseline):
    contract = json.loads((frontend/'contract.json').read_text())
    required = json.loads((frontend/'required_frames_manifest.json').read_text())
    import hashlib
    if hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest() != contract['required_sha256']:
        raise RuntimeError('Required frame manifest checksum changed')
    cache = baseline/'cache/unity'/contract['split']
    cached = json.loads((cache/'manifest.json').read_text())
    if cached['index_sha256'] != contract['index_sha256']:
        raise RuntimeError('Cache and frontend index hashes differ')
    for name, expected in cached['files'].items():
        if sha256(cache/name) != expected:
            raise RuntimeError(f'Original cache checksum mismatch: {name}')
    samples = [json.loads(line) for line in (cache/'samples.jsonl').read_text().splitlines()]
    frames = np.load(cache/'frames.npy', mmap_mode='r')
    pairs = required['pair_sequences']
    if contract['max_pairs'] is None and len(pairs) != cached['sample_count']:
        raise RuntimeError('Full cohort size mismatch')
    if contract['max_pairs'] is not None and (contract['split'] != 'val' or len(pairs) != contract['max_pairs']):
        raise RuntimeError('Invalid partial cohort')
    for index, (pair, sample) in enumerate(zip(pairs, samples)):
        for key in ('person_id', 'action_id', 'cam1_id', 'cam2_id'):
            if pair[key] != sample[key]:
                raise RuntimeError(f'Pair ordering mismatch at {index}: {key}')
        if not np.array_equal(pair['frame_indices'], frames[index]):
            raise RuntimeError(f'Frame ordering mismatch at {index}')
    stores = {}
    manifests = {}
    for name, setting in CONDITIONS.items():
        manifest = json.loads((frontend/name/'frontend_manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['run_signature'] != contract['run_signature']:
            raise RuntimeError(f'Incomplete/incompatible frontend: {name}')
        expected_hashes = {entry['path']: entry['sha256'] for entry in manifest['entries']}
        if len(expected_hashes) != len(required['streams']):
            raise RuntimeError('Incomplete stream manifest')
        store = {}
        for stream in required['streams']:
            path = stream_output_path(frontend/name, stream).resolve()
            if sha256(path) != expected_hashes.get(str(path)) or not validate_matched_stream(
                    path, stream['frame_indices'], setting, contract['run_signature']):
                raise RuntimeError(f'Invalid paired input: {path}')
            with np.load(path, allow_pickle=False) as data:
                store[(stream['person_id'], stream['action_id'], stream['camera_id'])] = dict(data)
        stores[name] = store
        manifests[name] = manifest
    target = np.load(cache/'target.npy', mmap_mode='r')
    if target.shape != (cached['sample_count'], 30, 15, 3):
        raise RuntimeError('Invalid target cache shape')
    return contract, pairs, stores, target, cached, manifests


def camera_matrices(data_root, pairs):
    projections, intrinsics, hashes = {}, {}, {}
    for pair in pairs:
        for camera in (pair['cam1_id'], pair['cam2_id']):
            key = (pair['person_id'], camera)
            if key in projections:
                continue
            folder = data_root/'data_pole_ski'/key[0]/'cameras'/camera.removeprefix('capture_')
            intrinsic = json.loads((folder/'intrinsics.json').read_text(encoding='utf-8-sig'))
            intrinsics[key] = np.array([[intrinsic['fx'], 0, intrinsic['cx']],
                [0, intrinsic['fy'], intrinsic['cy']], [0, 0, 1]], np.float64)
            projections[key] = unity_projection_matrix(data_root, *key)
            for name in ('intrinsics.json', 'extrinsics.json'):
                hashes[str(folder/name)] = sha256(folder/name)
    return projections, intrinsics, hashes


def validate_condition_result(saved, protocol_hash, condition, sample_count):
    try:
        if saved['status'] != 'complete' or saved['protocol_sha256'] != protocol_hash or saved['condition'] != condition:
            return False
        expected = {(angle, method) for angle in ANGLES for method in METHODS}
        actual = [(cell['angle_degrees'], cell['method']) for cell in saved['results']]
        if len(actual) != len(expected) or set(actual) != expected:
            return False
        for cell in saved['results']:
            metrics = cell['metrics']
            if not all(np.isfinite(metrics[key]) and metrics[key] >= 0 for key in ('mpjpe', 'pa_mpjpe', 'acceleration_error')):
                return False
            if any(metrics[key] != value for key, value in {
                'sample_count': sample_count, 'point_count': sample_count*30*13,
                'pa_point_count': sample_count*30*13, 'acceleration_point_count': sample_count*28*13}.items()):
                return False
            if cell['reused_no_camera_result'] != (cell['angle_degrees'] != 0 and cell['method'] in ('canonfuse3d', 'canonical_avg')):
                return False
        if saved['detection_coverage']['sample_count'] != sample_count or saved['detection_coverage']['frame_positions'] != sample_count*30:
            return False
        return all(saved['geometry_coverage'][str(a)]['point_count_all15'] == sample_count*30*15 for a in ANGLES)
    except (KeyError, TypeError, ValueError):
        return False


def run(args):
    import torch
    from dual2pose.eval.main_baseline_data import batch_slices, canonical_batch
    from dual2pose.eval.pa_mpjpe import PAMetricAccumulator, PA_PROTOCOL
    from dual2pose.eval.evaluate_main_baseline_pa import selected_model
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract, pairs, stores, target, cached, frontend_manifests = load_inputs(args.frontend, args.baseline)
    projections, intrinsics, camera_hashes = camera_matrices(args.data_root, pairs)
    records = {name: json.loads((args.baseline/f'results/unity/{name}.json').read_text())
               for name in ('canonfuse3d', 'dlt_mlp')}
    provenance = {'frontend_contract': contract, 'original_cache': cached,
        'frontend_manifest_sha256': {name: sha256(args.frontend/name/'frontend_manifest.json') for name in CONDITIONS},
        'frozen_models': {name: {key: record[key] for key in ('checkpoint', 'checkpoint_sha256')}
                          for name, record in records.items()},
        'camera_file_sha256': camera_hashes, 'batch_size': 256, 'joint_subset': 'common13 after all15 canonicalization',
        'angles_degrees': ANGLES, 'rotation': 'right camera local positive y; center fixed; left unchanged',
        'input_2d': 'SAM3D projected original-image pixels from same selected person and masked RGB as 3D',
        'missing_policy': 'zero invalid DLT vectors before unchanged batch-anchor canonicalization; failed anchors can affect the whole batch; no dropped samples; MLP receives validity mask',
        'robust_policy': 'archived 25px gate; interpolate finite rejected points, ungated fallback for all-rejected finite tracks; missing observations not interpolated',
        'pa_protocol': PA_PROTOCOL, 'source_sha256': {str(p.relative_to(REPO_ROOT)): sha256(p) for p in (
            Path(__file__), REPO_ROOT/'dual2pose/eval/main_baseline_geometry.py',
            REPO_ROOT/'dual2pose/eval/main_baseline_data.py', REPO_ROOT/'dual2pose/eval/pa_mpjpe.py',
            REPO_ROOT/'dual2pose/trainer/canonicalize.py', REPO_ROOT/'dual2pose/models/crossview_fusion.py',
            REPO_ROOT/'dual2pose/models/dlt_residual_baseline.py')},
        'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': args.device,
        'precision': 'FP32 neural/canonical; float64 DLT and Procrustes; TF32 disabled', 'retrained': False}
    provenance = json.loads(json.dumps(provenance))
    args.output.mkdir(parents=True, exist_ok=True)
    pp = args.output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != provenance:
        raise FileExistsError('Evaluation contract changed; preserve existing results')
    write_json(pp, provenance)
    models = {name: selected_model(name, record, 15, args.device) for name, record in records.items()}
    tensor = lambda value: torch.as_tensor(np.array(value, copy=True), device=args.device)
    started = time.monotonic()
    with torch.inference_mode():
        for condition, left_name, right_name in EVALUATION_CONDITIONS:
            output = args.output/f'{condition}.json'
            if output.exists():
                saved = json.loads(output.read_text())
                if not validate_condition_result(saved, sha256(pp), condition, len(pairs)):
                    raise FileExistsError(f'Invalid existing result: {output}')
                continue
            metrics = {(angle, method): PAMetricAccumulator() for angle in ANGLES for method in METHODS
                       if angle == 0 or method not in ('canonfuse3d', 'canonical_avg')}
            counts = {angle: Counter() for angle in ANGLES}
            detections = Counter()
            for batch_index, (start, stop) in enumerate(batch_slices(len(pairs), 256)):
                rows = pairs[start:stop]
                left, right = [], []
                for pair in rows:
                    key = (pair['person_id'], pair['action_id'])
                    left.append(select_frames(stores[left_name][(*key, pair['cam1_id'])], pair['frame_indices']))
                    right.append(select_frames(stores[right_name][(*key, pair['cam2_id'])], pair['frame_indices']))
                stack = lambda items, key: np.stack([item[key] for item in items])
                lf, rf = stack(left, 'detection_failed'), stack(right, 'detection_failed')
                detections.update(left_failed_frame_positions=int(lf.sum()), right_failed_frame_positions=int(rf.sum()),
                    either_failed_frame_positions=int((lf | rf).sum()), both_failed_frame_positions=int((lf & rf).sum()),
                    frame_positions=int(lf.size), sample_count=len(rows),
                    left_failed_batch_anchor=int(lf[0, 0]), right_failed_batch_anchor=int(rf[0, 0]),
                    samples_in_left_failed_anchor_batches=len(rows)*int(lf[0, 0]),
                    samples_in_right_failed_anchor_batches=len(rows)*int(rf[0, 0]))
                l, r, truth = canonical_batch(tensor(stack(left, 'pose')), tensor(stack(right, 'pose')),
                    tensor(target[start:stop]), dataset='unity')
                for method, prediction in [('canonical_avg', (l+r)/2), ('canonfuse3d', models['canonfuse3d'](l, r))]:
                    metrics[(0., method)].update(prediction[:, :, 2:], truth[:, :, 2:])
                pl = np.stack([projections[(row['person_id'], row['cam1_id'])] for row in rows])[:, None]
                uv_l, uv_r = stack(left, 'keypoints_2d'), stack(right, 'keypoints_2d')
                for angle in ANGLES:
                    pr = np.stack([rotate_projection(projections[(row['person_id'], row['cam2_id'])],
                        intrinsics[(row['person_id'], row['cam2_id'])], angle) for row in rows])[:, None]
                    geometry = geometry_predictions(pl, pr, uv_l, uv_r)
                    counts[angle].update(geometry['counts'])
                    for geometry_method, valid_key in [('dlt', 'dlt_valid'), ('robust_dlt', 'robust_valid')]:
                        missing_anchor = not geometry[valid_key][0, 0, [0, 1, 6, 7, 14]].all()
                        counts[angle].update({f'{geometry_method}_missing_batch_anchor': int(missing_anchor),
                            f'{geometry_method}_samples_in_missing_anchor_batches': len(rows)*int(missing_anchor)})
                    dlt, robust, _ = canonical_batch(tensor(geometry['dlt']), tensor(geometry['robust_dlt']),
                        tensor(target[start:stop]), dataset='unity')
                    residual = models['dlt_mlp'](dlt, tensor(geometry['reprojection_error_px']), tensor(geometry['dlt_valid']))
                    for method, prediction in [('dlt', dlt), ('robust_dlt', robust), ('dlt_mlp', residual)]:
                        metrics[(angle, method)].update(prediction[:, :, 2:], truth[:, :, 2:])
                if batch_index % 16 == 0 or stop == len(pairs):
                    progress = {'status': 'running', 'condition': condition, 'samples': stop,
                        'total_samples': len(pairs), 'elapsed_seconds': time.monotonic()-started, 'pid': os.getpid()}
                    write_json(args.output/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            results = []
            for angle in ANGLES:
                for method in METHODS:
                    source_angle = 0. if method in ('canonfuse3d', 'canonical_avg') else angle
                    result = metrics[(source_angle, method)].result()
                    if result['sample_count'] != len(pairs) or result['point_count'] != len(pairs)*30*13:
                        raise RuntimeError('Metric coverage mismatch')
                    results.append({'angle_degrees': angle, 'method': method, 'metrics': result,
                                    'reused_no_camera_result': angle != 0 and source_angle == 0})
            write_json(output, {'status': 'complete', 'condition': condition, 'protocol_sha256': sha256(pp),
                'results': results, 'detection_coverage': dict(detections),
                'geometry_coverage': {str(angle): dict(value) for angle, value in counts.items()},
                'unique_source_detection_failures': {name: frontend_manifests[name]['detection_failed']
                    for name in {left_name, right_name}}})
    for condition, _, _ in EVALUATION_CONDITIONS:
        saved = json.loads((args.output/f'{condition}.json').read_text())
        if not validate_condition_result(saved, sha256(pp), condition, len(pairs)):
            raise RuntimeError(f'Final artifact validation failed: {condition}')
    write_json(args.output/'progress.json', {'status': 'complete', 'conditions': len(EVALUATION_CONDITIONS),
        'unique_metric_cells': 77, 'display_cells_including_camera_invariant_reuse': 105,
        'sample_count_per_cell': len(pairs), 'split': contract['split'],
        'elapsed_seconds': time.monotonic()-started, 'pid': os.getpid()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
