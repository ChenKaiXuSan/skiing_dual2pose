"""Target-free, bounded input preparation for external-baseline diagnostics.

This module is not a full evaluation loader. It refuses test splits and exports
at most four windows from the first archived protocol batch. In particular it
never opens target.npy, GT 2D, labels HDF5 or camera calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from dual2pose.eval.main_baseline_data import sha256
from dual2pose.eval.main_baseline_geometry import (
    _load_index, _match_source_samples, _resolve_dataset_path,
)
from dual2pose.map_config import filter_sam3d_body_kpts
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


@dataclass(frozen=True)
class PoseProbe:
    raw_views: np.ndarray  # N,T,view,common13,xyz; original camera coordinates
    canonical_views: np.ndarray
    average: np.ndarray
    frame_ids: np.ndarray
    samples: list[dict]
    provenance: dict


def _verify(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f'Input checksum mismatch: {path}')


def load_pose_probe(cache_dir: Path, count: int = 2) -> PoseProbe:
    """Read only inputs from a verified train/val cache; preserve its anchor."""
    root = Path(cache_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    dataset, split = manifest['dataset'], manifest['split']
    if dataset not in ('unity', 'ski') or split not in ('train', 'val'):
        raise ValueError('Smoke accepts unity/ski train or val only, never test')
    if not 1 <= count <= 4:
        raise ValueError('Smoke count must be between 1 and 4')
    if manifest['representation'] != 'raw_pre_batch_canonicalization':
        raise ValueError('Expected raw pre-batch-canonicalization cache')
    total, joints = manifest['sample_count'], (15 if dataset == 'unity' else 13)
    if total < count or manifest['num_joints'] != joints or manifest['time_window'] != 30:
        raise ValueError('Cache sample count, joint count or time window mismatch')
    names = ('left.npy', 'right.npy', 'frames.npy', 'samples.jsonl')
    for name in names:
        _verify(root / name, manifest['files'][name])
    batch_size = 256 if dataset == 'unity' else 4
    stop = min(total, batch_size)
    indices = list(range(2, 15)) if dataset == 'unity' else list(range(13))
    kwargs = {} if dataset == 'unity' else dict(left_hip=4, right_hip=5, neck=12)
    raw, canonical = [], []
    for name in ('left', 'right'):
        mapped = np.load(root / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        if mapped.shape != (total, 30, joints, 3) or mapped.dtype != np.float32:
            raise ValueError(f'Invalid {name} shape/dtype: {mapped.shape}/{mapped.dtype}')
        batch = np.array(mapped[:stop], copy=True)
        if not np.isfinite(batch).all():
            raise ValueError(f'Nonfinite {name} input')
        transformed = canonicalize_pose_torch(torch.from_numpy(batch), **kwargs)[0].numpy()
        # Select after the native 15/13-joint transform, not before it.
        raw.append(batch[:count, :, indices].copy())
        canonical.append(transformed[:count, :, indices].copy())
    frame_map = np.load(root / 'frames.npy', mmap_mode='r', allow_pickle=False)
    if frame_map.shape != (total, 30) or not np.issubdtype(frame_map.dtype, np.integer):
        raise ValueError('Invalid cached frame shape/dtype')
    frames = np.array(frame_map[:count], copy=True)
    if np.any(frames < 0) or np.any(np.diff(frames, axis=1) < 0):
        raise ValueError('Frame IDs must be nonnegative and nondecreasing')
    with (root / 'samples.jsonl').open() as handle:
        samples = [json.loads(next(handle)) for _ in range(count)]
    if [sample['index'] for sample in samples] != list(range(count)):
        raise ValueError('Cached sample index/order mismatch')
    views = np.stack(canonical, axis=2)
    if not np.isfinite(views).all():
        raise ValueError('Nonfinite canonical input')
    provenance = {
        'dataset': dataset, 'split': split, 'cache_dir': str(root),
        'manifest_sha256': sha256(root / 'manifest.json'),
        'verified_input_files': {name: manifest['files'][name] for name in names},
        'source_index': manifest['index'], 'source_index_sha256': manifest['index_sha256'],
        'cache_sample_count': total, 'selected_count': count, 'samples': samples,
        'protocol_batch_size': batch_size, 'canonicalized_batch_count': stop,
        'anchor_sample_index': 0, 'joint_subset': 'common13',
        'native_joint_count': joints, 'native_to_common_indices': indices,
        'ground_truth_loaded': False, 'supplied_calibration': False,
        'input_code_sha256': sha256(__file__),
        'canonicalizer_sha256': sha256(Path(__file__).parents[1] / 'trainer/canonicalize.py'),
    }
    return PoseProbe(np.stack(raw, axis=2), views, views.mean(axis=2), frames,
                     samples, provenance)


def load_estimated_2d(probe: PoseProbe, dataset_root: Path) -> tuple[np.ndarray, dict]:
    """Pair native estimated 2D by source identity/frame, without calibration."""
    dataset = probe.provenance['dataset']
    index = Path(probe.provenance['source_index'])
    _verify(index, probe.provenance['source_index_sha256'])
    sources = _match_source_samples(dataset, probe.samples,
                                   _load_index(index, probe.provenance['split']))
    file_hashes, sequences = {}, []
    for source, frames in zip(sources, probe.frame_ids):
        keys = ('sam3d_cam1_kpt2d_dir', 'sam3d_cam2_kpt2d_dir') if dataset == 'unity' else (
            'cam1_sam3d_kpt2d_dir', 'cam2_sam3d_kpt2d_dir')
        views = []
        for key in keys:
            directory = _resolve_dataset_path(str(source[key]), Path(dataset_root))
            # Parse names rather than assume a particular zero-padding width.
            paths = list(directory.glob('kpt2d_*.npy' if dataset == 'unity' else '*_sam3d_body.npz'))
            by_frame = {}
            for path in paths:
                frame = int(path.stem.rsplit('_', 1)[-1] if dataset == 'unity'
                            else path.name.split('_', 1)[0])
                if frame in by_frame:
                    raise ValueError(f'Duplicate estimated-2D frame {frame}: {directory}')
                by_frame[frame] = path
            points = []
            for frame in frames:
                if int(frame) not in by_frame:
                    raise FileNotFoundError(f'Missing estimated 2D frame {frame}: {directory}')
                path = by_frame[int(frame)]
                if dataset == 'unity':
                    native = np.load(path, allow_pickle=False)
                else:
                    # Existing, trusted local SAM3D archive format uses an object dictionary.
                    with np.load(path, allow_pickle=True) as archive:
                        native = archive['output'].item()['pred_keypoints_2d']
                common = filter_sam3d_body_kpts(native)[2:15]
                if common.shape != (13, 2) or not np.isfinite(common).all():
                    raise ValueError(f'Invalid estimated 2D shape/values: {path}')
                points.append(common.astype(np.float32))
                file_hashes[str(path)] = sha256(path)
            views.append(np.stack(points))
        sequences.append(np.stack(views, axis=1))
    return np.stack(sequences), {
        'coordinate_space': 'image_pixels', 'observation_type': 'estimated_point_positions',
        'uncertainty_available': False, 'supplied_calibration': False,
        'ground_truth_loaded': False, 'verified_source_index_sha256': sha256(index),
        'observation_file_sha256': file_hashes,
    }
