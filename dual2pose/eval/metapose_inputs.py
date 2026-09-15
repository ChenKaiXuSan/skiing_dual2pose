"""Prediction-only MetaPose input adaptation; no labels or camera calibration."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite {name}')
    return value


def normalize_observations(xyz, uv):
    xyz, uv = np.asarray(xyz, dtype=np.float64), np.asarray(uv, dtype=np.float64)
    if xyz.shape[-3:] != (2, 13, 3) or uv.shape != (*xyz.shape[:-1], 2):
        raise ValueError('Expected matching (...,2,13,3) and (...,2,13,2)')
    finite(xyz, 'native 3D'); finite(uv, 'estimated 2D')
    origin = uv.min(axis=-2)
    side = np.ptp(uv, axis=-2).max(axis=-1)
    xc = xyz[..., :2] - xyz[..., :2].mean(axis=-2, keepdims=True)
    uc = uv - uv.mean(axis=-2, keepdims=True)
    denom = np.sum(xc * xc, axis=(-2, -1))
    if np.any(side <= 1e-8) or np.any(denom <= 1e-12):
        raise ValueError('Degenerate predicted box or 3D spread')
    a = np.sum(xc * uc, axis=(-2, -1)) / denom
    if np.any(a <= 1e-8):
        raise ValueError('Nonpositive predicted weak-camera scale')
    b = uv.mean(axis=-2) - a[..., None] * xyz[..., :2].mean(axis=-2)
    q = a / side
    t = np.concatenate(((b-origin)/side[..., None], np.zeros((*q.shape, 1))), axis=-1)
    return (xyz*q[..., None, None]+t[..., None, :],
            (uv-origin[..., None, :])/side[..., None, None], q, t, origin, side)


def restore_left_gauge(projected, q, t, native_left):
    projected, native_left = np.asarray(projected), np.asarray(native_left)
    if projected.shape != native_left.shape or projected.shape[-2:] != (13, 3):
        raise ValueError('Expected matching left-camera common13 poses')
    if np.any(np.asarray(q) <= 0):
        raise ValueError('Scale must be positive')
    raw = (projected - np.asarray(t)[..., None, :]) / np.asarray(q)[..., None, None]
    raw += native_left[..., [4, 5], :].mean(-2, keepdims=True) - raw[..., [4, 5], :].mean(-2, keepdims=True)
    return finite(raw, 'restored prediction').astype(np.float32)


def fit_point_uncertainty(residuals):
    r = finite(np.asarray(residuals, dtype=np.float64), 'training 2D residuals')
    if r.shape[-2:] != (13, 2) or r.size == 0:
        raise ValueError('Expected residuals ending in (13,2)')
    r = r.reshape(-1, 13, 2)
    weights, variances = [], []
    for j in range(13):
        distance = np.sum(r[:, j]**2, axis=-1)
        v = np.maximum(np.quantile(distance/2, [.1, .4, .7, .95]), 1e-6)
        w = np.full(4, .25)
        for _ in range(30):
            logp = np.log(w)[None] - np.log(v)[None] - distance[:, None]/(2*v[None])
            prob = np.exp(logp-logp.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            mass = np.maximum(prob.sum(axis=0), 1e-12)
            v = np.maximum((prob*distance[:, None]).sum(axis=0)/(2*mass), 1e-6)
            w = np.maximum(mass/mass.sum(), 1e-8); w /= w.sum()
        order = np.argsort(v)
        weights.append(w[order].tolist()); variances.append(v[order].tolist())
    return dict(weights=weights, variances=variances, components=4,
                center='estimated point; no GT center', fit_split='train',
                fit_observations_per_joint=len(r), em_iterations=30, variance_floor=1e-6)


def point_mixtures(uv, uncertainty):
    uv = finite(np.asarray(uv, dtype=np.float64), 'normalized estimated 2D')
    if uv.shape[-3:] != (2, 13, 2):
        raise ValueError('Expected (...,2,13,2) estimated observations')
    w, v = (np.asarray(uncertainty[k]) for k in ('weights', 'variances'))
    if w.shape != (13, 4) or v.shape != (13, 4) or not np.allclose(w.sum(-1), 1):
        raise ValueError('Invalid frozen mixture statistics')
    if np.any(w <= 0) or np.any(v < 1e-6):
        raise ValueError('Nonpositive mixture weight or variance')
    out = np.empty((*uv.shape[:-1], 4, 4), dtype=np.float64)
    out[..., 0] = w
    out[..., 1:3] = uv[..., None, :]
    out[..., 3] = v
    return finite(out, 'point mixtures')


class NativeInputs:
    """Verified full split, strictly prediction/identity fields only."""
    def __init__(self, cache_dir, dataset_root):
        from dual2pose.eval.main_baseline_geometry import _load_index, _match_source_samples
        self.cache = Path(cache_dir).resolve()
        self.root = Path(dataset_root).resolve()
        self.manifest = json.loads((self.cache/'manifest.json').read_text())
        m = self.manifest
        self.dataset, self.split, self.count = m['dataset'], m['split'], m['sample_count']
        if self.dataset not in ('unity', 'ski') or self.split not in ('train', 'val', 'test'):
            raise ValueError('Unknown dataset/split')
        if m['representation'] != 'raw_pre_batch_canonicalization' or m['time_window'] != 30:
            raise ValueError('Wrong input representation')
        for name in ('left.npy', 'right.npy', 'frames.npy', 'samples.jsonl'):
            if sha256(self.cache/name) != m['files'][name]:
                raise ValueError(f'Input hash mismatch: {name}')
        index = Path(m['index'])
        if sha256(index) != m['index_sha256']:
            raise ValueError('Index hash mismatch')
        samples = [json.loads(s) for s in (self.cache/'samples.jsonl').read_text().splitlines()]
        if [s['index'] for s in samples] != list(range(self.count)):
            raise ValueError('Sample order mismatch')
        self.sources = _match_source_samples(self.dataset, samples, _load_index(index, self.split))
        self.frames = np.load(self.cache/'frames.npy', mmap_mode='r')
        joints = 15 if self.dataset == 'unity' else 13
        self.views = [np.load(self.cache/f'{v}.npy', mmap_mode='r') for v in ('left', 'right')]
        if any(v.shape != (self.count, 30, joints, 3) for v in self.views):
            raise ValueError('Native input shape mismatch')
        if self.frames.shape != (self.count, 30) or np.any(self.frames < 0) or np.any(np.diff(self.frames, axis=1) < 0):
            raise ValueError('Invalid frames')
        self.observation_hashes = {}

    @lru_cache(maxsize=8192)
    def directory(self, archived):
        from dual2pose.eval.main_baseline_geometry import _resolve_dataset_path
        root = _resolve_dataset_path(archived, self.root)
        paths = root.glob('kpt2d_*.npy' if self.dataset == 'unity' else '*_sam3d_body.npz')
        found = {}
        for p in paths:
            frame = int(p.stem.rsplit('_', 1)[-1] if self.dataset == 'unity' else p.name.split('_', 1)[0])
            if frame in found:
                raise ValueError(f'Duplicate frame {frame}: {root}')
            found[frame] = p
        return found

    @lru_cache(maxsize=400000)
    def observation(self, path):
        from dual2pose.map_config import filter_sam3d_body_kpts
        if self.dataset == 'unity':
            native = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=True) as archive:
                native = archive['output'].item()['pred_keypoints_2d']
        points = finite(filter_sam3d_body_kpts(native)[2:15], str(path))
        if points.shape != (13, 2):
            raise ValueError(f'Wrong estimated 2D shape: {path}')
        self.observation_hashes[str(path)] = sha256(path)
        return points

    def batch(self, start, stop):
        raw = np.stack([np.array(v[start:stop, :, -13:, :]) for v in self.views], axis=2)
        keys = ('sam3d_cam1_kpt2d_dir', 'sam3d_cam2_kpt2d_dir') if self.dataset == 'unity' else ('cam1_sam3d_kpt2d_dir', 'cam2_sam3d_kpt2d_dir')
        uv = np.empty((stop-start, 30, 2, 13, 2), dtype=np.float32)
        for i in range(start, stop):
            for view, key in enumerate(keys):
                paths = self.directory(str(self.sources[i][key]))
                uv[i-start, :, view] = [self.observation(paths[int(f)]) for f in self.frames[i]]
        return raw, uv
