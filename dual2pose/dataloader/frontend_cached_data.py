"""Verified, opt-in storage acceleration for frontend adaptation.

Arrays contain the exact raw outputs of FrontEndManifest.load_pose, before the
trainer's per-frame canonicalization. Source-block order and DataLoader policy
are preserved. Only a completed manifest publishes a newly built cache.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule

from dual2pose.dataloader.unity_cached_data import RawUnityCacheDataset
from dual2pose.eval import frontend_manifest
from dual2pose.eval.frontend_manifest import FrontEndManifest
from dual2pose.eval.main_baseline_data import sha256


def implementation_digest():
    """Invalidate caches when packing or source conversion implementation changes."""
    return hashlib.sha256(('frontend-aligned-cache-v1\n' + sha256(__file__) + '\n' +
                           sha256(frontend_manifest.__file__)).encode()).hexdigest()


def _as_manifest(source):
    return source if isinstance(source, FrontEndManifest) else FrontEndManifest.load(Path(source))


def _source_hashes(source):
    # Key rather than absolute path identifies a stream, so cache directories can
    # move with the same manifest and pose contents across cluster nodes.
    hashes = {path: sha256(path) for path in set(source.entries.values())}
    return [[*key, hashes[path]] for key, path in sorted(source.entries.items())]


def _expected(base_dataset, source, split):
    if not isinstance(base_dataset, RawUnityCacheDataset):
        raise TypeError('Frontend cache requires a verified RawUnityCacheDataset')
    if split not in ('train', 'val') or base_dataset.manifest['split'] != split:
        raise ValueError('Frontend cache split mismatch; adaptation uses train/val only')
    source.validate_coverage(base_dataset.samples)
    return dict(format_version=1, dataset='unity', split=split,
                frontend_name=source.frontend_name, sample_count=len(base_dataset),
                time_window=base_dataset.manifest['time_window'], num_joints=15,
                representation='raw_frontend_pre_batch_canonicalization',
                raw_manifest_sha256=base_dataset.provenance['manifest_sha256'],
                frontend_manifest_sha256=sha256(source.source_path),
                joint_indices=list(source.joint_indices) if source.joint_indices is not None else None,
                implementation_sha256=implementation_digest(), source_sha256=_source_hashes(source))


class _OneStreamManifest(FrontEndManifest):
    """Builder groups requests by path; retain at most one decoded source stream."""
    def _load_pose_arrays(self, pose_path):
        if getattr(self, '_cached_path', None) != pose_path:
            object.__setattr__(self, '_cached_arrays', super()._load_pose_arrays(pose_path))
            object.__setattr__(self, '_cached_path', pose_path)
        return self._cached_arrays


def build_frontend_cache(root, base_dataset, source, *, split):
    """Build once, refusing to replace any existing destination.

Each source file is decoded once regardless of camera-pair multiplicity. The
unchanged load_pose method performs frame selection, interpolation, and joints.
Source checksums are checked again before the completion manifest is published.
"""
    root = Path(root)
    source = _as_manifest(source)
    expected = _expected(base_dataset, source, split)
    root.mkdir(parents=True, exist_ok=False)
    shape = (len(base_dataset), expected['time_window'], 15, 3)
    arrays = {name: np.lib.format.open_memmap(root / f'{name}.npy', mode='w+',
                                            dtype=np.float32, shape=shape)
              for name in ('left', 'right')}
    requests = defaultdict(list)
    for index, sample in enumerate(base_dataset.samples):
        for name, field in (('left', 'cam1_id'), ('right', 'cam2_id')):
            camera = str(sample[field])
            key = (str(sample['person_id']), str(sample['action_id']), camera)
            requests[source.entries[key]].append((index, name, camera))
    cached_source = _OneStreamManifest(source.frontend_name, source.entries, source.source_path,
                                      source.joint_indices, source.metadata)
    for requests_for_path in requests.values():
        aligned = OrderedDict()
        for index, name, camera in requests_for_path:
            frames = base_dataset.arrays['frames'][index]
            key = frames.tobytes()
            if key not in aligned:
                pose = cached_source.load_pose(
                    base_dataset.samples[index], camera, expected['time_window'], 15,
                    target_frame_indices=torch.from_numpy(np.array(frames, copy=True)))
                aligned[key] = pose.numpy()
                if len(aligned) > 128:
                    aligned.popitem(last=False)
            aligned.move_to_end(key)
            arrays[name][index] = aligned[key]
    for array in arrays.values():
        array.flush()
    if _expected(base_dataset, source, split) != expected:
        raise ValueError('Frontend cache source mismatch during build')
    manifest = dict(expected, files={f'{name}.npy': sha256(root / f'{name}.npy')
                                     for name in ('left', 'right')})
    # All arrays and source checks complete before the only readiness marker.
    pending = root / 'manifest.pending.json'
    pending.write_text(json.dumps(manifest, indent=2) + '\n')
    pending.replace(root / 'manifest.json')
    return manifest


class VerifiedFrontEndPoseDataset(Dataset):
    def __init__(self, base_dataset, source, root, *, split):
        self.base_dataset = base_dataset
        self.source = _as_manifest(source)
        self.root = Path(root)
        manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        expected = _expected(base_dataset, self.source, split)
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f'Frontend cache {key} mismatch')
        if set(self.manifest.get('files', {})) != {'left.npy', 'right.npy'}:
            raise ValueError('Frontend cache file manifest mismatch')
        self.arrays = {}
        shape = (len(base_dataset), expected['time_window'], 15, 3)
        for name in ('left', 'right'):
            path = self.root / f'{name}.npy'
            if sha256(path) != self.manifest['files'][path.name]:
                raise ValueError(f'Frontend cache checksum mismatch: {name}')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f'Frontend cache shape/dtype mismatch: {name}')
            for start in range(0, len(array), 1024):
                if not np.isfinite(array[start:start + 1024]).all():
                    raise ValueError(f'Frontend cache finite-value mismatch: {name}')
            self.arrays[name] = array
        self.provenance = dict(manifest=str(manifest_path.resolve()),
                               manifest_sha256=sha256(manifest_path), **expected)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        sample['kpt3d_sam'] = {
            key: torch.from_numpy(np.array(self.arrays[name][index], copy=True))
            for key, name in (('cam1', 'left'), ('cam2', 'right'))}
        sample['_frontend_name'] = self.source.frontend_name
        return sample


class VerifiedMixedFrontEndDataset(Dataset):
    def __init__(self, base_dataset, sources, root, *, split):
        sources = tuple(None if source is None else _as_manifest(source) for source in sources)
        self.source_names = tuple('sam3d' if source is None else source.frontend_name for source in sources)
        if not len(base_dataset) or not sources or len(set(self.source_names)) != len(sources):
            raise ValueError('Mixed frontend cache requires data and unique nonempty sources')
        self.base_dataset = base_dataset
        self.datasets = tuple(base_dataset if source is None else VerifiedFrontEndPoseDataset(
            base_dataset, source, Path(root) / source.frontend_name / split, split=split)
            for source in sources)
        self.provenance = {name: dataset.provenance for name, dataset in zip(self.source_names, self.datasets)}

    def __len__(self):
        return len(self.base_dataset) * len(self.datasets)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index, base_index = divmod(index, len(self.base_dataset))
        sample = self.datasets[source_index][base_index]
        sample['_frontend_name'] = self.source_names[source_index]
        return sample


class VerifiedFrontendDataModule(LightningDataModule):
    """Train/val-only adapter; inherit the supplied Unity loader policies."""
    def __init__(self, base_dm, cache_root, *, train_sources, val_sources):
        super().__init__()
        from dual2pose.dataloader.unity_cached_data import VerifiedCacheUnityDataModule
        if not isinstance(base_dm, VerifiedCacheUnityDataModule):
            raise TypeError('Frontend cache requires VerifiedCacheUnityDataModule')
        self.base_dm = base_dm
        self.cache_root = Path(cache_root)
        self.sources = dict(train=tuple(train_sources), val=tuple(val_sources))
        self.cache_provenance = {}
        self._wrapped = False

    def prepare_data(self):
        self.base_dm.prepare_data()

    def setup(self, stage=None):
        if stage not in (None, 'fit', 'validate'):
            raise ValueError('Frontend cache supports adaptation train/val only')
        if self._wrapped:
            return
        for split in ('train', 'val'):
            # Calling base_dm.setup would load test GT as well. Adaptation needs
            # only these two splits; preserve the base verification contract.
            raw = RawUnityCacheDataset(self.base_dm.cache_root / split, split=split,
                index_path=self.base_dm._index_mapping,
                index_mapping=self.base_dm._dataset_idx[split], target_t=self.base_dm._time_window)
            self.base_dm.cache_provenance[split] = raw.provenance
            dataset = VerifiedMixedFrontEndDataset(raw, self.sources[split], self.cache_root, split=split)
            setattr(self.base_dm, f'{split}_gait_dataset', dataset)
            self.cache_provenance[split] = dataset.provenance
        self._wrapped = True

    def train_dataloader(self):
        return self.base_dm.train_dataloader()

    def val_dataloader(self):
        return self.base_dm.val_dataloader()


def main(argv=None):
    """Build/verify one frontend split without model or GPU initialization."""
    import argparse
    import time
    from dual2pose.training_protocol import validate_fold_metadata
    from dual2pose.eval.frontend_manifest import FrontEndPoseDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-cache-root', type=Path, required=True)
    parser.add_argument('--index-path', type=Path, required=True)
    parser.add_argument('--frontend-manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--mode', choices=('build', 'verify'), default='build')
    parser.add_argument('--verify-samples', type=int, default=24)
    args = parser.parse_args(argv)
    if args.verify_samples < 1:
        parser.error('--verify-samples must be positive')
    started = time.monotonic()
    validate_fold_metadata(args.index_path, args.fold)
    index = json.loads(args.index_path.read_text())[args.split]
    raw = RawUnityCacheDataset(args.raw_cache_root / args.split, split=args.split,
                              index_path=args.index_path, index_mapping=index)
    source = FrontEndManifest.load(args.frontend_manifest)
    if Path(source.frontend_name).name != source.frontend_name or source.frontend_name in ('.', '..'):
        raise ValueError('Frontend name must be a directory basename')
    root = args.cache_root / source.frontend_name / args.split
    if args.mode == 'build':
        build_frontend_cache(root, raw, source, split=args.split)
    dataset = VerifiedFrontEndPoseDataset(raw, source, root, split=args.split)
    reference = FrontEndPoseDataset(raw, source)
    selected = np.linspace(0, len(raw) - 1, min(args.verify_samples, len(raw)), dtype=np.int64)
    for index in selected:
        actual, expected = dataset[int(index)], reference[int(index)]
        for camera in ('cam1', 'cam2'):
            if not torch.equal(actual['kpt3d_sam'][camera], expected['kpt3d_sam'][camera]):
                raise ValueError(f'Frontend cache exact-value mismatch at {index}/{camera}')
    print(json.dumps(dict(mode=args.mode, root=str(root.resolve()), sample_count=len(dataset),
                          verified_samples=len(selected), max_abs_difference=0,
                          manifest_sha256=dataset.provenance['manifest_sha256'],
                          elapsed_seconds=time.monotonic() - started), sort_keys=True))


if __name__ == '__main__':
    main()
