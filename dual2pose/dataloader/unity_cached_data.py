"""Opt-in verified raw Unity cache; preserves the original training batch contract.

Only storage changes: values are native 15-joint poses before canonicalization.
The inherited DataLoader policy preserves shuffle, drop_last and worker seeding.
No random operations occur while loading or verifying this cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dual2pose.dataloader.data_loader import UnityDataModule
from dual2pose.eval.main_baseline_data import sha256


class RawUnityCacheDataset(Dataset):
    def __init__(self, root, *, split, index_path, index_mapping, target_t=30):
        self.root = Path(root)
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        expected = dict(dataset='unity', split=split, sample_count=len(index_mapping),
                        time_window=target_t, num_joints=15,
                        representation='raw_pre_batch_canonicalization',
                        index_sha256=sha256(index_path))
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f'Unity raw cache {key} mismatch: {self.manifest.get(key)!r} != {value!r}')
        names = {'left.npy', 'right.npy', 'target.npy', 'frames.npy', 'samples.jsonl'}
        if set(self.manifest.get('files', {})) != names:
            raise ValueError('Unity raw cache file manifest mismatch')
        for name in sorted(names):
            if sha256(self.root / name) != self.manifest['files'][name]:
                raise ValueError(f'Unity raw cache checksum mismatch: {name}')
        self.samples = [json.loads(line) for line in (self.root/'samples.jsonl').read_text().splitlines()]
        if len(self.samples) != len(index_mapping):
            raise ValueError('Unity raw cache metadata count mismatch')
        for i, (sample, source) in enumerate(zip(self.samples, index_mapping)):
            if sample.get('index') != i or sample.get('num_aligned_frames') != target_t:
                raise ValueError(f'Unity raw cache metadata order/frame mismatch at {i}')
            for key in ('person_id', 'action_id', 'cam1_id', 'cam2_id'):
                if sample.get(key) != source.get(key, 'unknown'):
                    raise ValueError(f'Unity raw cache metadata mismatch at {i}: {key}')
        self.arrays = {name: np.load(self.root/f'{name}.npy', mmap_mode='r', allow_pickle=False)
                       for name in ('left', 'right', 'target', 'frames')}
        for name, array in self.arrays.items():
            shape = (len(index_mapping), target_t) + (() if name=='frames' else (15,3))
            dtype = np.int64 if name=='frames' else np.float32
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f'Unity raw cache shape/dtype mismatch: {name}')
            for start in range(0, len(array), 4096):
                if not np.isfinite(array[start:start+4096]).all():
                    raise ValueError(f'Unity raw cache nonfinite values: {name}')
        self.provenance = dict(manifest=str(self.manifest_path.resolve()),
                               manifest_sha256=sha256(self.manifest_path), **expected)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Copy only this item to writable memory: trainer and caller mutations
        # must not alter the immutable memory-mapped cache.
        values = {name: torch.from_numpy(np.array(array[index], copy=True))
                  for name, array in self.arrays.items()}
        return dict(frame_indices=values['frames'],
                    meta={k:v for k,v in self.samples[index].items() if k!='index'},
                    kpt3d_gt=values['target'], kpt3d_gt_canonical={},
                    kpt3d_sam=dict(cam1=values['left'], cam2=values['right']),
                    kpt3d_sam_canonical={})


class VerifiedCacheUnityDataModule(UnityDataModule):
    """cache_root contains train/, val/, test/ for the selected Unity fold."""
    def __init__(self, opt, cache_root=None):
        super().__init__(opt)
        if self._load_frames or self._load_2d_kpt or not self._load_3d_kpt:
            raise ValueError('Raw Unity cache requires 3D-keypoints-only training')
        if self._time_window != 30:
            raise ValueError('Existing Unity raw caches require time_window=30')
        self.cache_root = Path(cache_root if cache_root is not None else opt.data.unity.raw_cache_root)
        self.cache_provenance = {}

    def setup(self, stage=None):
        for split in ('train','val','test'):
            attribute = f'{split}_gait_dataset'
            if hasattr(self, attribute):
                continue
            dataset = RawUnityCacheDataset(self.cache_root/split, split=split,
                index_path=self._index_mapping, index_mapping=self._dataset_idx[split],
                target_t=self._time_window)
            setattr(self, attribute, dataset)
            self.cache_provenance[split] = dataset.provenance
