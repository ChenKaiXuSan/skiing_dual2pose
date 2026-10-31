#!/usr/bin/env python3
"""Bounded Unity diagnostic. Writes only a new report, never training artifacts.

Use --repo to import the exact frozen training source. CPU tests read a few
samples. --gpu is opt-in and refuses a GPU with existing compute processes.
The early-sampling experiment is an in-process subclass, not a source edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='saved Hydra config or Lightning hparams.yaml')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--cache-root', type=Path, help='optional raw Unity cache directory with train/val/test')
    parser.add_argument('--loader-batches', type=int, default=0, help='bounded raw-cache loader throughput, 0 disables')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu-batch', type=int, default=4096)
    parser.add_argument('--gpu-steps', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a fresh report path')
    if not 0 <= args.loader_batches <= 47:
        parser.error('loader-batches must be 0..47')
    if args.loader_batches and not args.cache_root:
        parser.error('--loader-batches requires --cache-root')
    if not 1 <= args.samples <= 32 or not 1 <= args.gpu_steps <= 10:
        parser.error('samples must be 1..32 and gpu-steps 1..10')
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from dual2pose.dataloader.data_loader import UnityDataModule
    from dual2pose.dataloader.unity_dataset_dual_view import DualViewUnityDataset
    from dual2pose.dataloader.utils import uniform_subsample_along_dim
    from dual2pose.training_protocol import resolve_fold_index_path, validate_fold_metadata
    from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer

    cfg = OmegaConf.load(args.config)
    if 'hparams' in cfg:
        cfg = cfg.hparams
    cfg.data.unity.root_path = str(args.root.resolve())
    index = resolve_fold_index_path(args.root.resolve(), int(cfg.train.fold))
    validate_fold_metadata(index, int(cfg.train.fold))
    cfg.data.unity.index_mapping_path = str(index)
    if cfg.data.load_frames or cfg.data.load_2d_kpt or not cfg.data.load_3d_kpt:
        parser.error('diagnostic requires the keypoints-only Unity training config')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    dm = UnityDataModule(cfg)
    dm.prepare_data()
    dm.setup('fit')

    class EarlySampleDataset(DualViewUnityDataset):
        def _load_single_variant_keypoints(self, **kwargs):
            indices = torch.tensor(kwargs['common_idx'], dtype=torch.long)
            kwargs['common_idx'] = uniform_subsample_along_dim(indices, self._target_t, 0).tolist()
            return super()._load_single_variant_keypoints(**kwargs)

    report = dict(repo=str(args.repo.resolve()), config=str(args.config.resolve()),
                  torch_version=torch.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  num_workers=int(cfg.data.num_workers), batch_size=int(cfg.data.batch_size),
                  time_window=int(cfg.data.time_window), canonical_mode='per_frame',
                  index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(), samples=[])
    for rel in ('dual2pose/dataloader/unity_dataset_dual_view.py',
                'dual2pose/models/crossview_fusion.py', 'dual2pose/trainer/canonicalize.py'):
        report.setdefault('source_sha256', {})[rel] = hashlib.sha256((args.repo / rel).read_bytes()).hexdigest()

    def save():
        # The path was reserved exclusively below; only this diagnostic owns it.
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f)

    def measure(dataset, idx):
        count = 0
        load_seconds = 0.0
        original = np.load
        def counted(*a, **kw):
            nonlocal count, load_seconds
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                count += 1
                load_seconds += time.perf_counter() - start
        start = time.perf_counter()
        with patch.object(np, 'load', counted):
            sample = dataset[idx]
        return sample, dict(seconds=time.perf_counter() - start,
                            np_load_count=count, np_load_seconds=load_seconds)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        return a == b

    datasets = [('train', dm.train_gait_dataset), ('val', dm.val_gait_dataset)]
    if args.cache_root:
        datasets.append(('test', dm.test_gait_dataset))
    for split, dataset in datasets:
        cached = None
        if args.cache_root:
            from dual2pose.dataloader.unity_cached_data import RawUnityCacheDataset
            started = time.perf_counter()
            cached = RawUnityCacheDataset(args.cache_root / split, split=split,
                index_path=index, index_mapping=dataset._index_mapping, target_t=int(cfg.data.time_window))
            report.setdefault('cache_verification', {})[split] = dict(
                seconds=time.perf_counter()-started, **cached.provenance)
            save()
        fast = EarlySampleDataset(dataset._index_mapping, load_frames=False,
                                  load_2d_kpt=False, load_3d_kpt=True, load_mask=False,
                                  target_t=int(cfg.data.time_window))
        positions = torch.linspace(0, len(dataset)-1, args.samples).round().long().tolist()
        for idx in positions:
            baseline, first = measure(dataset, idx)
            repeat, warm = measure(dataset, idx)
            candidate, early = measure(fast, idx)
            row = dict(split=split, index=idx, meta=baseline['meta'], baseline=first,
                       baseline_warm=warm, early_sample=early,
                       exact_repeat=equal(baseline, repeat), exact_early_sample=equal(baseline, candidate))
            if cached is not None:
                cache_sample, cache_timing = measure(cached, idx)
                row['raw_cache'] = cache_timing
                row['exact_raw_cache'] = equal(baseline, cache_sample)
            report['samples'].append(row)
            save()
            print(json.dumps(row), flush=True)
            if not row['exact_repeat'] or not row['exact_early_sample'] or not row.get('exact_raw_cache', True):
                raise RuntimeError('exact-equivalence check failed')

        if cached is not None and args.loader_batches and split in ('train', 'val'):
            from torch.utils.data import DataLoader, Subset
            count = min(len(cached), args.loader_batches * int(cfg.data.batch_size))
            # Select across the whole cache using an isolated RNG. Only the
            # requested number of samples is prefetched, bounding I/O.
            generator = torch.Generator().manual_seed(int(cfg.train.seed))
            indices = torch.randperm(len(cached), generator=generator)[:count].tolist()
            loader = DataLoader(Subset(cached, indices), batch_size=int(cfg.data.batch_size),
                                num_workers=int(cfg.data.num_workers), pin_memory=False,
                                shuffle=False, drop_last=split == 'train')
            started = time.perf_counter()
            last = started
            times = []
            seen = 0
            for loaded in loader:
                now = time.perf_counter()
                times.append(now-last)
                seen += len(loaded['kpt3d_gt'])
                last = time.perf_counter()
            total = time.perf_counter() - started
            report.setdefault('cache_loader', {})[split] = dict(
                batches=len(times), samples=seen, seconds=total, samples_per_second=seen/total,
                batch_wait_seconds=times, workers=int(cfg.data.num_workers),
                includes_worker_start_and_shutdown=True,
                note='bounded randomly selected raw-cache subset; CPU loader only')
            save()
            print(json.dumps({'cache_loader': {split: report['cache_loader'][split]}}), flush=True)

    if args.gpu:
        if not os.environ.get('PBS_JOBID'):
            raise RuntimeError('--gpu requires a separate allocated PBS diagnostic job')
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True)
        if query.stdout.strip():
            raise RuntimeError('existing GPU compute process detected; refusing interference')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.manual_seed(int(cfg.train.seed))
        model = CrossViewFusionTrainer(cfg).cuda().train()
        model.log = lambda *a, **k: None
        batch = dict(kpt3d_sam=dict(cam1=torch.randn(args.gpu_batch, int(cfg.data.time_window), 15, 3, device='cuda'),
                                    cam2=torch.randn(args.gpu_batch, int(cfg.data.time_window), 15, 3, device='cuda')),
                     kpt3d_gt=torch.randn(args.gpu_batch, int(cfg.data.time_window), 15, 3, device='cuda'))
        optimizer = torch.optim.AdamW(model.models.parameters(), lr=model.lr, weight_decay=model.weight_decay)
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.gpu_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = model._shared_step(batch, 'train')
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if i:
                elapsed.append(duration)
            print(json.dumps(dict(gpu_step=i, seconds=duration, loss=float(loss))), flush=True)
        report['synthetic_gpu'] = dict(device=torch.cuda.get_device_name(), batch=args.gpu_batch,
            steps_seconds=elapsed, median_seconds=statistics.median(elapsed),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            caveat='synthetic shapes and complete shared_step; excludes loader and Lightning logging')
        model.eval()
        validation_elapsed = []
        with torch.no_grad():
            for i in range(args.gpu_steps + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss = model._shared_step(batch, 'val')
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                if i:
                    validation_elapsed.append(duration)
                print(json.dumps(dict(gpu_validation_step=i, seconds=duration, loss=float(loss))), flush=True)
        report['synthetic_gpu']['validation_steps_seconds'] = validation_elapsed
        report['synthetic_gpu']['validation_median_seconds'] = statistics.median(validation_elapsed)
        report['synthetic_gpu']['shape_matched_compute_epoch_seconds'] = (
            (len(dm.train_gait_dataset) // args.gpu_batch) * statistics.median(elapsed)
            + ((len(dm.val_gait_dataset) + args.gpu_batch - 1) // args.gpu_batch)
            * statistics.median(validation_elapsed)
        ) if args.gpu_batch == int(cfg.data.batch_size) else None
        report['synthetic_gpu']['epoch_estimate_caveat'] = (
            '47 train + 32 validation batches; compute-only estimate, ignores partial final validation batch')
        save()
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
