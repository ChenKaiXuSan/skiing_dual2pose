"""Immutable raw-pose caches for the archived IVC main-table protocol.

Canonicalization deliberately happens AFTER each training batch is shuffled.
Caching canonical poses would silently replace the archived batch anchor.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dual2pose.trainer.canonicalize import canonicalize_pose_torch

ROOT = Path(__file__).resolve().parents[2]
_NP_LOAD = np.load
EXPECTED = {"unity": {"train": 193320, "val": 128880, "test": 64440},
            "ski": {"train": 145, "test": 30}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def batch_slices(count, batch_size, drop_last=False):
    if batch_size <= 0 or count < 0:
        raise ValueError("Invalid batch size or sample count")
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        if drop_last and stop - start != batch_size:
            break
        yield start, stop


def canonical_batch(left, right, target, *, dataset):
    if dataset not in EXPECTED:
        raise ValueError(dataset)
    kwargs = {} if dataset == "unity" else {"left_hip": 4, "right_hip": 5, "neck": 12}
    return tuple(canonicalize_pose_torch(x, **kwargs)[0] for x in (left, right, target))


@lru_cache(maxsize=32768)
def _cached_npy(path):
    array = _NP_LOAD(path)
    # Return the same read-only values to the original dataset. It never mutates
    # source arrays; torch conversion/filtering remains in the original loader.
    array.setflags(write=False)
    return array


def _load_with_npy_cache(file, *args, **kwargs):
    if isinstance(file, (str, Path)) and str(file).endswith(".npy") and not args and not kwargs:
        return _cached_npy(str(file))
    return _NP_LOAD(file, *args, **kwargs)


def _worker_init(worker_id):
    torch.set_num_threads(1)
    np.load = _load_with_npy_cache


def create_datasets(dataset, splits):
    from omegaconf import OmegaConf
    config = OmegaConf.load(ROOT / "configs/dual2pose.yaml")
    if dataset == "unity":
        from dual2pose.dataloader.data_loader import UnityDataModule
        dm = UnityDataModule(config)
        dm.prepare_data()
        dm.setup()
        index = Path(config.data.unity.index_mapping_path)
        datasets = {s: getattr(dm, f"{s}_gait_dataset") for s in splits}
        collate = None
    else:
        from dual2pose.dataloader.ski_poseptz_dataset_dual_view import LabeledSkiPosePTZDataset
        from dual2pose.eval.eval_ski_poseptz import _collate_ski_poseptz_batch
        cfg = config.data.ski_pose_ptz
        index = Path(cfg.index_mapping_path)
        datasets = {s: LabeledSkiPosePTZDataset(
            index, transform=None, load_frames=False, load_2d_kpt=False,
            load_3d_kpt=True, target_t=30, split=s,
            path_rewrite_from=str(cfg.index_path_rewrite_from),
            path_rewrite_to=str(cfg.root_path)) for s in splits}
        collate = _collate_ski_poseptz_batch
    return datasets, collate, index


def export_cache(dataset, splits, output, workers=8):
    datasets, collate, index = create_datasets(dataset, splits)
    index_hash = sha256(index)
    for split, source in datasets.items():
        dest = Path(output) / dataset / split
        dest.mkdir(parents=True, exist_ok=True)
        manifest_path = dest / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["index_sha256"] != index_hash or previous["sample_count"] != len(source):
                raise RuntimeError(f"Existing cache has different provenance: {dest}")
            for filename, digest in previous["files"].items():
                if sha256(dest / filename) != digest:
                    raise RuntimeError(f"Existing cache checksum mismatch: {dest / filename}")
            print(f"Verified existing cache {dest}", flush=True)
            continue
        if any(dest.glob("*.npy")):
            raise RuntimeError(f"Incomplete cache exists: {dest}; use a new output directory")
        count, joints = len(source), (15 if dataset == "unity" else 13)
        if count != EXPECTED[dataset][split]:
            raise RuntimeError(f"Unexpected {dataset}/{split} count: {count}")
        arrays = {name: np.lib.format.open_memmap(dest / f"{name}.npy", mode="w+",
                  dtype=np.float32, shape=(count, 30, joints, 3))
                  for name in ("left", "right", "target")}
        frames = np.lib.format.open_memmap(dest / "frames.npy", mode="w+", dtype=np.int64,
                                          shape=(count, 30))
        loader = DataLoader(source, batch_size=256 if dataset == "unity" else 4,
                            shuffle=False, drop_last=False, num_workers=workers,
                            collate_fn=collate, worker_init_fn=_worker_init,
                            persistent_workers=workers > 0)
        if workers == 0:
            _worker_init(0)
        offset, started = 0, time.monotonic()
        with (dest / "samples.jsonl").open("w") as meta_file:
            for step, batch in enumerate(loader):
                views = batch["kpt3d_sam"]
                left, right = views["cam1"], views["cam2"]
                target = batch["kpt3d_gt"]
                size = len(left)
                for name, tensor in zip(arrays, (left, right, target)):
                    array = tensor.numpy().astype(np.float32, copy=False)
                    if array.shape != (size, 30, joints, 3) or not np.isfinite(array).all():
                        raise RuntimeError(f"Invalid pose in {dataset}/{split}/{offset}: {name}")
                    arrays[name][offset:offset + size] = array
                frames[offset:offset + size] = batch["frame_indices"].numpy()
                for i in range(size):
                    record = {"index": offset+i}
                    for key, values in batch["meta"].items():
                        value = values[i]
                        record[key] = value.item() if torch.is_tensor(value) else value
                    meta_file.write(json.dumps(record) + "\n")
                offset += size
                if step % 20 == 0 or offset == count:
                    progress = {"pid": os.getpid(), "dataset": dataset, "split": split,
                                "completed": offset, "total": count,
                                "elapsed_seconds": time.monotonic()-started}
                    write_json(dest / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        np.load = _NP_LOAD
        if offset != count:
            raise RuntimeError("Incomplete cache")
        for array in [*arrays.values(), frames]:
            array.flush()
        files = [f"{name}.npy" for name in arrays] + ["frames.npy", "samples.jsonl"]
        write_json(manifest_path, {"dataset": dataset, "split": split,
                   "sample_count": count, "time_window": 30, "num_joints": joints,
                   "indexed_sample_count": 150 if dataset == "ski" and split == "train" else count,
                   "representation": "raw_pre_batch_canonicalization",
                   "index": str(index), "index_sha256": index_hash,
                   "config_sha256": sha256(ROOT / "configs/dual2pose.yaml"),
                   "export_code_sha256": sha256(__file__),
                   "files": {p: sha256(dest / p) for p in files}})
        print(f"Completed verified {dataset}/{split}: {count} samples", flush=True)


def load_cache(path, device="cpu"):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if sha256(path / filename) != expected:
            raise RuntimeError(f"Cache checksum mismatch: {path / filename}")
    expected_shape = (manifest["sample_count"], manifest["time_window"], manifest["num_joints"], 3)
    tensors = []
    for name in ("left", "right", "target"):
        values = _NP_LOAD(path / f"{name}.npy")
        if values.shape != expected_shape:
            raise RuntimeError(f"Cache shape mismatch: {name}: {values.shape} vs {expected_shape}")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError(f"Cache dtype or finite-value violation: {name}")
        tensors.append(torch.from_numpy(values).to(device))
    return tuple(tensors), manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(EXPECTED), required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "train", "val"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    export_cache(args.dataset, args.splits, args.output, args.workers)
