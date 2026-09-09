"""Train and evaluate calibrated DLT baselines for the IVC main tables.

The runner consumes immutable raw-pose and geometry exports.  Geometry is
independently body-canonicalized together with the raw target at the complete
protocol batch size.  The learned method sees only canonical DLT,
reprojection error, and DLT validity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

from dual2pose.eval.main_baseline_data import (
    ROOT,
    batch_slices,
    canonical_batch,
    sha256,
    write_json,
)
from dual2pose.experiments.run_main_baselines import MetricAccumulator
from dual2pose.models.dlt_residual_baseline import (
    DLTResidualMLP,
    dlt_residual_loss,
    ski_predicted_pelvis_relative,
)


EXPECTED = {
    "unity": {"train": (193320, 30, 15, 3), "val": (128880, 30, 15, 3), "test": (64440, 30, 15, 3)},
    "ski": {"train": (145, 30, 13, 3), "test": (30, 30, 13, 3)},
}
GEOMETRY_DTYPES = {
    "dlt": np.dtype("float32"),
    "robust": np.dtype("float32"),
    "dlt_valid": np.dtype("bool"),
    "robust_valid": np.dtype("bool"),
    "gate": np.dtype("bool"),
    "reprojection": np.dtype("float32"),
    "interpolated": np.dtype("bool"),
    "fallback": np.dtype("bool"),
}
COORDINATE_NOTE = {
    "unity": (
        "Calibrated metric-world DLT and the raw target are independently "
        "body-canonicalized by the archived batch-first-frame procedure."
    ),
    "ski": (
        "Ski pseudo-GT is a GT-3D-fitted camera-local constructed reference; "
        "its source labels are root-relative. Before archived canonicalization, "
        "each calibrated-world DLT frame is translated by its own predicted "
        "pelvis midpoint (common13 joints 4 and 5). No scale, ground-truth "
        "feature, test similarity, or fitted transform is used."
    ),
}


@dataclass(frozen=True)
class SplitInputs:
    target: np.ndarray
    dlt: np.ndarray
    robust_dlt: np.ndarray
    dlt_valid: np.ndarray
    robust_valid: np.ndarray
    reprojection_error_px: np.ndarray
    cache_manifest: dict[str, Any]
    geometry_manifest: dict[str, Any]
    cache_manifest_path: Path
    geometry_manifest_path: Path


def _validate_array_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
    nonnegative: bool = False,
) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing input array: {path}")
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"Input checksum mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"Input shape mismatch for {path}: {array.shape} != {expected_shape}"
        )
    if array.dtype != np.dtype(expected_dtype):
        raise RuntimeError(
            f"Input dtype mismatch for {path}: {array.dtype} != {expected_dtype}"
        )
    chunk_size = max(1, min(len(array), 4096))
    for start in range(0, len(array), chunk_size):
        values = np.asarray(array[start : start + chunk_size])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Input contains nonfinite values: {path}")
        if nonnegative and np.any(values < 0):
            raise RuntimeError(f"Input contains negative values: {path}")
    return array


def _require_new_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"Preserving existing result: {path}")


def _validate_plain_file(path: Path, expected_sha256: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing provenance input: {path}")
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"Input checksum mismatch: {path}")


def _load_verified_split(root: Path, dataset: str, split: str) -> SplitInputs:
    expected_shape = EXPECTED[dataset][split]
    cache_dir = root / "cache" / dataset / split
    geometry_dir = root / "geometry" / dataset / split
    cache_manifest_path = cache_dir / "manifest.json"
    geometry_manifest_path = geometry_dir / "manifest.json"
    if not cache_manifest_path.exists() or not geometry_manifest_path.exists():
        raise RuntimeError(f"Incomplete cache/geometry manifests for {dataset}/{split}")
    cache_manifest = json.loads(cache_manifest_path.read_text())
    geometry_manifest = json.loads(geometry_manifest_path.read_text())

    cache_expected = {
        "dataset": dataset,
        "split": split,
        "sample_count": expected_shape[0],
        "time_window": expected_shape[1],
        "num_joints": expected_shape[2],
        "representation": "raw_pre_batch_canonicalization",
    }
    for key, expected in cache_expected.items():
        if cache_manifest.get(key) != expected:
            raise RuntimeError(
                f"Cache manifest mismatch {dataset}/{split} {key}: "
                f"{cache_manifest.get(key)!r} != {expected!r}"
            )
    cache_files = cache_manifest.get("files", {})
    required_cache = {"left.npy", "right.npy", "target.npy", "frames.npy", "samples.jsonl"}
    if set(cache_files) != required_cache:
        raise RuntimeError(f"Unexpected cache file manifest for {dataset}/{split}")
    _validate_plain_file(
        cache_dir / "samples.jsonl", cache_files["samples.jsonl"]
    )
    if sha256(Path(cache_manifest["index"])) != cache_manifest["index_sha256"]:
        raise RuntimeError(f"Current cache index differs for {dataset}/{split}")
    pose_arrays = {}
    for name in ("left", "right", "target"):
        pose_arrays[name] = _validate_array_file(
            cache_dir / f"{name}.npy",
            expected_sha256=cache_files[f"{name}.npy"],
            expected_shape=expected_shape,
            expected_dtype=np.dtype("float32"),
        )
    _validate_array_file(
        cache_dir / "frames.npy",
        expected_sha256=cache_files["frames.npy"],
        expected_shape=expected_shape[:2],
        expected_dtype=np.dtype("int64"),
    )

    geometry_expected = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "shape": list(expected_shape),
        "samples": expected_shape[0],
        "frames_per_sample": expected_shape[1],
        "joints": expected_shape[2],
        "coordinate_space": "calibrated_world_m",
        "estimated_2d_only": True,
        "gt_2d_used": False,
        "gt_fitted_transform_used": False,
        "gate_threshold_px": 25.0,
    }
    for key, expected in geometry_expected.items():
        if geometry_manifest.get(key) != expected:
            raise RuntimeError(
                f"Geometry manifest mismatch {dataset}/{split} {key}: "
                f"{geometry_manifest.get(key)!r} != {expected!r}"
            )
    if geometry_manifest.get("cache_frames_sha256") != cache_files["frames.npy"]:
        raise RuntimeError(f"Geometry/cache frame hash mismatch for {dataset}/{split}")
    if geometry_manifest.get("cache_samples_sha256") != cache_files["samples.jsonl"]:
        raise RuntimeError(f"Geometry/cache sample hash mismatch for {dataset}/{split}")
    if sha256(Path(geometry_manifest["source_index"])) != geometry_manifest["source_index_sha256"]:
        raise RuntimeError(f"Current geometry index differs for {dataset}/{split}")
    files = geometry_manifest.get("files", {})
    hashes = geometry_manifest.get("output_sha256", {})
    if set(files) != set(GEOMETRY_DTYPES):
        raise RuntimeError(f"Unexpected geometry file manifest for {dataset}/{split}")
    expected_points_shape = expected_shape[:-1]
    geometry_arrays = {}
    for key, dtype in GEOMETRY_DTYPES.items():
        filename = files[key]
        if filename not in hashes:
            raise RuntimeError(f"Missing geometry output hash for {filename}")
        shape = expected_shape if key in {"dlt", "robust"} else expected_points_shape
        geometry_arrays[key] = _validate_array_file(
            geometry_dir / filename,
            expected_sha256=hashes[filename],
            expected_shape=shape,
            expected_dtype=dtype,
            nonnegative=key == "reprojection",
        )
    points = int(np.prod(expected_points_shape))
    counts = geometry_manifest.get("counts", {})
    if counts.get("points") != points:
        raise RuntimeError(f"Geometry point count mismatch for {dataset}/{split}")
    if counts.get("dlt_valid") != points or counts.get("robust_valid") != points:
        raise RuntimeError(
            f"Geometry contains invalid points for {dataset}/{split}; no samples may be dropped"
        )
    if not np.asarray(geometry_arrays["dlt_valid"]).all():
        raise RuntimeError(f"DLT validity mask contains false points for {dataset}/{split}")
    if not np.asarray(geometry_arrays["robust_valid"]).all():
        raise RuntimeError(f"Robust DLT validity mask contains false points for {dataset}/{split}")
    if dataset == "ski" and "no GT-free exact pre-canonical map" not in str(
        geometry_manifest.get("ski_limitation")
    ):
        raise RuntimeError("Ski coordinate limitation is missing")

    print(
        json.dumps(
            {
                "status": "verified_inputs",
                "dataset": dataset,
                "split": split,
                "samples": expected_shape[0],
                "cache_manifest_sha256": sha256(cache_manifest_path),
                "geometry_manifest_sha256": sha256(geometry_manifest_path),
            }
        ),
        flush=True,
    )
    return SplitInputs(
        target=pose_arrays["target"],
        dlt=geometry_arrays["dlt"],
        robust_dlt=geometry_arrays["robust"],
        dlt_valid=geometry_arrays["dlt_valid"],
        robust_valid=geometry_arrays["robust_valid"],
        reprojection_error_px=geometry_arrays["reprojection"],
        cache_manifest=cache_manifest,
        geometry_manifest=geometry_manifest,
        cache_manifest_path=cache_manifest_path,
        geometry_manifest_path=geometry_manifest_path,
    )


def _tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.array(values, copy=True)).to(device)


def _prediction_reference(prediction: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset == "ski":
        return ski_predicted_pelvis_relative(prediction)
    return prediction


def _metric_dict(dataset: str) -> dict[str, MetricAccumulator]:
    names = ("all15", "common13") if dataset == "unity" else ("common13",)
    return {name: MetricAccumulator() for name in names}


def _update_metrics(
    metrics: dict[str, MetricAccumulator],
    prediction: torch.Tensor,
    target: torch.Tensor,
    dataset: str,
) -> None:
    for subset, accumulator in metrics.items():
        indices = slice(2, None) if dataset == "unity" and subset == "common13" else slice(None)
        accumulator.update(prediction[:, :, indices], target[:, :, indices])


@torch.no_grad()
def evaluate(
    inputs: SplitInputs,
    *,
    dataset: str,
    batch_size: int,
    device: str,
    method: str,
    model: DLTResidualMLP | None = None,
) -> dict[str, dict[str, float | int]]:
    if method not in {"dlt", "robust_dlt", "dlt_mlp"}:
        raise ValueError(method)
    if method == "dlt_mlp" and model is None:
        raise ValueError("dlt_mlp evaluation requires a model")
    if model is not None:
        model.eval()
    metrics = _metric_dict(dataset)
    count = len(inputs.target)
    for start, stop in batch_slices(count, batch_size):
        raw_name = "robust_dlt" if method == "robust_dlt" else "dlt"
        raw_prediction = _tensor(getattr(inputs, raw_name)[start:stop], device)
        raw_prediction = _prediction_reference(raw_prediction, dataset)
        raw_target = _tensor(inputs.target[start:stop], device)
        canonical_prediction, _, canonical_target = canonical_batch(
            raw_prediction, raw_prediction, raw_target, dataset=dataset
        )
        if method == "dlt_mlp":
            reprojection = _tensor(inputs.reprojection_error_px[start:stop], device)
            validity = _tensor(inputs.dlt_valid[start:stop], device)
            canonical_prediction = model(canonical_prediction, reprojection, validity)
        _update_metrics(metrics, canonical_prediction, canonical_target, dataset)
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def _input_provenance(inputs: SplitInputs) -> dict[str, Any]:
    geometry_files = inputs.geometry_manifest["files"]
    geometry_hashes = inputs.geometry_manifest["output_sha256"]
    return {
        "cache_manifest": str(inputs.cache_manifest_path),
        "cache_manifest_sha256": sha256(inputs.cache_manifest_path),
        "cache_files": inputs.cache_manifest["files"],
        "geometry_manifest": str(inputs.geometry_manifest_path),
        "geometry_manifest_sha256": sha256(inputs.geometry_manifest_path),
        "geometry_files": {
            key: {
                "filename": filename,
                "sha256": geometry_hashes[filename],
            }
            for key, filename in geometry_files.items()
        },
    }


def _provenance(
    args: argparse.Namespace,
    inputs: SplitInputs,
    *,
    runner_hash: str,
    model_hash: str,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "test_batch_size": 256 if args.dataset == "unity" else 4,
        "time_window": 30,
        "drop_last": False,
        "input_group": "estimated 2D + supplied calibration",
        "inputs": _input_provenance(inputs),
        "geometry_manifest": inputs.geometry_manifest,
        "coordinate_note": COORDINATE_NOTE[args.dataset],
        "ski_prediction_root_centered": args.dataset == "ski",
        "reference_root": (
            "midpoint_common13_joints_4_5_per_frame"
            if args.dataset == "ski"
            else None
        ),
        "root_centering_source_evidence": (
            "Ski-PosePTZ labels H5 original hip joint 0 and midpoint of "
            "original joints 1 and 4 are exactly zero across train and test."
            if args.dataset == "ski"
            else None
        ),
        "canonicalization": (
            "geometry and target independently; after forming the complete "
            "protocol batch; archived first-sample first-frame batch anchor"
        ),
        "gt_feature_inputs": False,
        "oracle_conversion": False,
        "test_fitted_transform": False,
        "runner_sha256": runner_hash,
        "model_code_sha256": model_hash,
        "torch_version": torch.__version__,
        "command": list(sys.argv),
    }


def run_reference_evaluation(args: argparse.Namespace) -> None:
    output_dir = args.root / "results" / args.dataset
    paths = {
        "dlt": output_dir / "dlt.json",
        "robust_dlt": output_dir / "robust_dlt.json",
    }
    _require_new_paths(paths.values())
    inputs = _load_verified_split(args.root, args.dataset, "test")
    runner_hash = sha256(__file__)
    model_hash = sha256(ROOT / "dual2pose/models/dlt_residual_baseline.py")
    batch_size = 256 if args.dataset == "unity" else 4
    records = {}
    for method in ("dlt", "robust_dlt"):
        started = time.monotonic()
        metrics = evaluate(
            inputs,
            dataset=args.dataset,
            batch_size=batch_size,
            device=args.device,
            method=method,
        )
        records[method] = {
            "method": method,
            "metrics": metrics,
            "provenance": _provenance(
                args, inputs, runner_hash=runner_hash, model_hash=model_hash
            ),
            "training": None,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "elapsed_seconds": time.monotonic() - started,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, record in records.items():
        write_json(paths[method], record)
        print(json.dumps(record, allow_nan=False), flush=True)


def _checkpoint_payload(
    model: DLTResidualMLP,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best: float,
    signature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_mpjpe": best,
        "signature": signature,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_rng(payload: dict[str, Any]) -> None:
    required = {
        "torch_rng_state",
        "cuda_rng_state",
        "python_rng_state",
        "numpy_rng_state",
    }
    if not required.issubset(payload):
        raise RuntimeError("Checkpoint lacks strict RNG resume state")
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload["cuda_rng_state"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])


def _signature(
    args: argparse.Namespace,
    train_inputs: SplitInputs,
    val_inputs: SplitInputs | None,
    model: DLTResidualMLP,
    *,
    train_batch_size: int,
    microbatch_size: int,
    runner_hash: str,
    model_hash: str,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "method": "dlt_mlp",
        "seed": args.seed,
        "epochs": args.epochs,
        "training_batch_size": train_batch_size,
        "microbatch_size": microbatch_size,
        "train_drop_last": True,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "loss": "L1 position + 0.01 predicted acceleration norm",
        "validation_batch_size": 4096 if args.dataset == "unity" else None,
        "test_batch_size": 256 if args.dataset == "unity" else 4,
        "selection": "minimum validation all15 MPJPE" if args.dataset == "unity" else "fixed final epoch 99",
        "canonicalization": "complete shuffled batch before any microbatch",
        "shuffle": "torch.randperm on CPU with seed + epoch",
        "train_inputs": _input_provenance(train_inputs),
        "validation_inputs": _input_provenance(val_inputs) if val_inputs else None,
        "runner_sha256": runner_hash,
        "model_code_sha256": model_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "float32; TF32 matmul disabled",
        "architecture": {
            "features": "flattened canonical DLT 3J + log1p(reprojection_px/25) J + DLT validity J",
            "hidden_layers": [256, 256],
            "activation": "ReLU",
            "dropout": 0.1,
            "output": "3J residual over canonical DLT",
        },
        "geometry_preprocessing": (
            "per-frame predicted pelvis midpoint (common13 joints 4 and 5) "
            "subtracted before archived canonicalization"
            if args.dataset == "ski"
            else "none"
        ),
    }


def _load_training_batch(
    inputs: SplitInputs,
    indices: np.ndarray,
    *,
    dataset: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_dlt = _tensor(inputs.dlt[indices], device)
    raw_dlt = _prediction_reference(raw_dlt, dataset)
    raw_target = _tensor(inputs.target[indices], device)
    canonical_dlt, _, canonical_target = canonical_batch(
        raw_dlt, raw_dlt, raw_target, dataset=dataset
    )
    reprojection = _tensor(inputs.reprojection_error_px[indices], device)
    validity = _tensor(inputs.dlt_valid[indices], device)
    return canonical_dlt, reprojection, validity, canonical_target


def smoke(args: argparse.Namespace) -> None:
    inputs = _load_verified_split(args.root, args.dataset, "train")
    count = min(args.smoke_samples, len(inputs.target))
    model = DLTResidualMLP(EXPECTED[args.dataset]["train"][2]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    dlt, reprojection, validity, target = _load_training_batch(
        inputs, np.arange(count), dataset=args.dataset, device=args.device
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(dlt, reprojection, validity)
    loss = dlt_residual_loss(prediction, target)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite actual-batch smoke loss")
    print(
        json.dumps(
            {
                "status": "actual_batch_smoke_passed",
                "dataset": args.dataset,
                "samples": count,
                "loss": loss.item(),
                "prediction_shape": list(prediction.shape),
            }
        ),
        flush=True,
    )


def train(args: argparse.Namespace) -> None:
    result_path = args.root / "results" / args.dataset / "dlt_mlp.json"
    _require_new_paths([result_path])
    train_inputs = _load_verified_split(args.root, args.dataset, "train")
    val_inputs = (
        _load_verified_split(args.root, args.dataset, "val")
        if args.dataset == "unity"
        else None
    )
    train_batch_size = 4096 if args.dataset == "unity" else 4
    microbatch_size = args.microbatch or train_batch_size
    if microbatch_size <= 0 or microbatch_size > train_batch_size:
        raise ValueError("microbatch must be in (0, training_batch_size]")
    runner_hash = sha256(__file__)
    model_source = ROOT / "dual2pose/models/dlt_residual_baseline.py"
    model_hash = sha256(model_source)
    num_joints = EXPECTED[args.dataset]["train"][2]
    model = DLTResidualMLP(num_joints=num_joints).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    signature = _signature(
        args,
        train_inputs,
        val_inputs,
        model,
        train_batch_size=train_batch_size,
        microbatch_size=microbatch_size,
        runner_hash=runner_hash,
        model_hash=model_hash,
    )
    training_name = "dlt_mlp_root_relative" if args.dataset == "ski" else "dlt_mlp"
    output = args.root / "training" / args.dataset / training_name
    config_path = output / "config.json"
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    history_path = output / "history.jsonl"
    first_epoch = 0
    best = float("inf")
    if last_path.exists():
        if not config_path.exists():
            raise RuntimeError("Incomplete run has checkpoint without config")
        saved_config = json.loads(config_path.read_text())
        if saved_config.get("signature") != signature:
            raise RuntimeError("Resume configuration/cache/code mismatch")
        saved = torch.load(last_path, map_location=args.device, weights_only=False)
        if saved.get("signature") != signature:
            raise RuntimeError("Checkpoint resume signature mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch = int(saved["epoch"]) + 1
        best = float(saved["best_validation_mpjpe"])
        _restore_rng(saved)
        if history_path.exists():
            lines = [line for line in history_path.read_text().splitlines() if line]
            if not lines or json.loads(lines[-1])["epoch"] != saved["epoch"]:
                raise RuntimeError("History/checkpoint epoch mismatch; refusing incomplete resume")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Incomplete run without resumable last.pt: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            config_path,
            {
                "signature": signature,
                "device": args.device,
                "command": list(sys.argv),
            },
        )

    started = time.monotonic()
    with history_path.open("a") as history:
        for epoch in range(first_epoch, args.epochs):
            epoch_started = time.monotonic()
            model.train()
            generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch)
            order = torch.randperm(len(train_inputs.target), generator=generator).numpy()
            trained = 0
            loss_sum = 0.0
            for start, stop in batch_slices(
                len(order), train_batch_size, drop_last=True
            ):
                indices = order[start:stop]
                dlt, reprojection, validity, target = _load_training_batch(
                    train_inputs,
                    indices,
                    dataset=args.dataset,
                    device=args.device,
                )
                optimizer.zero_grad(set_to_none=True)
                for micro_start, micro_stop in batch_slices(
                    len(dlt), microbatch_size
                ):
                    prediction = model(
                        dlt[micro_start:micro_stop],
                        reprojection[micro_start:micro_stop],
                        validity[micro_start:micro_stop],
                    )
                    loss = dlt_residual_loss(
                        prediction, target[micro_start:micro_stop]
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
                    (loss * ((micro_stop - micro_start) / len(dlt))).backward()
                    loss_sum += loss.item() * (micro_stop - micro_start)
                optimizer.step()
                trained += len(dlt)
            expected_trained = 192512 if args.dataset == "unity" else 144
            if trained != expected_trained:
                raise RuntimeError(
                    f"Training coverage mismatch: {trained} != {expected_trained}"
                )
            validation = (
                evaluate(
                    val_inputs,
                    dataset=args.dataset,
                    batch_size=4096,
                    device=args.device,
                    method="dlt_mlp",
                    model=model,
                )
                if val_inputs is not None
                else None
            )
            value = validation["all15"]["mpjpe"] if validation else None
            improved = value is not None and value < best
            if improved:
                best = float(value)
            payload = _checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                best=best,
                signature=signature,
            )
            if improved:
                _save_checkpoint(best_path, payload)
            _save_checkpoint(last_path, payload)
            entry = {
                "epoch": epoch,
                "training_samples": trained,
                "training_loss": loss_sum / trained,
                "validation": validation,
                "best_validation_mpjpe": best if validation else None,
                "epoch_seconds": time.monotonic() - epoch_started,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.write(json.dumps(entry, allow_nan=False) + "\n")
            history.flush()
            write_json(
                output / "progress.json",
                {"pid": os.getpid(), "status": "training", **entry},
            )
            print(json.dumps(entry, allow_nan=False), flush=True)

    selected = best_path if args.dataset == "unity" else last_path
    if not selected.exists():
        raise RuntimeError(f"Selected checkpoint is missing: {selected}")
    checkpoint = torch.load(selected, map_location=args.device, weights_only=False)
    if checkpoint.get("signature") != signature:
        raise RuntimeError("Selected checkpoint signature mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_inputs = _load_verified_split(args.root, args.dataset, "test")
    metrics = evaluate(
        test_inputs,
        dataset=args.dataset,
        batch_size=256 if args.dataset == "unity" else 4,
        device=args.device,
        method="dlt_mlp",
        model=model,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "method": "dlt_mlp",
        "metrics": metrics,
        "provenance": _provenance(
            args, test_inputs, runner_hash=runner_hash, model_hash=model_hash
        ),
        "training": {
            **signature,
            "command": list(sys.argv),
        },
        "selected_epoch": int(checkpoint["epoch"]),
        "checkpoint": str(selected),
        "checkpoint_sha256": sha256(selected),
        "total_seconds": time.monotonic() - started,
    }
    write_json(result_path, record)
    write_json(
        output / "progress.json",
        {"pid": os.getpid(), "status": "complete", "result": str(result_path)},
    )
    print(json.dumps(record, allow_nan=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--mode", choices=("evaluate", "train", "smoke"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--smoke-samples", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    if args.mode == "train" and args.epochs != 100:
        raise ValueError("The frozen protocol requires exactly 100 training epochs")
    if args.seed != 42:
        raise ValueError("The frozen protocol requires seed 42")
    if args.smoke_samples <= 0:
        raise ValueError("smoke-samples must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == "evaluate":
        run_reference_evaluation(args)
    elif args.mode == "train":
        train(args)
    else:
        smoke(args)


if __name__ == "__main__":
    main()
