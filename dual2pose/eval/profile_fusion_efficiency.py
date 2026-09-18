"""Matched local fusion timing, parameter size, and explicitly partial FLOPs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from dual2pose.eval.main_baseline_data import ROOT, sha256, write_json
from dual2pose.experiments.run_main_baselines import ReferenceModel, CHECKPOINTS
from dual2pose.models.main_baselines import build_baseline


def parameter_summary(model):
    parameters = list(model.parameters())
    return dict(total_parameters=sum(p.numel() for p in parameters),
                trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
                weight_bytes=sum(p.numel() * p.element_size() for p in parameters))


@torch.no_grad()
def measure(model, left, right, warmup=20, repeats=100):
    if warmup < 1 or repeats < 2:
        raise ValueError("Use at least one warmup and two measurements")
    model.eval()
    is_cuda = left.is_cuda
    def sync():
        if is_cuda:
            torch.cuda.synchronize(left.device)
    for _ in range(warmup):
        model(left, right)
    sync()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(left.device)
        initial_memory = torch.cuda.memory_allocated(left.device)
    samples = []
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        model(left, right)
        sync()
        samples.append(1000 * (time.perf_counter() - started))
    peak = torch.cuda.max_memory_allocated(left.device) - initial_memory if is_cuda else None
    # Operator-dispatch counter covers Conv1d, unlike profiler event FLOPs.
    with FlopCounterMode(display=False) as counter:
        model(left, right)
        sync()
    counted = {str(op): int(value) for op, value in counter.get_flop_counts().get("Global", {}).items()}
    return {
        **parameter_summary(model), "input_shape": list(left.shape), "dtype": str(left.dtype),
        "latency_ms_samples": samples, "latency_ms_median": float(np.median(samples)),
        "latency_ms_mean": float(np.mean(samples)), "latency_ms_std": float(np.std(samples)),
        "warmup": warmup, "repeats": repeats, "peak_allocated_bytes": peak,
        "memory_scope": "incremental peak tensor allocation above pre-forward model and input allocation",
        "counted_flops": sum(counted.values()), "counted_flops_by_operator": counted,
        "matrix_convolution_macs": sum(counted.values()) / 2,
        "flops_complete": False,
        "excluded_arithmetic": ["SVD/determinant", "normalization", "nonlinearities", "elementwise operations"],
        "flop_scope": "registered matrix/convolution operations; 2 FLOPs per multiply-accumulate; not all pipeline arithmetic",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    inputs = tuple(torch.randn(1, 30, 15, 3, device=args.device) for _ in range(2))
    results = {}
    for name in ("canonfuse3d", "tcn", "smoothnet"):
        if name == "canonfuse3d":
            checkpoint = CHECKPOINTS["unity"]
            model = ReferenceModel(checkpoint).to(args.device)
        else:
            record = json.loads((ROOT/"logs/ivc_mmsports_extension/main_baselines/20260906/results/unity"/f"{name}.json").read_text())
            checkpoint = Path(record["checkpoint"])
            if sha256(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint mismatch: {checkpoint}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_baseline(name, num_joints=15, time_window=30).to(args.device)
            model.load_state_dict(saved["state_dict"], strict=True)
        results[name] = measure(model, *inputs)
        results[name]["checkpoint_sha256"] = sha256(checkpoint)
        print(name, results[name]["total_parameters"], results[name]["latency_ms_median"], flush=True)
        del model
    write_json(args.output, dict(
        status="complete_local_models_partial_flops", results=results,
        device=args.device, gpu=torch.cuda.get_device_name(args.device) if inputs[0].is_cuda else None,
        torch_version=torch.__version__, tf32=False, cpu_threads=4,
        source_sha256=sha256(__file__),
        timing_scope="canonical inputs to output; includes internal Sim3 for CanonFuse3D; excludes upstream, raw canonicalization, I/O",
        limitation="synthetic fixed-shape timing inputs; external-method timing and unsupported arithmetic not yet covered",
    ))


if __name__ == "__main__":
    main()
