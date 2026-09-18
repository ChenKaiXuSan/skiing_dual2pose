"""Frozen-checkpoint gate interventions; not retrained architecture ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dual2pose.eval.main_baseline_data import (
    ROOT, batch_slices, canonical_batch, load_cache, sha256, write_json,
)
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator
from dual2pose.experiments.run_main_baselines import CHECKPOINTS, ReferenceModel

CONDITIONS = ("clean", "left_zero", "right_zero",
              "switch_left_to_right", "switch_right_to_left")


def corrupt_pair(left, right, condition):
    """Zero raw poses before canonicalization; never modify cached inputs."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    left, right = left.clone(), right.clone()
    midpoint = left.shape[1] // 2
    if condition == "left_zero":
        left.zero_()
    elif condition == "right_zero":
        right.zero_()
    elif condition == "switch_left_to_right":
        left[:, :midpoint] = 0
        right[:, midpoint:] = 0
    elif condition == "switch_right_to_left":
        right[:, :midpoint] = 0
        left[:, midpoint:] = 0
    return left, right


def intervention_predictions(aux):
    result = {}
    for gate, alpha in (("dynamic", aux["alpha"]), ("mean", .5), ("left", 1.), ("right", 0.)):
        base = alpha * aux["base_l"] + (1 - alpha) * aux["base_r"]
        result[f"{gate}_residual"] = base + aux["output_residual"]
        result[f"{gate}_no_residual"] = base
    return result


@torch.no_grad()
def evaluate_condition(model, arrays, dataset, condition, batch_size, progress=None):
    model.eval()
    metrics = {}
    sums = np.zeros(6, dtype=np.float64)
    n, preference_hits, preference_count = 0, 0, 0
    histogram = np.zeros(20, dtype=np.int64)
    time_sum = None
    exemplars = None
    reconstruction_max_error = 0.
    for start, stop in batch_slices(len(arrays[0]), batch_size):
        raw_l, raw_r = corrupt_pair(arrays[0][start:stop], arrays[1][start:stop], condition)
        left, right, target = canonical_batch(raw_l, raw_r, arrays[2][start:stop], dataset=dataset)
        output, aux = model(left, right, return_components=True)
        predictions = intervention_predictions(aux)
        reconstruction_max_error = max(reconstruction_max_error,
            (output - predictions["dynamic_residual"]).abs().max().item())
        predictions["canonical_average"] = .5 * (left + right)
        for name, prediction in predictions.items():
            if name not in metrics:
                subsets = ("all15", "common13") if dataset == "unity" else ("common13",)
                metrics[name] = {key: PAMetricAccumulator() for key in subsets}
            for subset, accumulator in metrics[name].items():
                idx = slice(2, None) if dataset == "unity" and subset == "common13" else slice(None)
                accumulator.update(prediction[:, :, idx], target[:, :, idx])
        a = aux["alpha"].squeeze(-1).double()
        el = torch.linalg.vector_norm(aux["base_l"] - target, dim=-1).double()
        er = torch.linalg.vector_norm(aux["base_r"] - target, dim=-1).double()
        d = er - el  # Positive means the left candidate has lower reference error.
        sums += np.array([a.sum().item(), d.sum().item(), (a*a).sum().item(),
                          (d*d).sum().item(), (a*d).sum().item(), (el+er).sum().item()])
        n += a.numel()
        non_tied = (d.abs() > 1e-12) & ((a-.5).abs() > 1e-12)
        preference_hits += int(((a > .5) == (d > 0))[non_tied].sum())
        preference_count += int(non_tied.sum())
        histogram += torch.histc(a.float(), bins=20, min=0, max=1).long().cpu().numpy()
        batch_sum = a.sum(dim=0).cpu().numpy()
        time_sum = batch_sum if time_sum is None else time_sum + batch_sum
        if exemplars is None:
            # Fixed first two records, not selected by accuracy.
            exemplars = {key: aux[key][:2].cpu().tolist()
                         for key in ("alpha", "base_l", "base_r", "output_residual")}
            exemplars["sample_indices"] = [start+i for i in range(min(2, stop-start))]
        if progress is not None:
            progress(stop, len(arrays[0]))
    if n == 0:
        raise ValueError("Empty evaluation")
    covariance = sums[4] - sums[0] * sums[1] / n
    variance_a = max(0., sums[2] - sums[0] ** 2 / n)
    variance_d = max(0., sums[3] - sums[1] ** 2 / n)
    denom = np.sqrt(variance_a * variance_d)
    return {
        "condition": condition,
        "metrics": {name: {key: acc.result() for key, acc in subsets.items()}
                    for name, subsets in metrics.items()},
        "gate_diagnostics": {
            "joint_frame_count": n, "alpha_mean": sums[0] / n,
            "alpha_std_joint_frames": np.sqrt(variance_a / n),
            "candidate_error_correlation": covariance / denom if denom > 0 else None,
            "candidate_preference_accuracy": preference_hits / preference_count if preference_count else None,
            "non_tied_preference_count": preference_count,
            "histogram_edges": np.linspace(0, 1, 21).tolist(),
            "histogram_counts": histogram.tolist(),
            "alpha_mean_by_time_joint": (time_sum / len(arrays[0])).tolist(),
            "candidate_definition": "actual base_l/base_r in the model output coordinates, not raw views",
            "target_use": "diagnostics and scoring only",
        },
        "default_reconstruction_max_abs_error": reconstruction_max_error,
        "fixed_first_two_exemplars": exemplars,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    arrays, manifest = load_cache(args.cache_root / args.dataset / args.split, args.device)
    if args.limit:
        arrays = tuple(a[:args.limit] for a in arrays)
    model = ReferenceModel(CHECKPOINTS[args.dataset]).model.to(args.device).eval()
    full_test = args.split == "test" and args.limit == 0
    config = {
        "status": "running", "experiment": "frozen checkpoint intervention, not retraining",
        "dataset": args.dataset, "split": args.split, "limit": args.limit,
        "full_test": full_test, "samples": len(arrays[0]), "cache_manifest": manifest,
        "checkpoint_sha256": sha256(CHECKPOINTS[args.dataset]),
        "source_sha256": {str(p): sha256(p) for p in (Path(__file__), ROOT/"dual2pose/models/crossview_fusion.py")},
        "corruption": "complete raw-pose zeroing; switch at floor(T/2); reference untouched",
        "anchor_warning": "corruption may also change batch-anchor landmarks",
    }
    write_json(args.output / "config.json", config)
    started = time.monotonic()
    for condition in CONDITIONS:
        def progress(done, total):
            write_json(args.output / "progress.json",
                       dict(status="running", condition=condition, completed=done,
                            total=total, elapsed_seconds=time.monotonic()-started))
        result = evaluate_condition(model, arrays, args.dataset, condition,
                                    256 if args.dataset == "unity" else 4, progress)
        if condition == "clean" and full_test:
            reference = ROOT/"logs/ivc_mmsports_extension/main_baselines/20260906/results"/args.dataset/"canonfuse3d.json"
            old = json.loads(reference.read_text())["metrics"]
            for subset, values in old.items():
                for key in ("mpjpe", "acceleration_error"):
                    delta = abs(result["metrics"]["dynamic_residual"][subset][key] - values[key])
                    if delta > 5e-7:
                        raise RuntimeError(f"Archived clean control mismatch: {subset}/{key}: {delta}")
        write_json(args.output / f"{condition}.json", result)
        print(json.dumps({"condition": condition, "metrics": result["metrics"]["dynamic_residual"]}), flush=True)
    write_json(args.output / "progress.json",
               dict(status="complete_full_test" if full_test else "complete_smoke",
                    conditions=list(CONDITIONS), samples=len(arrays[0]),
                    elapsed_seconds=time.monotonic()-started))


if __name__ == "__main__":
    main()
