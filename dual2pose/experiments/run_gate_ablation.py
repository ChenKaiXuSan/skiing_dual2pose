"""Matched single-seed Unity gate/output-residual ablations in a new namespace."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion
from dual2pose.eval.main_baseline_data import ROOT, batch_slices, canonical_batch, load_cache, sha256, write_json
from dual2pose.experiments.run_main_baselines import MetricAccumulator, _save_checkpoint
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator

TRAIN_VARIANTS = {
    "full": dict(gate_mode="dynamic", disable_output_residual=False),
    "fixed_gate": dict(gate_mode="mean", disable_output_residual=False),
    "no_output_residual": dict(gate_mode="dynamic", disable_output_residual=True),
}


def build_model(variant):
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Not a trainable variant: {variant}")
    return CrossViewCanonicalFusion(num_heads=4, **TRAIN_VARIANTS[variant])


def fusion_objective(prediction, alpha, left, right, target):
    delta = left - right
    view = .5 * (F.l1_loss(prediction + (1-alpha)*delta, left) +
                 F.l1_loss(prediction - alpha*delta, right))
    balance = (alpha.mean() - .5).abs()
    entropy = -(alpha * torch.log(alpha.clamp(min=1e-6)) +
                (1-alpha) * torch.log((1-alpha).clamp(min=1e-6))).mean()
    acceleration = prediction[:, 2:] - 2*prediction[:, 1:-1] + prediction[:, :-2]
    return (F.l1_loss(prediction, target) + .05*view + .02*balance -
            .005*entropy + .01*torch.linalg.vector_norm(acceleration, dim=-1).mean())


@torch.no_grad()
def evaluate(model, arrays, batch_size=4096, with_pa=False):
    model.eval()
    names = ("all15", "common13") if with_pa else ("all15",)
    metrics = {name: PAMetricAccumulator() if with_pa else MetricAccumulator() for name in names}
    for start, stop in batch_slices(len(arrays[0]), batch_size):
        left, right, target = canonical_batch(*(a[start:stop] for a in arrays), dataset="unity")
        prediction, _ = model(left, right)
        for name, accumulator in metrics.items():
            idx = slice(2, None) if name == "common13" else slice(None)
            accumulator.update(prediction[:, :, idx], target[:, :, idx])
    return {key: value.result() for key, value in metrics.items()}


def run_variant(args, variant, train_arrays, val_arrays, train_manifest, val_manifest):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output / variant
    output.mkdir()
    model = build_model(variant).to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=.001, weight_decay=.01)
    config = dict(
        variant=variant, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        validation_batch_size=args.batch_size, evaluation_batch_size=256,
        train_drop_last=True, validation_drop_last=False,
        initialization="from scratch; same seed and complete initial parameter layout",
        loss="archived full fusion objective; fixed-gate regularizers remain constants",
        selection="minimum full-validation all15 MPJPE",
        optimizer="AdamW lr=0.001 weight_decay=0.01; no scheduler",
        dtype="float32; TF32 disabled", smoke_steps=args.smoke_steps,
        train_manifest=train_manifest, val_manifest=val_manifest,
        source_sha256={str(p): sha256(p) for p in (
            Path(__file__), ROOT/"dual2pose/models/crossview_fusion.py",
            ROOT/"dual2pose/eval/main_baseline_data.py")},
    )
    write_json(output/"config.json", config)
    started, best = time.monotonic(), float("inf")
    epochs = 1 if args.smoke_steps else args.epochs
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_arrays[0]), generator=generator).to(args.device)
        loss_sum, trained = 0., 0
        for step, (start, stop) in enumerate(batch_slices(len(order), args.batch_size, drop_last=True)):
            indices = order[start:stop]
            left, right, target = canonical_batch(*(a[indices] for a in train_arrays), dataset="unity")
            optimizer.zero_grad(set_to_none=True)
            prediction, aux = model(left, right)
            loss = fusion_objective(prediction, aux["alpha"], left, right, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {variant}/{epoch}/{step}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient: {variant}/{epoch}/{step}")
            optimizer.step()
            trained += stop-start
            loss_sum += loss.item() * (stop-start)
            if step % 10 == 0:
                write_json(output/"progress.json", dict(status="training", pid=os.getpid(),
                    epoch=epoch, completed_samples=trained, total_samples=len(order),
                    elapsed_seconds=time.monotonic()-started))
            if args.smoke_steps and step+1 >= args.smoke_steps:
                break
        if not trained:
            raise ValueError("No complete training batch")
        validation_arrays = tuple(a[:args.batch_size] for a in val_arrays) if args.smoke_steps else val_arrays
        validation = evaluate(model, validation_arrays, args.batch_size)
        score = validation["all15"]["mpjpe"]
        if score < best:
            best = score
            _save_checkpoint(output/"best.pt", model, optimizer, epoch, best, config)
        _save_checkpoint(output/"last.pt", model, optimizer, epoch, best, config)
        record = dict(epoch=epoch, train_samples=trained, loss=loss_sum/trained,
                      validation=validation, best_validation_mpjpe=best,
                      elapsed_seconds=time.monotonic()-started)
        with (output/"history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
        write_json(output/"progress.json", dict(status="epoch_complete", pid=os.getpid(), **record))
        print(json.dumps(dict(variant=variant, **record)), flush=True)
    if args.smoke_steps:
        write_json(output/"progress.json", dict(status="complete_train_smoke_not_admissible",
                  train_samples=trained, validated_samples=len(validation_arrays[0])))
        return
    # Test data are opened only after training and validation selection are over.
    saved = torch.load(output/"best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    test_arrays, test_manifest = load_cache(args.cache_root/"unity/test", args.device)
    metrics = evaluate(model, test_arrays, batch_size=256, with_pa=True)
    write_json(output/"metrics.json", dict(
        status="complete_full_test", variant=variant, metrics=metrics,
        selected_epoch=saved["epoch"], checkpoint_sha256=sha256(output/"best.pt"),
        test_manifest=test_manifest, configuration=config,
    ))
    write_json(output/"progress.json", dict(status="complete_full_test", selected_epoch=saved["epoch"],
        elapsed_seconds=time.monotonic()-started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["all", *TRAIN_VARIANTS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--smoke-steps", type=int, default=0)
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.smoke_steps < 0:
        p.error("Invalid budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        train_arrays, train_manifest = load_cache(args.cache_root/"unity/train", args.device)
        val_arrays, val_manifest = load_cache(args.cache_root/"unity/val", args.device)
        variants = list(TRAIN_VARIANTS) if args.variant == "all" else [args.variant]
        for variant in variants:
            write_json(args.output/"progress.json", dict(status="running", active_variant=variant,
                       variants=variants, pid=os.getpid(), smoke_steps=args.smoke_steps))
            run_variant(args, variant, train_arrays, val_arrays, train_manifest, val_manifest)
            torch.cuda.empty_cache()
        write_json(args.output/"progress.json", dict(
            status="complete_train_smoke_not_admissible" if args.smoke_steps else "complete_full_test",
            variants=variants,
            deterministic_fourth_cell="frozen mean_no_residual; no trainable output path"))
    except Exception as exc:
        write_json(args.output/"failure.json", dict(type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    main()
