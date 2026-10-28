#!/usr/bin/env python3
"""Lightweight adaptation diagnostic on the hand-held dual-Pocket recording.

Mirrors the front-end adaptation protocol of the manuscript (frozen checkpoint,
20 epochs, heads-only or full-parameter update) but uses a temporal split of the
single chen_4 recording: the model is adapted on one block of windows and tested on
the disjoint remaining block, in both directions.  Same subject and session, so this
is a deployment-calibration demonstration, not a generalization test.

Example::

    python -m dual2pose.pocket3.adapt --gpu 1
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dual2pose.pocket3 import evaluate as ep
from dual2pose.frontend_adaptation import configure_trainable_scope
from dual2pose.models.main_baselines import aligned_average
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL


def load_windows(args):
    root = Path(args.dataset_root)
    gl = ep.load_gt(root / "gt/pocket_aligned" / f"{args.session}_pocket_1.npz")
    gr = ep.load_gt(root / "gt/pocket_aligned" / f"{args.session}_pocket_2.npz")
    sam = root / "sam3d_body_results" / args.session
    L, lv, lb = ep.load_sam3d_dir(sam / "pocket_1", len(gl["frame"]))
    R, rv, rb = ep.load_sam3d_dir(sam / "pocket_2", len(gr["frame"]))
    j, _ = ep.pair_by_utc(gl["utc"], gr["utc"])
    usable = (j >= 0) & ep.subject_in_view(lb, lv, args.frame_height) & np.where(j >= 0, ep.subject_in_view(rb, rv, args.frame_height)[np.clip(j, 0, None)], False)
    starts = ep.build_windows(usable, args.window, args.window, gl["reliable"][:, ep.ANCHOR_JOINTS].all(1))
    if not starts:
        raise ValueError("No valid complete Pocket3 windows")
    li = np.stack([np.arange(s, s + args.window) for s in starts])
    return L[li], R[j[li]], gl["joints"][li], gl["reliable"][li], starts


def canon(x, device, *, model=None):
    if model is not None:
        return model.canonicalize(ep.native_pose(model, torch.from_numpy(x).to(device)))[0]
    c, _ = canonicalize_pose_torch(torch.from_numpy(x).to(device), left_hip=ep.LEFT_HIP, right_hip=ep.RIGHT_HIP, neck=ep.NECK, mode="per_frame")
    return c


def evaluate(model, L, R, G, REL, device):
    """Direct common-13 MPJPE (mm), with separately reported aligned diagnostics.

    The averaging baselines, model and reference all use the per-frame body frame.
    """
    ep.validate_targets(G, REL)
    model.eval()
    raw = ep.run_methods(L, R, G, {"model": model}, device)
    outs = {"canonical_average": raw["canonical_average"], "aligned_average": raw["aligned_average"],
            "model": raw["canonfuse3d[model]"], "gate_only": raw["canonfuse3d_gate_only[model]"], "gt": raw["gt"]}
    ref = {k: "gt" for k in ("canonical_average", "aligned_average", "model", "gate_only")}
    pf = {k: ep.per_frame_canonical(v) for k, v in outs.items()}
    res = {}
    for k in ("canonical_average", "aligned_average", "model", "gate_only"):
        g = ref[k]
        res[k] = {
            "mpjpe_mm": 1000 * ep.masked_mpjpe(outs[k], outs[g], REL, ep.COMMON13)[0],
            "accel_mm": 1000 * ep.masked_accel_error(outs[k], outs[g], REL, ep.COMMON13)[0],
            "recanonicalized_mpjpe_mm": 1000 * ep.masked_mpjpe(pf[k], pf[g], REL, ep.COMMON13)[0],
            # Historical keys are auxiliary aliases, never selection criteria.
            "per_frame_canonical_mpjpe_mm": 1000 * ep.masked_mpjpe(pf[k], pf[g], REL, ep.COMMON13)[0],
            "pa_mpjpe_mm": 1000 * ep.masked_pa_mpjpe(outs[k], outs[g], REL, ep.COMMON13)[0],
            "per_frame_canonical_accel_mm": 1000 * ep.masked_accel_error(pf[k], pf[g], REL, ep.COMMON13)[0],
        }
    return res


def adapt(model, L, R, G, REL, device, scope: str, epochs: int, lr: float, seed: int):
    ep.validate_pocket_models({"model": model})
    ep.validate_targets(G, REL)
    if epochs <= 0 or lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    configure_trainable_scope(model, scope)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    n = L.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n)
        for w in order.tolist():
            batch = ep.training_batch(model, L[w:w + 1], R[w:w + 1], G[w:w + 1], REL[w:w + 1], device)
            loss = model._shared_step(batch, stage="train")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite Pocket3 adaptation loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=ep.DEFAULT_ROOT)
    ap.add_argument("--session", default="chen_4")
    ap.add_argument("--ckpt", type=Path, default=ep.DEFAULT_CKPT)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--frame-height", type=float, default=1080.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adapt-fraction", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=ep.default_output_dir("adaptation"))
    args = ap.parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    from dual2pose.pocket3.train import load_frames, windows_in, gather
    if not 0 < args.adapt_fraction < 1:
        raise ValueError("adapt-fraction must lie strictly between zero and one")
    L, R, j, gl, usable, anchor = load_frames(args)
    # Freeze frame blocks BEFORE deriving valid windows; availability cannot move the boundary.
    F = len(usable)
    cut = int(round(args.adapt_fraction * F))
    if not 0 < cut < F:
        raise ValueError("Adaptation and test require nonempty temporal blocks")
    blocks = {"adapt_first_test_last": ((0, cut), (cut, F)),
              "adapt_last_test_first": ((F - cut, F), (0, F - cut))}
    base = ep.load_model(args.ckpt, device)
    # disable Lightning logging side effects during manual training
    base.log = lambda *a, **kw: None
    results = {"source_frames": F, "splits": {}, "epochs": args.epochs, "lr": args.lr,
               "evaluation_protocol": dict(EVALUATION_PROTOCOL), "comparison_frame": dict(ep.COMPARISON_FRAME), "primary_metric": "mpjpe_mm",
               "checkpoint": str(args.ckpt), "canonical_mode": base.canonical_mode, "scopes": ["heads_only", "full"]}
    for name, (adapt_block, test_block) in blocks.items():
        ai = windows_in(usable, anchor, *adapt_block, args.window, args.window)
        ti = windows_in(usable, anchor, *test_block, args.window, args.window)
        tr = gather(L, R, j, gl, ai, args.window)
        te = gather(L, R, j, gl, ti, args.window)
        entry = {"adapt_windows": ai, "test_windows": ti,
                 "blocks": {"adapt": list(adapt_block), "test": list(test_block)},
                 "selection": "fixed_final_epoch", "selected_epoch": args.epochs,
                 "frozen": evaluate(base, *te, device)}
        for scope in ("heads_only", "full"):
            m = adapt(base, *tr, device, scope, args.epochs, args.lr, args.seed)
            entry[f"adapted_{scope}"] = evaluate(m, *te, device)
        results["splits"][name] = entry
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=1))
    for name, e in results["splits"].items():
        print(f"--- {name}: adapt {len(e['adapt_windows'])} windows, test {len(e['test_windows'])} windows")
        for cond in ("frozen", "adapted_heads_only", "adapted_full"):
            r = e[cond]
            print(f"  {cond:14s} avg {r['canonical_average']['mpjpe_mm']:5.1f}/{r['canonical_average']['pa_mpjpe_mm']:4.1f}  "
                  f"model {r['model']['mpjpe_mm']:5.1f}/{r['model']['pa_mpjpe_mm']:4.1f}  "
                  f"gate-only {r['gate_only']['mpjpe_mm']:5.1f}/{r['gate_only']['pa_mpjpe_mm']:4.1f}  (common-13 direct MPJPE / auxiliary PA, mm)")


if __name__ == "__main__":
    main()
