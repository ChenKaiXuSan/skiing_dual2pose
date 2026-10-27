#!/usr/bin/env python3
"""Lightweight adaptation diagnostic on the hand-held dual-Pocket recording.

Mirrors the front-end adaptation protocol of the manuscript (frozen checkpoint,
20 epochs, heads-only or full-parameter update) but uses a temporal split of the
single chen_4 recording: the model is adapted on one block of windows and tested on
the disjoint remaining block, in both directions.  Same subject and session, so this
is a deployment-calibration demonstration, not a generalization test.

Example::

    python -m dual2pose.eval.adapt_pocket3 --gpu 1 --output-dir logs/eval_pocket3_chen4/adaptation
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dual2pose.eval import eval_pocket3 as ep
from dual2pose.frontend_adaptation import configure_trainable_scope
from dual2pose.models.main_baselines import aligned_average
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


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
    li = np.stack([np.arange(s, s + args.window) for s in starts])
    return L[li], R[j[li]], gl["joints"][li], gl["reliable"][li], starts


def canon(x, device):
    c, _ = canonicalize_pose_torch(torch.from_numpy(x).to(device), left_hip=ep.LEFT_HIP, right_hip=ep.RIGHT_HIP, neck=ep.NECK)
    return c


def evaluate(model, L, R, G, REL, device):
    """Per-frame body-frame MPJPE / PA-MPJPE (common 13, mm) for averaging, frozen/adapted model and gate-only."""
    outs = {"canonical_average": [], "aligned_average": [], "model": [], "gate_only": [], "gt": []}
    model.eval()
    with torch.no_grad():
        for w in range(L.shape[0]):
            lc, rc, gc = canon(L[w:w + 1], device), canon(R[w:w + 1], device), canon(G[w:w + 1], device)
            fused, aux = model.models(lc, rc, return_components=True)
            outs["canonical_average"].append((0.5 * (lc + rc)).cpu().numpy())
            outs["aligned_average"].append(aligned_average(lc, rc).cpu().numpy())
            outs["model"].append(fused.cpu().numpy())
            outs["gate_only"].append((fused - aux["output_residual"]).cpu().numpy())
            outs["gt"].append(gc.cpu().numpy())
    outs = {k: np.concatenate(v) for k, v in outs.items()}
    pf = {k: ep.per_frame_canonical(v) for k, v in outs.items()}
    res = {}
    for k in ("canonical_average", "aligned_average", "model", "gate_only"):
        res[k] = {
            "per_frame_canonical_mpjpe_mm": 1000 * ep.masked_mpjpe(pf[k], pf["gt"], REL, ep.COMMON13)[0],
            "pa_mpjpe_mm": 1000 * ep.masked_pa_mpjpe(outs[k], outs["gt"], REL, ep.COMMON13)[0],
            "per_frame_canonical_accel_mm": 1000 * ep.masked_accel_error(pf[k], pf["gt"], REL, ep.COMMON13)[0],
        }
    return res


def adapt(model, L, R, G, device, scope: str, epochs: int, lr: float, seed: int):
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
            batch = {"kpt3d_sam": {"cam1": torch.from_numpy(L[w:w + 1]).to(device),
                                   "cam2": torch.from_numpy(R[w:w + 1]).to(device)},
                     "kpt3d_gt": torch.from_numpy(G[w:w + 1]).to(device)}
            loss = model._shared_step(batch, stage="train")
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
    ap.add_argument("--output-dir", type=Path, default=ep.REPO_ROOT / "logs/eval_pocket3_chen4/adaptation")
    args = ap.parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    L, R, G, REL, starts = load_windows(args)
    W = L.shape[0]
    k = int(round(args.adapt_fraction * W))
    base = ep.load_model(args.ckpt, device)
    # disable Lightning logging side effects during manual training
    base.log = lambda *a, **kw: None
    results = {"windows": W, "splits": {}, "epochs": args.epochs, "lr": args.lr,
               "checkpoint": str(args.ckpt), "scopes": ["heads_only", "full"]}
    for name, (ai, ti) in {"adapt_first_test_last": (slice(0, k), slice(k, W)),
                           "adapt_last_test_first": (slice(W - k, W), slice(0, W - k))}.items():
        entry = {"adapt_windows": [int(s) for s in np.array(starts)[ai]], "test_windows": [int(s) for s in np.array(starts)[ti]],
                 "frozen": evaluate(base, L[ti], R[ti], G[ti], REL[ti], device)}
        for scope in ("heads_only", "full"):
            m = adapt(base, L[ai], R[ai], G[ai], device, scope, args.epochs, args.lr, args.seed)
            entry[f"adapted_{scope}"] = evaluate(m, L[ti], R[ti], G[ti], REL[ti], device)
        results["splits"][name] = entry
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=1))
    for name, e in results["splits"].items():
        print(f"--- {name}: adapt {len(e['adapt_windows'])} windows, test {len(e['test_windows'])} windows")
        for cond in ("frozen", "adapted_heads_only", "adapted_full"):
            r = e[cond]
            print(f"  {cond:14s} avg {r['canonical_average']['per_frame_canonical_mpjpe_mm']:5.1f}/{r['canonical_average']['pa_mpjpe_mm']:4.1f}  "
                  f"model {r['model']['per_frame_canonical_mpjpe_mm']:5.1f}/{r['model']['pa_mpjpe_mm']:4.1f}  "
                  f"gate-only {r['gate_only']['per_frame_canonical_mpjpe_mm']:5.1f}/{r['gate_only']['pa_mpjpe_mm']:4.1f}  (MPJPE/PA mm)")


if __name__ == "__main__":
    main()
