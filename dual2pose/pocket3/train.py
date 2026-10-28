#!/usr/bin/env python3
"""Train CanonFuse3D on the chen_4 hand-held recording with a chronological train/val/test split.

Timeline (usable paired frames) is cut into three blocks: train (first ``--train-frac``), validation
(next ``--val-frac``), test (rest), and the mirrored split (test first).  Training windows are taken
with stride ``--train-stride`` inside the train block only; validation and test use non-overlapping
30-frame windows. Model selection uses validation common-13 MPJPE directly in the per-frame body frame; the test block is
evaluated once with the selected weights.  Two initializations: the frozen Unity checkpoint and random.

Example::

    python -m dual2pose.pocket3.train --gpu 1
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dual2pose.config import load_config
from dual2pose.pocket3 import adapt as ad
from dual2pose.pocket3 import evaluate as ep
from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL


def load_frames(args):
    root = Path(args.dataset_root)
    gl = ep.load_gt(root / "gt/pocket_aligned" / f"{args.session}_pocket_1.npz")
    gr = ep.load_gt(root / "gt/pocket_aligned" / f"{args.session}_pocket_2.npz")
    sam = root / "sam3d_body_results" / args.session
    L, lv, lb = ep.load_sam3d_dir(sam / "pocket_1", len(gl["frame"]))
    R, rv, rb = ep.load_sam3d_dir(sam / "pocket_2", len(gr["frame"]))
    j, _ = ep.pair_by_utc(gl["utc"], gr["utc"])
    usable = (j >= 0) & ep.subject_in_view(lb, lv, args.frame_height) & np.where(j >= 0, ep.subject_in_view(rb, rv, args.frame_height)[np.clip(j, 0, None)], False)
    anchor = gl["reliable"][:, ep.ANCHOR_JOINTS].all(1)
    return L, R, j, gl, usable, anchor


def windows_in(usable, anchor, lo, hi, window, stride):
    if not 0 <= lo < hi <= len(usable):
        raise ValueError("Temporal block must be nonempty and inside the recording")
    mask = np.zeros_like(usable); mask[lo:hi] = usable[lo:hi]
    return ep.build_windows(mask, window, stride, anchor)


def gather(L, R, j, gl, starts, window):
    if not starts:
        raise ValueError("Temporal block contains no valid complete windows")
    li = np.stack([np.arange(s, s + window) for s in starts])
    return L[li], R[j[li]], gl["joints"][li], gl["reliable"][li]


def fresh_model(device):
    cfg = load_config(ep.REPO_ROOT / "configs" / "dual2pose.yaml")
    cfg.model.backbone = "crossview_fusion"
    cfg.data.load_frames = False; cfg.data.load_2d_kpt = False; cfg.data.load_3d_kpt = True
    cfg.log_path = str(ep.default_output_dir("training"))
    return CrossViewFusionTrainer(cfg).to(device)


def save_weights(model, path: Path):
    """Export selected weights with the protocol and configuration needed to reload them."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": {"hparams": OmegaConf.to_container(model.hparams.hparams, resolve=True)},
    }
    model.on_save_checkpoint(checkpoint)
    checkpoint["pocket3_comparison_frame"] = dict(ep.COMPARISON_FRAME)
    torch.save(checkpoint, path)


def train(model, tr, va, device, epochs, lr, patience, seed):
    ep.validate_pocket_models({"model": model})
    ep.validate_targets(tr[2], tr[3]); ep.validate_targets(va[2], va[3])
    if epochs <= 0 or patience <= 0 or lr <= 0:
        raise ValueError("epochs, patience and learning rate must be positive")
    torch.manual_seed(seed); np.random.seed(seed)
    model = copy.deepcopy(model); model.log = lambda *a, **k: None
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Ltr, Rtr, Gtr, RELtr = tr
    best, best_state, best_epoch, bad, history = float("inf"), copy.deepcopy(model.state_dict()), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(Ltr.shape[0]); losses = []
        for w in order:
            batch = ep.training_batch(model, Ltr[w:w + 1], Rtr[w:w + 1], Gtr[w:w + 1], RELtr[w:w + 1], device)
            loss = model._shared_step(batch, stage="train")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite Pocket3 training loss")
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        val = ad.evaluate(model, *va, device)["model"]["mpjpe_mm"]
        if not np.isfinite(val):
            raise ValueError("Validation common13 MPJPE must be finite")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_common13_mpjpe_mm": val})
        if val < best - 1e-3:
            best, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), {"best_epoch": best_epoch, "best_val_mm": best, "epochs_run": epoch, "history": history,
                          "selection_metric": "mpjpe_mm", "evaluation_protocol": dict(EVALUATION_PROTOCOL)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=ep.DEFAULT_ROOT)
    ap.add_argument("--session", default="chen_4")
    ap.add_argument("--ckpt", type=Path, default=ep.DEFAULT_CKPT)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--frame-height", type=float, default=1080.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--lr-scratch", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=ep.default_output_dir("training"))
    ap.add_argument("--save-weights", action="store_true", help="save selected weights, configuration and protocol under <output-dir>/weights/")
    args = ap.parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    L, R, j, gl, usable, anchor = load_frames(args)
    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1 and args.train_frac + args.val_frac < 1):
        raise ValueError("train/val fractions must define three nonempty temporal blocks")
    idx = np.where(usable)[0]; n = len(idx)
    if n < 3:
        raise ValueError("Insufficient usable frames for train/val/test blocks")
    cut1, cut2 = idx[int(args.train_frac * n)], idx[int((args.train_frac + args.val_frac) * n)]
    F = len(usable)
    splits = {
        "train_first_test_last": {"train": (0, cut1), "val": (cut1, cut2), "test": (cut2, F)},
        "train_last_test_first": {"test": (0, F - cut2 + 0), "val": (F - cut2, F - cut1), "train": (F - cut1, F)},
    }
    frozen = ep.load_model(args.ckpt, device); frozen.log = lambda *a, **k: None
    results = {"settings": vars(args) | {"dataset_root": str(args.dataset_root), "ckpt": str(args.ckpt), "output_dir": str(args.output_dir)}, "splits": {},
               "evaluation_protocol": dict(EVALUATION_PROTOCOL), "comparison_frame": dict(ep.COMPARISON_FRAME), "primary_metric": "mpjpe_mm"}
    for name, blocks in splits.items():
        tr_starts = windows_in(usable, anchor, *blocks["train"], args.window, args.train_stride)
        va_starts = windows_in(usable, anchor, *blocks["val"], args.window, args.window)
        te_starts = windows_in(usable, anchor, *blocks["test"], args.window, args.window)
        tr, va, te = (gather(L, R, j, gl, s, args.window) for s in (tr_starts, va_starts, te_starts))
        entry = {"blocks": {k: [int(v[0]), int(v[1])] for k, v in blocks.items()},
                 "window_starts": {"train": tr_starts, "val": va_starts, "test": te_starts},
                 "n_windows": {"train": len(tr_starts), "val": len(va_starts), "test": len(te_starts)},
                 "frozen": ad.evaluate(frozen, *te, device)}
        m_ft, info_ft = train(frozen, tr, va, device, args.epochs, args.lr_finetune, args.patience, args.seed)
        entry["finetuned_from_unity"] = ad.evaluate(m_ft, *te, device) | {"training": {k: v for k, v in info_ft.items() if k != "history"}}
        torch.manual_seed(args.seed)
        m_sc, info_sc = train(fresh_model(device), tr, va, device, args.epochs, args.lr_scratch, args.patience, args.seed)
        entry["trained_from_scratch"] = ad.evaluate(m_sc, *te, device) | {"training": {k: v for k, v in info_sc.items() if k != "history"}}
        entry["history"] = {"finetuned_from_unity": info_ft["history"], "trained_from_scratch": info_sc["history"]}
        if args.save_weights:
            wdir = args.output_dir / "weights"; wdir.mkdir(parents=True, exist_ok=True)
            save_weights(m_ft, wdir / f"{name}_finetuned_from_unity.pt")
            save_weights(m_sc, wdir / f"{name}_trained_from_scratch.pt")
        results["splits"][name] = entry
        print(f"--- {name}: train {len(tr_starts)} windows (stride {args.train_stride}), val {len(va_starts)}, test {len(te_starts)}")
        for cond in ("frozen", "finetuned_from_unity", "trained_from_scratch"):
            r = entry[cond]; extra = f"  (best epoch {r['training']['best_epoch']}, val {r['training']['best_val_mm']:.1f})" if "training" in r else ""
            print(f"  {cond:22s} avg {r['canonical_average']['mpjpe_mm']:5.1f}/{r['canonical_average']['pa_mpjpe_mm']:4.1f}  "
                  f"aligned {r['aligned_average']['mpjpe_mm']:5.1f}/{r['aligned_average']['pa_mpjpe_mm']:4.1f}  "
                  f"model {r['model']['mpjpe_mm']:5.1f}/{r['model']['pa_mpjpe_mm']:4.1f}  "
                  f"gate-only {r['gate_only']['mpjpe_mm']:5.1f}/{r['gate_only']['pa_mpjpe_mm']:4.1f}{extra}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
