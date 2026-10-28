#!/usr/bin/env python3
"""Evaluate CanonFuse3D on a hand-held dual-Pocket recording against a video 3D reference.

Data package ``skiing_pocket3_20261012``: two hand-held DJI Osmo Pocket 3 cameras
follow one subject indoors; for session ``chen_4`` a 3D reference (COCO-17 keypoints,
metres, floor frame) was triangulated from eight synchronized external cameras and
resampled to every Pocket frame (``gt/pocket_aligned/<session>_pocket_<n>.npz``).

Protocol (mirrors the canonical-space protocol of the manuscript):

* Each clip starts with (and occasionally returns to) the phone showing the QR-code
  clock; frames whose largest detection is shorter than ``--min-bbox-frac`` of the
  frame height are excluded (in-view rule).
* ``pocket_1`` is the left stream and ``pocket_2`` the right stream.  Frames are paired
  by the UTC time of each frame recorded by the QR-clock synchronization (nearest
  neighbour, residual below half a frame).
* SAM 3D Body per-frame predictions (``SAM3Dbody.main_video``) are filtered to the
  15-joint model skeleton; the COCO-17 reference is mapped to the same skeleton
  (eyes, shoulders, elbows, wrists as hands, hips, knees, ankles; neck := shoulder
  midpoint).
* Consecutive non-overlapping windows of ``--window`` frames; each frame is
  canonicalized independently for the left stream, right stream and reference.
  Each checkpoint retains its native skeleton and saved face-sign convention.
  Its output axes are converted using the input-left frame into the declared
  Pocket15 comparison frame; all baselines and GT use that same frame. No GT
  fit or output recanonicalization enters the primary metric.
* Metrics on the common 13 joints (model indices 2--14) under the reference
  reliability mask: MPJPE over reliable joint-frames, PA-MPJPE over frames whose 13
  joints are all reliable, acceleration error over fully reliable frame triplets.
  A 12-joint variant without the approximated neck is reported as a check.
* Methods: left canonical, right canonical, canonical average, aligned average
  (Sim(3) right-to-left), and for every frozen checkpoint CanonFuse3D and its
  gate-only output (learned output residual removed at inference).
* Primary metric: common-13 MPJPE of the model output directly against the
  per-frame canonical reference, with no further output alignment. PA-MPJPE,
  output-recanonicalized MPJPE, window-aligned MPJPE and root-relative MPJPE
  are auxiliary diagnostics.

Outputs (``--output-dir``): ``summary.json``, ``per_frame.csv``, ``per_window.csv``,
``pocket3_<session>.tex`` and ``per_frame_mpjpe.pdf``.

Example::

    python -m dual2pose.pocket3.evaluate --session chen_4 --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual2pose.map_config import (  # noqa: E402
    FILTERED_15_TO_COMMON_13_INDICES,
    FILTERED_KPTS_MAPPING,
    filter_sam3d_body_kpts,
)
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL, common13_pose

COMPARISON_FRAME = {
    "axes": "Pocket15 hips/neck and real eyes, per-frame face_sign=true",
    "native_output_mapping": "input-left R_native.T @ R_comparison; no GT or output fit",
    "auxiliary_recanonicalization": "common13 hips/neck, face_sign=false",
}

DEFAULT_ROOT = Path("/home/kaixu_chen/skiing/data/skiing_pocket3_20261012")
# Override with DUAL2POSE_CKPT.  The archived manuscript checkpoint below was
# trained under the first-frame protocol and is rejected by the protocol check;
# point this at a checkpoint trained with dual2pose/train_unity.py on this branch.
DEFAULT_CKPT = Path(os.environ.get(
    "DUAL2POSE_CKPT", str(REPO_ROOT / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt")))
FPS = 29.97
LEFT_HIP, RIGHT_HIP, NECK = 6, 7, 14

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
          "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee",
          "r_knee", "l_ankle", "r_ankle"]
# model 15-joint order (FILTERED_KPTS_MAPPING) -> COCO-17 source joint(s)
MODEL15_FROM_COCO: Dict[int, Tuple[str, ...]] = {
    0: ("l_eye",), 1: ("r_eye",),
    2: ("l_shoulder",), 3: ("r_shoulder",),
    4: ("l_elbow",), 5: ("r_elbow",),
    6: ("l_hip",), 7: ("r_hip",),
    8: ("l_knee",), 9: ("r_knee",),
    10: ("l_ankle",), 11: ("r_ankle",),
    12: ("r_wrist",), 13: ("l_wrist",),
    14: ("l_shoulder", "r_shoulder"),   # neck := shoulder midpoint
}
COMMON13 = list(FILTERED_15_TO_COMMON_13_INDICES)       # 2..14
COMMON12_NO_NECK = [i for i in COMMON13 if i != NECK]   # 2..13
BASE_METHODS = ["left", "right", "canonical_average", "aligned_average"]
METHOD_LABELS = {
    "left": "Left canonical (pocket 1)",
    "right": "Right canonical (pocket 2)",
    "canonical_average": "Canonical average",
    "aligned_average": "Aligned average",
}


def default_output_dir(task: str) -> Path:
    """Each invocation gets a fresh namespace; historical evidence is untouched."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return REPO_ROOT / "logs" / "per_frame_common13_v1" / "pocket3" / task / stamp


def fusion_methods(labels: List[str]) -> List[str]:
    """Method keys for every checkpoint: full model and the gate-only output (residual removed)."""
    out = []
    for lab in labels:
        out += [f"canonfuse3d[{lab}]", f"canonfuse3d_gate_only[{lab}]"]
    return out


def method_label(m: str) -> str:
    if m in METHOD_LABELS:
        return METHOD_LABELS[m]
    lab = m[m.index("[") + 1:-1]
    if m.startswith("canonfuse3d_gate_only"):
        return f"CanonFuse3D, gate only ({lab} checkpoint)"
    return f"CanonFuse3D ({lab} checkpoint)"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def map_coco17_to_model15(joints: np.ndarray, reliable: np.ndarray, names: List[str]
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """COCO-17 (F,17,3) and (F,17) -> model 15-joint (F,15,3) and (F,15)."""
    if list(names) != COCO17:
        raise ValueError(f"unexpected joint order: {list(names)}")
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros((joints.shape[0], 15, 3), dtype=np.float32)
    rel = np.zeros((joints.shape[0], 15), dtype=bool)
    for k, srcs in MODEL15_FROM_COCO.items():
        src = [idx[s] for s in srcs]
        out[:, k] = joints[:, src].mean(axis=1)
        rel[:, k] = reliable[:, src].all(axis=1) & np.isfinite(joints[:, src]).all(axis=(1, 2))
    return out, rel


def load_gt(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    joints, rel = map_coco17_to_model15(
        np.asarray(z["video_gt_joints"], dtype=np.float32),
        np.asarray(z["video_gt_reliable"], dtype=bool),
        [str(n) for n in z["joint_names"]],
    )
    return {"frame": np.asarray(z["frame"]), "utc": np.asarray(z["utc_unix_s"], dtype=np.float64),
            "joints": joints, "reliable": rel}


def load_sam3d_dir(path: Path, n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame NPZ written by SAM3Dbody.main_video -> (F,15,3) poses, (F,) validity, (F,4) bbox."""
    poses = np.zeros((n_frames, 15, 3), dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)
    bbox = np.full((n_frames, 4), np.nan, dtype=np.float32)
    for f in sorted(path.glob("*_sam3d_body.npz")):
        idx = int(f.name.split("_")[0])
        if idx >= n_frames:
            continue
        with np.load(f, allow_pickle=True) as z:
            out = z["output"].item()
        kp = np.asarray(out["pred_keypoints_3d"], dtype=np.float32)
        poses[idx] = filter_sam3d_body_kpts(kp)
        valid[idx] = np.isfinite(poses[idx]).all()
        bbox[idx] = np.asarray(out["bbox"], dtype=np.float32)[:4]
    return poses, valid, bbox


def subject_in_view(bbox: np.ndarray, valid: np.ndarray, frame_height: float,
                    min_frac: float = 0.35) -> np.ndarray:
    """Per-frame flag: the largest detection is at least ``min_frac`` of the frame height.

    Each clip starts with, and occasionally returns to, the phone showing the
    QR-code clock used for synchronization; the detector then returns the hand or
    the phone (detections 100-350 px tall in 1080p), whereas the subject is
    450-960 px tall throughout the recording.  Frames failing the rule are
    excluded from the evaluation; no other image-based filtering is applied.
    """
    height = bbox[:, 3] - bbox[:, 1]
    return valid & np.nan_to_num(height >= min_frac * frame_height, nan=False)


# --------------------------------------------------------------------------- #
# pairing and windows
# --------------------------------------------------------------------------- #
def pair_by_utc(utc_left: np.ndarray, utc_right: np.ndarray, fps: float = FPS
                ) -> Tuple[np.ndarray, np.ndarray]:
    """For every left frame return the nearest right frame index (-1 if none within half a frame)
    and the signed residual right - left in seconds."""
    order = np.argsort(utc_right)
    sorted_utc = utc_right[order]
    pos = np.searchsorted(sorted_utc, utc_left)
    cand = np.stack([np.clip(pos - 1, 0, len(sorted_utc) - 1), np.clip(pos, 0, len(sorted_utc) - 1)], axis=1)
    diffs = sorted_utc[cand] - utc_left[:, None]
    best = np.argmin(np.abs(diffs), axis=1)
    j = order[cand[np.arange(len(utc_left)), best]]
    resid = diffs[np.arange(len(utc_left)), best]
    ok = np.abs(resid) <= 0.5 / fps
    return np.where(ok, j, -1), resid


ANCHOR_JOINTS = [0, 1, LEFT_HIP, RIGHT_HIP, NECK]   # landmarks that define the canonical frame


def build_windows(usable: np.ndarray, window: int, stride: int, anchor_ok: np.ndarray | None = None) -> List[int]:
    """Start indices of windows whose ``window`` consecutive left frames are all usable.

    ``anchor_ok`` (optional, per frame) additionally requires every frame of a window
    to have a reliable reference anchor (hips, neck, eyes): under the per-frame protocol
    each frame of the reference defines its own body frame, so one unreliable anchor
    frame would corrupt that frame's reference instead of just masking a joint.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    starts = []
    i = 0
    while i + window <= len(usable):
        if usable[i:i + window].all() and (anchor_ok is None or anchor_ok[i:i + window].all()):
            starts.append(i)
            i += stride
        else:
            i += 1
    return starts


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metric_inputs(pred, gt, rel, joints):
    """Legacy joint selectors refer to Pocket15; arrays may be native13 or 15."""
    indices = [COMMON13.index(j) for j in joints]
    pred, gt = common13_pose(pred), common13_pose(gt)
    rel = rel[..., COMMON13] if rel.shape[-1] == 15 else rel
    if pred.shape != gt.shape or rel.shape != gt.shape[:-1]:
        raise ValueError("Prediction, GT and reliability shapes differ")
    return pred[..., indices, :], gt[..., indices, :], rel[..., indices]


def masked_mpjpe(pred: np.ndarray, gt: np.ndarray, rel: np.ndarray, joints: List[int]) -> Tuple[float, int]:
    p, g, m = metric_inputs(pred, gt, rel, joints)
    if not m.any():
        raise ValueError("Primary MPJPE has no valid joint-frames")
    d = np.linalg.norm(p[m] - g[m], axis=-1)
    if not np.isfinite(d).all():
        raise ValueError("Primary MPJPE contains nonfinite reliable predictions or GT")
    return float(d.mean()), int(m.sum())


def per_frame_mpjpe(pred: np.ndarray, gt: np.ndarray, rel: np.ndarray, joints: List[int]) -> np.ndarray:
    p, g, m = metric_inputs(pred, gt, rel, joints)
    d = np.linalg.norm(p - g, axis=-1)
    with np.errstate(invalid="ignore"):
        return np.where(m.sum(-1) > 0, np.where(m, d, 0).sum(-1) / np.maximum(m.sum(-1), 1), np.nan)


def masked_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, rel: np.ndarray, joints: List[int]) -> Tuple[float, int, int]:
    from dual2pose.eval.pa_mpjpe import pa_joint_errors
    pred, gt, rel = metric_inputs(pred, gt, rel, joints)
    full = rel.all(axis=-1)
    if not full.any():
        return float("nan"), 0, 0
    err, degenerate = pa_joint_errors(pred[full], gt[full])
    return float(err.mean()), int(full.sum()), int(degenerate.sum())


def umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (scale, proper rotation, translation) mapping src -> dst, (N,3) each."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    u, sing, vt = np.linalg.svd(xs.T @ xd / len(src))
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = (sing * d).sum() / var_s if var_s > 0 else 1.0
    t = mu_d - scale * mu_s @ rot
    return scale, rot, t


def sequence_aligned_errors(pred: np.ndarray, gt: np.ndarray, rel: np.ndarray, joints: List[int]) -> np.ndarray:
    """Per-frame mean joint error (W,T) after ONE similarity transform per window.

    The transform is fitted on all reliable joint-frames of the window; frames without
    reliable joints are NaN.  This removes the global frame offset between the two
    skeleton conventions (SAM3D joints vs. COCO video reference) while keeping every
    per-frame pose and temporal error; it is the sequence-level analogue of the
    annotation alignment used for Ski-PTZ-Pose in the manuscript.
    """
    pred, gt, rel = metric_inputs(pred, gt, rel, joints)
    W, T = pred.shape[:2]
    out = np.full((W, T), np.nan)
    for w in range(W):
        m = rel[w]
        if m.sum() < 4:
            continue
        p = pred[w].astype(np.float64)
        g = gt[w].astype(np.float64)
        s_, r_, t_ = umeyama_similarity(p[m], g[m])
        d = np.linalg.norm(s_ * p @ r_ + t_ - g, axis=-1)
        with np.errstate(invalid="ignore"):
            out[w] = np.where(m.sum(-1) > 0, np.where(m, d, 0).sum(-1) / np.maximum(m.sum(-1), 1), np.nan)
    return out


def sequence_aligned_mpjpe(pred: np.ndarray, gt: np.ndarray, rel: np.ndarray, joints: List[int]) -> Tuple[float, int]:
    """Mean over all reliable joint-frames of the per-window similarity-aligned error."""
    errs = sequence_aligned_errors(pred, gt, rel, joints)
    counts = metric_inputs(pred, gt, rel, joints)[2].sum(-1)
    ok = counts > 0
    total = float((errs[ok] * counts[ok]).sum())
    count = int(counts[ok].sum())
    return (total / count if count else float("nan")), count


def per_frame_canonical(P: np.ndarray, *, model=None) -> np.ndarray:
    """Auxiliary recanonicalization; common13 uses hips/neck without invented eyes."""
    import torch
    from dual2pose.trainer.canonicalize import canonicalize_pose_torch
    import warnings
    W, T = P.shape[:2]
    J = P.shape[-2]
    poses = torch.from_numpy(np.ascontiguousarray(P)).float()
    c, transform = (model.canonicalize(native_pose(model, poses)) if model is not None else canonicalize_pose_torch(
        poses, left_hip=4 if J == 13 else LEFT_HIP, right_hip=5 if J == 13 else RIGHT_HIP,
        neck=12 if J == 13 else NECK, mode="per_frame", enforce_face_z_positive=J == 15))
    n_degenerate = int(transform["degenerate"].sum())
    if n_degenerate:
        warnings.warn(f"{n_degenerate} of {W * T} frames had degenerate landmarks; "
                      "their rotation was taken from a neighbouring frame of the same window")
    return c.detach().cpu().numpy()


def root_relative(P: np.ndarray) -> np.ndarray:
    """Subtract the per-frame hip midpoint (translation only)."""
    hips = [4, 5] if P.shape[-2] == 13 else [LEFT_HIP, RIGHT_HIP]
    return P - P[..., hips, :].mean(-2, keepdims=True)


def masked_accel_error(pred: np.ndarray, gt: np.ndarray, rel: np.ndarray, joints: List[int]) -> Tuple[float, int]:
    """Second-difference error within each window; pred/gt are (W,T,J,3)."""
    pred, gt, rel = metric_inputs(pred, gt, rel, joints)
    ap = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    ag = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
    m = rel[:, 2:] & rel[:, 1:-1] & rel[:, :-2]
    d = np.linalg.norm(ap - ag, axis=-1)
    mm = m
    return float(d[mm].mean()) if mm.any() else float("nan"), int(mm.sum())


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(ckpt: Path, device):
    from dual2pose.training_protocol import load_fusion_checkpoint

    return load_fusion_checkpoint(ckpt, device=device)


def model_inputs(model, left, right):
    """Canonicalize raw (W,T,15,3) tensors the way ``model`` was trained (per-frame body frame)."""
    return model.canonicalize(native_pose(model, left))[0], model.canonicalize(native_pose(model, right))[0]


def native_pose(model, pose):
    return common13_pose(pose) if getattr(model, "neck_idx", NECK) == 12 else pose


def comparison_canonical(pose):
    from dual2pose.trainer.canonicalize import canonicalize_pose_torch
    return canonicalize_pose_torch(pose, left_hip=LEFT_HIP, right_hip=RIGHT_HIP,
                                   neck=NECK, mode="per_frame", enforce_face_z_positive=True)


def to_comparison_frame(output, native_transform, raw_left):
    """Change axes only, using the input; retain every predicted offset and scale."""
    comparison = comparison_canonical(raw_left)[1]
    rotation = native_transform["R"].transpose(-1, -2) @ comparison["R"]
    return common13_pose(output) @ rotation


def training_batch(model, L, R, G, REL, device):
    """Source-only adapter; canonicalization and supervision belong to the trainer."""
    import torch
    tensors = [torch.as_tensor(x, device=device) for x in (L, R, G)]
    rel = torch.as_tensor(REL, device=device, dtype=torch.bool)
    if getattr(model, "neck_idx", NECK) == 12:
        rel = rel[..., COMMON13]
    return {"kpt3d_sam": {"cam1": native_pose(model, tensors[0]), "cam2": native_pose(model, tensors[1])},
            "kpt3d_gt": native_pose(model, tensors[2]), "kpt3d_gt_reliable": rel}


def validate_targets(G, REL):
    if G.shape[:-1] != REL.shape or G.shape[0] == 0:
        raise ValueError("Pocket3 targets require nonempty windows and matching reliability")
    if not REL[..., ANCHOR_JOINTS].all():
        raise ValueError("Every comparison frame requires reliable GT anchors")
    if not np.isfinite(G[REL]).all():
        raise ValueError("Reliable GT coordinates must be finite")
    import torch
    clean = np.where(REL[..., None], G, 0)
    if comparison_canonical(torch.as_tensor(clean))[1]["degenerate"].any():
        raise ValueError("GT anchors must define a nondegenerate body frame")


def validate_pocket_models(models):
    """Validate genuine native15/native13 skeletons without changing saved settings."""
    first = None
    for label, model in models.items():
        if model.canonical_mode != "per_frame":
            raise ValueError(f"{label}: Pocket3 canonicalization must be per_frame")
        hips_neck = (getattr(model, "left_hip_idx", LEFT_HIP),
                     getattr(model, "right_hip_idx", RIGHT_HIP), getattr(model, "neck_idx", NECK))
        if hips_neck not in ((LEFT_HIP, RIGHT_HIP, NECK), (4, 5, 12)):
            raise ValueError(f"{label}: unsupported checkpoint skeleton")
        face = getattr(model, "face_sign", True)
        if hips_neck == (4, 5, 12) and face:
            raise ValueError(f"{label}: common13 skeleton has no eye landmarks for face-sign canonicalization")
        if face and (getattr(model, "left_eye_idx", 0), getattr(model, "right_eye_idx", 1)) != (0, 1):
            raise ValueError(f"{label}: unsupported eye skeleton mapping")
        first = model if first is None else first
    return first


def run_methods(left: np.ndarray, right: np.ndarray, gt: np.ndarray, models: Dict[str, object], device,
                reference: str = "per_frame") -> Dict[str, np.ndarray]:
    """Raw Pocket15 sources -> comparison-frame common13 outputs, GT and gates.

    All methods use the same per-frame body frame. ``gt_per_frame`` is retained
    as an alias of ``gt`` for existing result consumers.
    """
    import torch
    from dual2pose.models.main_baselines import aligned_average
    from dual2pose.trainer.canonicalize import canonicalize_pose_torch

    if reference != "per_frame":
        raise ValueError("Pocket3 evaluation requires reference='per_frame'")
    validate_pocket_models(models)
    if left.shape[0] == 0:
        raise ValueError("Pocket3 evaluation requires nonempty windows")
    if left.shape != right.shape or left.shape != gt.shape or left.shape[-2:] != (15, 3):
        raise ValueError("Pocket3 source arrays must have matching (W,T,15,3) shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Pocket3 model inputs must be finite")

    def canonicalize(pose):
        return common13_pose(comparison_canonical(pose)[0])

    keys = BASE_METHODS + fusion_methods(list(models)) + ["gt", "gt_per_frame"] + [f"alpha[{lab}]" for lab in models]
    outs: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    W = left.shape[0]
    chunks = [slice(w, w + 1) for w in range(W)]
    with torch.no_grad():
        for sl in chunks:
            l = torch.from_numpy(left[sl]).to(device)
            r = torch.from_numpy(right[sl]).to(device)
            g = torch.from_numpy(gt[sl]).to(device)
            lc, rc, gc = canonicalize(l), canonicalize(r), canonicalize(g)
            outs["left"].append(lc.cpu().numpy())
            outs["right"].append(rc.cpu().numpy())
            outs["canonical_average"].append((0.5 * (lc + rc)).cpu().numpy())
            outs["aligned_average"].append(aligned_average(lc, rc).cpu().numpy())
            outs["gt"].append(gc.cpu().numpy())
            outs["gt_per_frame"].append(gc.cpu().numpy())
            for lab, model in models.items():
                # The learned fusion and baselines share the per-frame protocol.
                ml, transform = model.canonicalize(native_pose(model, l))
                mr = model.canonicalize(native_pose(model, r))[0]
                fused, aux = model.models(ml, mr, return_components=True)
                outs[f"canonfuse3d[{lab}]"].append(to_comparison_frame(fused, transform, l).cpu().numpy())
                outs[f"canonfuse3d_gate_only[{lab}]"].append(to_comparison_frame(fused - aux["output_residual"], transform, l).cpu().numpy())
                outs[f"alpha[{lab}]"].append(aux["alpha"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_table(path: Path, metrics: Dict[str, Dict[str, float]], methods: List[str], session: str,
                n_windows: int, n_frames: int) -> None:
    rows = []
    for m in methods:
        r = metrics[m]
        rows.append(f"{method_label(m)} & {r['mpjpe_m']*1000:.1f} & {r['seq_aligned_mpjpe_m']*1000:.1f} & "
                    f"{r['pa_mpjpe_m']*1000:.1f} & {r['accel_m']*1000:.2f} \\\\")
    body = "\n".join(rows)
    per_frame_note = " Primary MPJPE scores common-13 outputs directly against per-frame canonical GT, with no output alignment. Aligned scores are auxiliary."
    path.write_text(rf"""\begin{{table}}[!ht]
\centering
\caption{{{{\bf Hand-held dual-Pocket recording ({session}) against the external-camera video reference.}} Common-13 canonical-space errors in millimetres over {n_windows} non-overlapping 30-frame windows ({n_frames} frame pairs). Seq.-aligned MPJPE applies one similarity transform per window; PA-MPJPE aligns every frame; acceleration error is in mm per frame squared. Lower is better.}}
\label{{tab:pocket3_{session}}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & MPJPE $\downarrow$ & Seq.-aligned $\downarrow$ & PA-MPJPE $\downarrow$ & Accel. $\downarrow$ \\\\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{flushleft}}\footnotesize The reference is a COCO-17 video triangulation; the neck of the model skeleton is approximated by the shoulder midpoint, and the two eye landmarks enter canonicalization only. Both Pocket cameras are uncalibrated and hand-held; no camera parameter is used by any method. Gate only removes the learned output residual from the frozen model at inference.{per_frame_note}\end{{flushleft}}
\end{{table}}
""")


def write_figure(path: Path, series: Dict[str, np.ndarray], methods: List[str], time_s: np.ndarray,
                 alpha_mean: np.ndarray, primary: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    palette = ["#9e9e9e", "#bdbdbd", "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for k, m in enumerate(methods):
        axes[0].plot(time_s, series[m] * 1000, lw=1.0 if m in ("left", "right") else 1.3,
                     color=palette[k % len(palette)], label=method_label(m))
    axes[0].set_ylabel("Common-13 MPJPE, no output alignment (mm)")
    axes[0].legend(ncol=2, fontsize=7, frameon=False)
    axes[1].plot(time_s, alpha_mean, color="#444444", lw=1.0)
    axes[1].axhline(0.5, color="#999999", lw=0.6, ls=":")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel(f"Mean gate ({primary})")
    axes[1].set_xlabel("Time in pocket 1 (s)")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> Dict:
    import torch

    root = Path(args.dataset_root)
    gt_left = load_gt(root / "gt" / "pocket_aligned" / f"{args.session}_pocket_1.npz")
    gt_right = load_gt(root / "gt" / "pocket_aligned" / f"{args.session}_pocket_2.npz")
    sam_root = Path(args.sam3d_root) if args.sam3d_root else root / "sam3d_body_results" / args.session
    left, left_valid, left_bbox = load_sam3d_dir(sam_root / "pocket_1", len(gt_left["frame"]))
    right, right_valid, right_bbox = load_sam3d_dir(sam_root / "pocket_2", len(gt_right["frame"]))
    left_in_view = subject_in_view(left_bbox, left_valid, args.frame_height, args.min_bbox_frac)
    right_in_view = subject_in_view(right_bbox, right_valid, args.frame_height, args.min_bbox_frac)

    j, resid = pair_by_utc(gt_left["utc"], gt_right["utc"])
    usable = (j >= 0) & left_in_view & np.where(j >= 0, right_in_view[np.clip(j, 0, None)], False)
    anchor_ok = gt_left["reliable"][:, ANCHOR_JOINTS].all(axis=1)
    starts = build_windows(usable, args.window, args.stride, anchor_ok)
    if args.max_windows:
        starts = starts[: args.max_windows]
    if not starts:
        raise RuntimeError("no complete window; check SAM3D outputs and synchronization")

    W, T = len(starts), args.window
    li = np.stack([np.arange(s, s + T) for s in starts])          # (W,T) left frame indices
    ri = j[li]
    L = left[li]; R = right[ri]; G = gt_left["joints"][li]; REL = gt_left["reliable"][li]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    ckpts = parse_ckpts(args.ckpt)
    models = {lab: load_model(path, device) for lab, path in ckpts.items()}
    for lab, model in models.items():
        print(f"checkpoint {lab}: canonical_mode={model.canonical_mode}")
    primary = next(iter(ckpts))
    methods = BASE_METHODS + fusion_methods(list(ckpts))
    validate_targets(G, REL)
    out = run_methods(L, R, G, models, device, reference=args.reference)

    # Every method uses the same per-frame reference.
    ref = {m: "gt" for m in methods}
    pfc = {k: per_frame_canonical(out[k]) for k in methods + ["gt", "gt_per_frame"]}
    metrics = {}
    for m in methods:
        gt_m, pf_gt = out[ref[m]], pfc[ref[m]]
        row = {"reference_frame": "per_frame", "joint_subset": "common13", "output_alignment": "none"}
        for name, joints in (("", COMMON13), ("_no_neck", COMMON12_NO_NECK)):
            mp, n_pts = masked_mpjpe(out[m], gt_m, REL, joints)
            sa, n_sa = sequence_aligned_mpjpe(out[m], gt_m, REL, joints)
            rr, _ = masked_mpjpe(root_relative(out[m]), root_relative(gt_m), REL, joints)
            pf, _ = masked_mpjpe(pfc[m], pf_gt, REL, joints)
            pa, n_pa_frames, n_deg = masked_pa_mpjpe(out[m], gt_m, REL, joints)
            ac, n_ac = masked_accel_error(out[m], gt_m, REL, joints)
            ac_pf, _ = masked_accel_error(pfc[m], pf_gt, REL, joints)
            row.update({f"mpjpe_m{name}": mp, f"seq_aligned_mpjpe_m{name}": sa,
                        f"recanonicalized_mpjpe_m{name}": pf,
                        f"root_relative_mpjpe_m{name}": rr, f"per_frame_canonical_mpjpe_m{name}": pf,
                        f"pa_mpjpe_m{name}": pa, f"accel_m{name}": ac, f"per_frame_canonical_accel_m{name}": ac_pf,
                        f"mpjpe_points{name}": n_pts, f"seq_aligned_points{name}": n_sa,
                        f"pa_frames{name}": n_pa_frames, f"pa_degenerate_frames{name}": n_deg,
                        f"accel_points{name}": n_ac})
        metrics[m] = row
    for key in ("mpjpe_m", "seq_aligned_mpjpe_m", "per_frame_canonical_mpjpe_m", "pa_mpjpe_m"):
        base = metrics["canonical_average"][key]
        for m in methods:
            metrics[m][f"gain_vs_canonical_average_pct[{key}]"] = 100.0 * (base - metrics[m][key]) / base if base > 0 else float("nan")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {m: per_frame_mpjpe(out[m], out[ref[m]], REL, COMMON13).reshape(-1) for m in methods}
    series_aligned = {m: sequence_aligned_errors(out[m], out[ref[m]], REL, COMMON13).reshape(-1) for m in methods}
    series_pf = {m: per_frame_mpjpe(pfc[m], pfc[ref[m]], REL, COMMON13).reshape(-1) for m in methods}
    residual_norm = {lab: float(np.linalg.norm(out[f"canonfuse3d[{lab}]"] - out[f"canonfuse3d_gate_only[{lab}]"], axis=-1).mean())
                     for lab in ckpts}
    alpha_mean = out[f"alpha[{primary}]"].reshape(W * T, -1).mean(axis=1)
    time_s = gt_left["frame"][li].reshape(-1) / FPS
    with open(out_dir / "per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left_frame", "right_frame", "sync_residual_ms", "left_bbox_h_px", "right_bbox_h_px",
                    "reliable_joints_13", "alpha_mean"] + [f"mpjpe_{m}_m" for m in methods])
        for k in range(W * T):
            lf, rf = int(li.reshape(-1)[k]), int(ri.reshape(-1)[k])
            w.writerow([lf, rf, f"{resid[lf]*1000:.1f}", f"{left_bbox[lf,3]-left_bbox[lf,1]:.0f}",
                        f"{right_bbox[rf,3]-right_bbox[rf,1]:.0f}", int(REL.reshape(W * T, -1)[k][COMMON13].sum()),
                        f"{alpha_mean[k]:.4f}"] + [f"{series[m][k]:.5f}" for m in methods])
    with open(out_dir / "per_window.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "left_start", "left_end"] + [f"mpjpe_{m}_m" for m in methods])
        for k, s in enumerate(starts):
            w.writerow([k, int(s), int(s + T - 1)] + [f"{np.nanmean(series[m][k*T:(k+1)*T]):.5f}" for m in methods])
    write_table(out_dir / f"pocket3_{args.session}.tex", metrics, methods, args.session, W, W * T)
    write_figure(out_dir / "per_frame_mpjpe.pdf", series, methods, time_s, alpha_mean, primary)
    with open(out_dir / "per_frame_aligned.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left_frame", "time_s"] + [f"per_frame_canonical_err_{m}_m" for m in methods]
                   + [f"seq_aligned_err_{m}_m" for m in methods])
        for k in range(W * T):
            w.writerow([int(li.reshape(-1)[k]), f"{time_s[k]:.3f}"] + [f"{series_pf[m][k]:.5f}" for m in methods]
                       + [f"{series_aligned[m][k]:.5f}" for m in methods])

    def git_commit():
        try:
            return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "unknown"

    sam_manifests = {}
    for cam in ("pocket_1", "pocket_2"):
        mp = sam_root / cam / "manifest.json"
        sam_manifests[cam] = json.loads(mp.read_text()) if mp.exists() else None
        if sam_manifests[cam]:
            sam_manifests[cam].pop("none_detected", None)
    summary = {
        "evaluation_protocol": dict(EVALUATION_PROTOCOL),
        "comparison_frame": dict(COMPARISON_FRAME),
        "primary_metric": "mpjpe_m",
        "session": args.session,
        "dataset_root": str(root),
        "protocol": {
            "left_stream": "pocket_1", "right_stream": "pocket_2",
            "pairing": "nearest UTC frame, residual <= half a frame",
            "window_frames": T, "stride_frames": args.stride,
            "reference": "per_frame",
            "anchor_rule": "every frame of a window must have reliable reference hips, neck and eyes (each frame defines its own body frame)",
            "joint_subset": "common 13 (model indices 2-14); neck := COCO shoulder midpoint",
            "primary_alignment": "none after input/GT per-frame canonicalization; score outputs directly",
            "auxiliary_alignment": {
                "pa_mpjpe_m": "per-frame similarity fit",
                "recanonicalized_mpjpe_m": "rebuild each output's body frame; alias: per_frame_canonical_mpjpe_m",
                "seq_aligned_mpjpe_m": "one similarity fit per window",
                "root_relative_mpjpe_m": "subtract each output's pelvis",
            },
            "model_input_frame": {lab: model.canonical_mode for lab, model in models.items()},
            "reference_frame_per_method": {m: metrics[m]["reference_frame"] for m in methods},
            "mask": "video_gt_reliable from the pocket-aligned reference",
            "units": "metres (reference) / SAM3D metric camera coordinates; canonicalization keeps scale",
        },
        "counts": {
            "left_frames": int(len(gt_left["frame"])), "right_frames": int(len(gt_right["frame"])),
            "left_detected": int(left_valid.sum()), "right_detected": int(right_valid.sum()),
            "paired_frames": int((j >= 0).sum()), "usable_frames": int(usable.sum()),
            "left_in_view": int(left_in_view.sum()), "right_in_view": int(right_in_view.sum()),
            "in_view_rule": f"largest detection >= {args.min_bbox_frac} x frame height ({args.frame_height:.0f} px)",
            "windows": W, "evaluated_frames": W * T,
            "reliable_fraction_common13": float(REL[..., COMMON13].mean()),
            "sync_residual_ms_mean": float(np.mean(np.abs(resid[j >= 0])) * 1000),
            "sync_residual_ms_max": float(np.max(np.abs(resid[j >= 0])) * 1000),
        },
        "metrics": metrics,
        "methods": methods,
        "alpha": {lab: {"mean": float(out[f"alpha[{lab}]"].mean()), "std": float(out[f"alpha[{lab}]"].std())} for lab in ckpts},
        "output_residual_norm_m": residual_norm,
        "inputs": {
            "gt_pocket_1_md5": _md5(root / "gt" / "pocket_aligned" / f"{args.session}_pocket_1.npz"),
            "gt_pocket_2_md5": _md5(root / "gt" / "pocket_aligned" / f"{args.session}_pocket_2.npz"),
            "checkpoints": {lab: {"path": str(path.resolve()), "md5": _md5(path),
                                  "canonical_mode": models[lab].canonical_mode,
                                  "native_joints": 13 if models[lab].neck_idx == 12 else 15,
                                  "face_sign": models[lab].face_sign} for lab, path in ckpts.items()},
            "sam3d_manifests": sam_manifests,
        },
        "git_commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--session", default="chen_4")
    ap.add_argument("--sam3d-root", type=Path, default=None)
    ap.add_argument("--ckpt", action="append", default=None,
                    help="label=path of a frozen checkpoint; repeatable; default Unity=<benchmark checkpoint>")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--frame-height", type=float, default=1080.0)
    ap.add_argument("--reference", choices=("per_frame",), default="per_frame",
                    help="fixed per-frame body-frame protocol for every method")
    ap.add_argument("--min-bbox-frac", type=float, default=0.35,
                    help="in-view rule: a frame is used only if the detection height exceeds this fraction of the frame")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=default_output_dir("evaluation"))
    return ap.parse_args(argv)


def parse_ckpts(items) -> Dict[str, Path]:
    if not items:
        return {"Unity": DEFAULT_CKPT}
    out: Dict[str, Path] = {}
    for it in items:
        lab, _, path = it.partition("=")
        if not path:
            lab, path = f"ckpt{len(out) + 1}", lab
        out[lab] = Path(path)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    s = evaluate(args)
    keys = ("mpjpe_m", "pa_mpjpe_m", "accel_m", "recanonicalized_mpjpe_m")
    print(f"windows={s['counts']['windows']} frames={s['counts']['evaluated_frames']} usable={s['counts']['usable_frames']} "
          f"in_view L/R={s['counts']['left_in_view']}/{s['counts']['right_in_view']}")
    print(f"{'method':48s} " + " ".join(f"{k[:-2]:>18s}" for k in keys))
    for m in s["methods"]:
        print(f"{method_label(m):48s} " + " ".join(f"{s['metrics'][m][k]*1000:18.1f}" for k in keys))


if __name__ == "__main__":
    main()
