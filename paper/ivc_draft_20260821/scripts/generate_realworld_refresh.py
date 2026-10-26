"""Re-infer fixed, previously unshown clips; never synthesize or select by quality."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
DATA = Path("/home/kaixu_chen/skiing/data/sam3d_body_results/person")
CHECKPOINT = REPO / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
RUNS = (("pro_2", "Outdoor"), ("run_4", "Indoor"))
BONES = ((12, 0), (0, 2), (2, 11), (12, 1), (1, 3), (3, 10),
         (12, 4), (12, 5), (4, 5), (4, 6), (6, 8), (5, 7), (7, 9))


def select_midpoint(left_ids, right_ids):
    shared = sorted(set(left_ids) & set(right_ids))
    if not shared:
        raise ValueError("No shared frame IDs")
    return shared[len(shared) // 2]


def display_body(pose):
    pose = np.asarray(pose)
    if pose.shape != (15, 3) or not np.isfinite(pose).all():
        raise ValueError("Expected finite (15,3) pose")
    return pose[2:, :][:, [0, 2, 1]]


def display_limits(poses):
    points = np.stack([display_body(p) for p in np.asarray(poses).reshape(-1, 15, 3)])
    lower, upper = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    return (lower + upper) / 2, max(float((upper - lower).max()) * .55, .1)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from journal_figure_style import apply_journal_style, JOURNAL_COLORS
    sys.path[:0] = [str(REPO), str(REPO / "dual2pose")]
    from dual2pose.eval import visualize_true_data as vis
    from dual2pose.map_config import filter_sam3d_body_kpts

    # Correct the legacy module's path only within this figure-generation process.
    vis.REPO_ROOT = REPO
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = vis._load_model(CHECKPOINT).to("cpu")
    rows, records, arrays = [], [], {}
    for run, label in RUNS:
        left = vis._index_npz_files(DATA / run / "left")
        right = vis._index_npz_files(DATA / run / "right")
        shared = sorted(set(left) & set(right))
        fid = select_midpoint(left, right)
        pos = shared.index(fid)
        start = max(0, min(pos - 15, len(shared) - 30))
        ids = shared[start:start + 30]
        if len(ids) != 30 or any(b - a != 1 for a, b in zip(ids, ids[1:])):
            raise ValueError(f"{run}: expected 30 consecutive paired frames")
        inputs = []
        sources = []
        for view, files in (("left", left), ("right", right)):
            seq = []
            for index in ids:
                path = files[index]
                pose = np.asarray(filter_sam3d_body_kpts(vis._load_npz_kpt3d(path)), dtype=np.float32)
                if pose.shape != (15, 3) or not np.isfinite(pose).all():
                    raise ValueError(f"Invalid pose: {path}")
                seq.append(pose)
                sources.append({"view": view, "frame": index, "file": str(path), "sha256": sha256(path)})
            inputs.append(np.stack(seq))
        lc, rc, fused, _ = vis._run_model_on_window(model, *inputs, device="cpu")
        offset = ids.index(fid)
        poses = np.stack([lc[offset], rc[offset], (lc[offset] + rc[offset]) / 2, fused[offset]])
        frames = [vis._load_npz_frame(left[fid]), vis._load_npz_frame(right[fid])]
        if any(frame is None for frame in frames):
            raise ValueError("Missing RGB")
        rows.append((label, run, fid, frames, poses))
        arrays.update({f"{run}_left_input": inputs[0], f"{run}_right_input": inputs[1],
                       f"{run}_left_canonical": lc, f"{run}_right_canonical": rc,
                       f"{run}_fused": fused, f"{run}_displayed": poses})
        records.append({"run": run, "label": label, "frame": fid, "window": ids,
                        "rgb_shapes": [list(f.shape) for f in frames], "sources": sources})
        print(f"{run}: fixed midpoint {fid}, window {ids[0]}..{ids[-1]}", flush=True)

    apply_journal_style()
    colors = [JOURNAL_COLORS[c] for c in ("blue", "orange", "grey", "teal")]
    center, radius = display_limits(np.stack([row[-1] for row in rows]))
    fig = plt.figure(figsize=(10, 4.5))
    grid = fig.add_gridspec(2, 6, width_ratios=[1.15, 1.15, 1, 1, 1, 1],
                           left=.015, right=.995, bottom=.04, top=.90, hspace=.42, wspace=.025)
    titles = ["Left RGB", "Right RGB", "Left 3D", "Right 3D", "Canonical\naverage", "CanonFuse3D"]
    for r, (label, run, fid, frames, poses) in enumerate(rows):
        for c in range(6):
            ax = fig.add_subplot(grid[r, c], projection="3d" if c >= 2 else None)
            if c < 2:
                ax.imshow(frames[c])
                ax.set_axis_off()
            else:
                p = display_body(poses[c - 2])
                for i, j in BONES:
                    ax.plot(*p[[i, j]].T, color=colors[c - 2], linewidth=2.2)
                ax.scatter(*p.T, s=10, color=colors[c - 2], edgecolors="white", linewidths=.25)
                for setter, v in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                    setter(v - radius, v + radius)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=16, azim=-72)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_xlabel("X", labelpad=-14)
                ax.set_ylabel("Z", labelpad=-14)
                ax.set_zlabel("Y", labelpad=-14)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.pane.fill = False
            ax.set_title(titles[c], fontsize=12, pad=5)
        fig.text(.02, .96 if r == 0 else .475,
                 f"({'a' if r == 0 else 'b'}) {label} | {run}, frame {fid}",
                 fontsize=11, fontweight="bold")
    output = PAPER / "figures/journal/realworld_qualitative_refresh.pdf"
    preview = output.with_suffix(".png")
    fig.savefig(output)
    fig.savefig(preview, dpi=160)
    plt.close(fig)
    evidence = PAPER / "evidence/layout_revision_20260917"
    evidence.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evidence / "realworld_refresh_predictions.npz", **arrays)
    metadata = {"selection": "pro_2 and run_4; sorted shared-frame midpoint, chosen before inference",
                "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
                "device": "cpu", "window": 30, "batch_size": 1,
                "display": "13 body joints only; no synthetic head; axis permutation x,z,y; shared limits across all panels",
                "display_center": center.tolist(), "display_radius": radius,
                "rgb": "Full original frames, no crop or generated content",
                "gt": "No independent metric 3D ground truth; no accuracy claim",
                "examples": records, "figure_sha256": sha256(output),
                "predictions_sha256": sha256(evidence / "realworld_refresh_predictions.npz")}
    (evidence / "realworld_refresh_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
