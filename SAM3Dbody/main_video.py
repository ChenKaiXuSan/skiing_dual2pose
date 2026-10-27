#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Run SAM 3D Body on one plain video file and save one NPZ per frame.

Unlike ``main_unity.py`` / ``main_ski_pose_ptz.py``, which walk dataset
directories of extracted frames, this entry point streams frames straight
from a video with OpenCV, keeps only the largest detected person, and writes
``<output-dir>/<frame:06d>_sam3d_body.npz`` through :func:`SAM3Dbody.save.save_frame`
(key ``output``).  The RGB frame itself is *not* stored in the NPZ.

Example::

    python -m SAM3Dbody.main_video \
        --video /path/to/pocket_1.mp4 \
        --output-dir /path/to/sam3d_body_results/chen_4/pocket_1 \
        --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .infer import select_best_person, setup_sam_3d_body, setup_visualizer
from .save import save_frame
from .vis import vis_results

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def build_cfg(ckpt_root: Path, gpu: int):
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "sam3d_body.yaml")
    cfg.model.root_path = str(ckpt_root)
    cfg.infer.gpu = gpu
    return cfg


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return "unknown"


def run(args: argparse.Namespace) -> dict:
    logging.getLogger("SAM3Dbody.save").setLevel(logging.WARNING)
    cfg = build_cfg(args.ckpt_root, args.gpu)
    estimator = setup_sam_3d_body(cfg)
    visualizer = setup_visualizer() if args.vis_every > 0 else None

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_target = n_total if args.max_frames is None else min(n_total, args.max_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    none_detected: list[int] = []
    idx = 0
    t0 = time.time()
    with tqdm(total=n_target, desc=args.video.name, disable=args.quiet) as pbar:
        while idx < n_target:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            outputs = estimator.process_one_image(img=frame_rgb, bboxes=None)
            best, _ = select_best_person(outputs, verbose=False)
            if best is None:
                none_detected.append(idx)
            else:
                if visualizer is not None and idx % args.vis_every == 0:
                    vis_results(
                        img_cv2=frame_rgb,
                        outputs=[best],
                        save_dir=str(args.output_dir / "visualization"),
                        image_name=f"frame_{idx:06d}",
                        faces=estimator.faces,
                        visualizer=visualizer,
                        cfg=cfg.visualize,
                    )
                best = dict(best)
                best["frame_idx"] = idx
                save_frame(output=best, save_dir=args.output_dir, frame_idx=idx)
            idx += 1
            pbar.update(1)
    cap.release()
    torch.cuda.empty_cache()

    if none_detected:
        with open(args.output_dir / "none_detected_frames.txt", "w") as f:
            f.write("\n".join(str(i) for i in none_detected) + "\n")

    manifest = {
        "video": str(args.video.resolve()),
        "video_md5": _md5(args.video),
        "video_frames": n_total,
        "video_fps": fps,
        "video_size": [width, height],
        "frames_processed": idx,
        "frames_saved": idx - len(none_detected),
        "none_detected": none_detected,
        "checkpoint": str(Path(cfg.model.checkpoint_path).resolve()),
        "detector": cfg.model.get("detector_name", "vitdet"),
        "fov_estimator": cfg.model.get("fov_name", "moge2"),
        "git_commit": _git_commit(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "ckpt" / "sam-3d-body-dinov3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--vis-every", type=int, default=0, help="save a visualization every K frames (0 = none)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = run(parse_args())
    print(json.dumps({k: v for k, v in m.items() if k != "none_detected"}, indent=1))
