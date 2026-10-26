#!/usr/bin/env python3
"""Generate journal-specific dataset, overview, method, and qualitative figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

try:
    from .journal_figure_style import JOURNAL_COLORS, apply_journal_style
except ImportError:  # Direct script execution.
    from journal_figure_style import JOURNAL_COLORS, apply_journal_style


PAPER = Path(__file__).resolve().parents[1]
DATA_ROOT = PAPER.parents[2] / "data"
DATASET_EXAMPLE_SOURCES = {
    "unity": (
        DATA_ROOT
        / "skiing_unity_dataset"
        / "data_pole_ski"
        / "female"
        / "Anim_Female_Skier_Skiing"
        / "frames"
        / "capture_L3_A000"
        / "frame_000036.png",
        DATA_ROOT
        / "skiing_unity_dataset"
        / "data_pole_ski"
        / "female"
        / "Anim_Female_Skier_Skiing"
        / "frames"
        / "capture_L3_A210"
        / "frame_000036.png",
    ),
    "ski_ptz": (
        DATA_ROOT
        / "Ski-PosePTZ-CameraDataset-png"
        / "data"
        / "test"
        / "seq_412"
        / "cam_02"
        / "image_000075.png",
        DATA_ROOT
        / "Ski-PosePTZ-CameraDataset-png"
        / "data"
        / "test"
        / "seq_412"
        / "cam_03"
        / "image_000075.png",
    ),
    "author": (
        DATA_ROOT
        / "sam3d_body_results"
        / "person"
        / "run_3"
        / "left"
        / "frame_0000_sam_3d_body_outputs.npz",
        DATA_ROOT
        / "sam3d_body_results"
        / "person"
        / "run_3"
        / "right"
        / "frame_0000_sam_3d_body_outputs.npz",
    ),
}


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: float = 8.0,
    weight: str = "normal",
    radius: float = 0.018,
):
    edgecolor = edgecolor or JOURNAL_COLORS["black"]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    label = ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=JOURNAL_COLORS["black"],
        linespacing=1.15,
    )
    return patch, label


def count_text_overflows(fig, records) -> int:
    """Count labels whose rendered extents exceed their rounded boxes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflows = 0
    for patch, label in records:
        if not label.get_text():
            continue
        box_extent = patch.get_window_extent(renderer)
        text_extent = label.get_window_extent(renderer)
        if (
            text_extent.x0 < box_extent.x0 + 1
            or text_extent.x1 > box_extent.x1 - 1
            or text_extent.y0 < box_extent.y0 + 1
            or text_extent.y1 > box_extent.y1 - 1
        ):
            overflows += 1
    return overflows


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=JOURNAL_COLORS["black"],
            shrinkA=1,
            shrinkB=1,
        )
    )


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_study_overview_figure():
    """Show only the system flow, from two pose streams to one refined output."""
    apply_journal_style()
    plt.rcParams["mathtext.default"] = "regular"
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    records = []
    ax.text(0.5, 0.96, "Calibration-free post-estimation dual-view refinement",
            ha="center", va="top", fontsize=10, weight="bold")
    nodes = (
        (0.015, 0.15, "Two monocular\n3D pose streams\n$P^L, P^R$", "#DCEEFF"),
        (0.205, 0.16, "Human-centric\ncoordinates +\npose alignment", "#DFF3EF"),
        (0.405, 0.16, "Cross-view +\ntemporal\nfeatures", "#FFF0DF"),
        (0.605, 0.16, "Joint-frame gate\n+ residual\ncorrection", "#F2E7F5"),
        (0.805, 0.18, "Refined canonical\n3D pose sequence\n$P^F$", "#DFF3EF"),
    )
    for x, width, label, fill in nodes:
        records.append(_box(ax, x, 0.30, width, 0.40, label,
                            facecolor=fill, fontsize=7.5))
    for current, following in zip(nodes[:-1], nodes[1:]):
        _arrow(ax, (current[0] + current[1] + 0.006, 0.50),
               (following[0] - 0.006, 0.50))
    ax.text(0.5, 0.13, "Approximately synchronized pose inputs; no camera parameters",
            ha="center", va="center", fontsize=8)
    return fig, records


def render_study_overview(output: Path) -> None:
    """Show the pose-space refinement system flow."""
    fig, records = build_study_overview_figure()
    overflow_count = count_text_overflows(fig, records)
    if overflow_count:
        plt.close(fig)
        raise ValueError(f"Study-overview labels overflow {overflow_count} boxes")
    _save(fig, output)


def build_method_pipeline_figure():
    """Build the merged architecture + feature-construction figure."""
    apply_journal_style()
    plt.rcParams["mathtext.default"] = "regular"
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    records = []

    ax.text(0.5, 0.985, "CanonFuse3D: human-centric feature construction and gated fusion",
            ha="center", va="top", fontsize=10, weight="bold")

    # Row 1: pose-space preprocessing -------------------------------------------------
    row1 = (
        (0.02, 0.27, "1  Input pose streams\n$P^L,\\ P^R$  ($T\\times J\\times 3$ each)\nmonocular estimates per view",
         "#DCEEFF", JOURNAL_COLORS["blue"]),
        (0.345, 0.29, "2  Human-centric canonicalization\n$\\tilde{P}^L,\\ \\tilde{P}^R$\nhip midpoint, hip axis, neck axis",
         "#DFF3EF", JOURNAL_COLORS["teal"]),
        (0.69, 0.29, "3  Sequence-level alignment\n$\\tilde{P}^{L\\to R},\\ \\tilde{P}^{R\\to L}$, rotation cues\none similarity fit per direction",
         "#E8F1FB", JOURNAL_COLORS["blue"]),
    )
    y1, h1 = 0.80, 0.135
    for x, width, label, fill, edge in row1:
        records.append(_box(ax, x, y1, width, h1, label,
                            facecolor=fill, edgecolor=edge, fontsize=6.6))
    for current, following in zip(row1[:-1], row1[1:]):
        _arrow(ax, (current[0] + current[1] + 0.004, y1 + h1 / 2),
               (following[0] - 0.004, y1 + h1 / 2))

    # Row 2: feature construction container ------------------------------------------
    cy0, cy1 = 0.445, 0.745
    ax.add_patch(FancyBboxPatch((0.02, cy0), 0.96, cy1 - cy0,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor="#F6F7F9", edgecolor=JOURNAL_COLORS["orange"],
                                linewidth=1.0, linestyle="--"))
    ax.text(0.5, cy1 - 0.022,
            "4  Geometric-motion feature construction  (left stream shown; the right stream swaps L and R)",
            ha="center", va="top", fontsize=7.6, weight="bold", color=JOURNAL_COLORS["black"])
    _arrow(ax, (0.835, y1 - 0.004), (0.835, cy1 + 0.004))
    cues = (
        ("Own-view pose\n$\\tilde{P}^{L}$\n3 ch", "#DCEEFF"),
        ("Aligned other view\n$\\tilde{P}^{R\\to L}$\n3 ch", "#DFF3EF"),
        ("Cross-view difference\n$D^L=\\tilde{P}^{R\\to L}-\\tilde{P}^{L}$\n3 ch", "#FFF0DF"),
        ("Temporal velocity\n$V^L_t=\\tilde{P}^L_t-\\tilde{P}^L_{t-1}$\n3 ch", "#FBE6E2"),
        ("Alignment rotation\n$r^{R\\to L}$ (broadcast)\n3 ch", "#F2E7F5"),
    )
    cw, cgap, cy, ch = 0.176, 0.014, 0.545, 0.13
    for i, (label, color) in enumerate(cues):
        x = 0.035 + i * (cw + cgap)
        records.append(_box(ax, x, cy, cw, ch, label, facecolor=color, fontsize=6.6))
        _arrow(ax, (x + cw / 2, cy - 0.004), (x + cw / 2, cy - 0.034))
    ax.text(0.5, cy - 0.052,
            "concatenate at every joint and frame:  3 + 3 + 3 + 3 + 3 = 15 channels  $\\rightarrow$  $X^L$  (and $X^R$)",
            ha="center", va="center", fontsize=7.2)

    # Row 3: learned fusion ----------------------------------------------------------
    row3 = (
        (0.02, 0.18, "5  Shared joint encoder\nMLP, LayerNorm, GELU\n$H^L=\\phi(X^L),\\ H^R=\\phi(X^R)$",
         "#DCEEFF", JOURNAL_COLORS["blue"]),
        (0.215, 0.27, "6  Cross-view attention\nbidirectional, over joints per frame\n$\\hat{H}^L,\\ \\hat{H}^R$ with residual + LayerNorm",
         "#F2E7F5", JOURNAL_COLORS["magenta"]),
        (0.50, 0.185, "7  Temporal refinement\ndepthwise conv, $k=5$\n$\\bar{H}=\\hat{H}+\\mathrm{TC}(\\mathrm{LN}(\\hat{H}))$",
         "#FFF0DF", JOURNAL_COLORS["orange"]),
        (0.70, 0.295, "8  Gate + residual,  $Z=[\\bar{H}^L,\\bar{H}^R]$\n$\\alpha=\\sigma(g_\\theta(Z)),\\ \\Delta P=h_\\theta(Z)$\n$P^F=\\alpha B^L+(1-\\alpha)B^R+\\Delta P$",
         "#FBE6E2", JOURNAL_COLORS["red"]),
    )
    y3, h3 = 0.16, 0.175
    for x, width, label, fill, edge in row3:
        records.append(_box(ax, x, y3, width, h3, label,
                            facecolor=fill, edgecolor=edge, fontsize=6.2,
                            weight="bold" if label.startswith("8") else "normal"))
    for current, following in zip(row3[:-1], row3[1:]):
        _arrow(ax, (current[0] + current[1] + 0.004, y3 + h3 / 2),
               (following[0] - 0.004, y3 + h3 / 2))
    _arrow(ax, (0.115, cy0 - 0.004), (0.115, y3 + h3 + 0.004))

    ax.text(0.5, 0.075,
            "Base candidates  $B^L=\\frac{1}{2}(\\tilde{P}^L+\\tilde{P}^{R\\to L})$,  $B^R=\\frac{1}{2}(\\tilde{P}^R+\\tilde{P}^{L\\to R})$;  "
            "$\\alpha\\in[0,1]^{T\\times J\\times 1}$ is predicted per joint and frame.",
            ha="center", va="center", fontsize=7.0)
    ax.text(0.5, 0.03,
            "No camera intrinsics, extrinsics, or epipolar geometry are used at any stage.",
            ha="center", va="center", fontsize=7.0, color=JOURNAL_COLORS["grey"])
    return fig, records


def render_method_pipeline(output: Path) -> None:
    """Render the merged architecture and feature-construction figure."""
    fig, records = build_method_pipeline_figure()
    overflow_count = count_text_overflows(fig, records)
    if overflow_count:
        plt.close(fig)
        raise ValueError(f"Method-pipeline labels overflow {overflow_count} boxes")
    _save(fig, output)


def build_feature_details_figure():
    """Expand the five per-joint cues concatenated before the shared encoder."""
    apply_journal_style()
    plt.rcParams["mathtext.default"] = "regular"
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    records = []
    ax.text(0.5, 0.97, "Geometric-motion features: left stream",
            ha="center", va="top", fontsize=10, weight="bold")
    cues = (
        ("Own-view pose\n$\\tilde{P}^{L}$\n3 channels", "#DCEEFF"),
        ("Aligned other pose\n$\\tilde{P}^{R\\to L}$\n3 channels", "#DFF3EF"),
        ("Pose difference\n$D^L = \\tilde{P}^{R\\to L}-\\tilde{P}^{L}$\n3 channels", "#FFF0DF"),
        ("Temporal velocity\n$V^L$\n3 channels", "#FBE6E2"),
        ("Alignment rotation\n$r^{R\\to L}$\n3 channels", "#F2E7F5"),
    )
    for i, (label, color) in enumerate(cues):
        x = 0.01 + 0.20 * i
        records.append(_box(ax, x, 0.64, 0.18, 0.23, label,
                            facecolor=color, fontsize=7.1))
        _arrow(ax, (x + 0.09, 0.63), (x + 0.09, 0.51))
    records.append(_box(ax, 0.025, 0.35, 0.95, 0.15,
        "Channel concatenation at each joint and frame:  3 + 3 + 3 + 3 + 3 = 15",
        facecolor="#F3F7F9", fontsize=8.2))
    _arrow(ax, (0.50, 0.34), (0.50, 0.25))
    records.append(_box(ax, 0.26, 0.11, 0.48, 0.13,
        "$X^L$  (15 channels)  →  Shared joint encoder",
        facecolor="#E8F1FB", fontsize=8.2))
    ax.text(0.5, 0.025, "The right stream uses the same construction with L and R exchanged.",
            ha="center", va="center", fontsize=7.8)
    return fig, records


def render_feature_details(output: Path) -> None:
    fig, records = build_feature_details_figure()
    overflow_count = count_text_overflows(fig, records)
    if overflow_count:
        plt.close(fig)
        raise ValueError(f"Feature-detail labels overflow {overflow_count} boxes")
    _save(fig, output)


def _load_dataset_frame(path: Path) -> np.ndarray:
    """Load an RGB frame from an image file or a SAM3D result archive."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset-example frame is missing: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "frame" not in archive.files:
                raise KeyError(f"SAM3D archive has no 'frame' array: {path}")
            frame = np.asarray(archive["frame"]).copy()
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"Expected an HxWx3/4 frame in {path}, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.asarray(Image.fromarray(frame).convert("RGB"))
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _content_window(
    frame: np.ndarray,
    *,
    target_aspect: float = 16 / 9,
    padding: float = 0.22,
) -> tuple[int, int, int, int] | None:
    """Return (top, left, height, width) of the foreground crop, or None."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected an RGB-like frame, got {frame.shape}")
    foreground = np.max(frame[..., :3], axis=2) > 8
    if not np.any(foreground):
        return None
    rows, columns = np.nonzero(foreground)
    content_height = rows.max() - rows.min() + 1
    content_width = columns.max() - columns.min() + 1
    crop_height = int(np.ceil(content_height * (1 + 2 * padding)))
    crop_width = int(np.ceil(content_width * (1 + 2 * padding)))
    if crop_width / crop_height < target_aspect:
        crop_width = int(np.ceil(crop_height * target_aspect))
    else:
        crop_height = int(np.ceil(crop_width / target_aspect))
    height, width = frame.shape[:2]
    if crop_width > width:
        crop_width = width
        crop_height = min(height, int(round(crop_width / target_aspect)))
    if crop_height > height:
        crop_height = height
        crop_width = min(width, int(round(crop_height * target_aspect)))
    center_x = (columns.min() + columns.max() + 1) / 2
    center_y = (rows.min() + rows.max() + 1) / 2
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    left = min(max(left, 0), width - crop_width)
    top = min(max(top, 0), height - crop_height)
    return top, left, crop_height, crop_width


def _crop_to_content(
    frame: np.ndarray,
    *,
    target_aspect: float = 16 / 9,
    padding: float = 0.22,
) -> np.ndarray:
    """Crop a dark-background render around its foreground at a fixed aspect."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected an RGB-like frame, got {frame.shape}")
    foreground = np.max(frame[..., :3], axis=2) > 8
    if not np.any(foreground):
        return frame
    rows, columns = np.nonzero(foreground)
    content_height = rows.max() - rows.min() + 1
    content_width = columns.max() - columns.min() + 1
    crop_height = int(np.ceil(content_height * (1 + 2 * padding)))
    crop_width = int(np.ceil(content_width * (1 + 2 * padding)))
    if crop_width / crop_height < target_aspect:
        crop_width = int(np.ceil(crop_height * target_aspect))
    else:
        crop_height = int(np.ceil(crop_width / target_aspect))

    height, width = frame.shape[:2]
    if crop_width > width:
        crop_width = width
        crop_height = min(height, int(round(crop_width / target_aspect)))
    if crop_height > height:
        crop_height = height
        crop_width = min(width, int(round(crop_height * target_aspect)))

    center_x = (columns.min() + columns.max() + 1) / 2
    center_y = (rows.min() + rows.max() + 1) / 2
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    left = min(max(left, 0), width - crop_width)
    top = min(max(top, 0), height - crop_height)
    return frame[top : top + crop_height, left : left + crop_width]


def _validate_view_pair(name: str, paths) -> tuple[Path, Path]:
    pair = tuple(Path(path) for path in paths)
    if len(pair) != 2:
        raise ValueError(f"{name} requires exactly two synchronized views")
    return pair


def build_dataset_examples_figure(unity_views, ski_ptz_views, author_views):
    """Build the two-view by three-source dataset montage."""
    apply_journal_style()
    sources = (
        ("Unity synthetic", _validate_view_pair("Unity", unity_views)),
        ("Ski-PTZ-Pose", _validate_view_pair("Ski-PTZ-Pose", ski_ptz_views)),
        (
            "Author-collected videos",
            _validate_view_pair("Author-collected videos", author_views),
        ),
    )
    roles = (
        "Quantitative\nAvatar 3D ground truth",
        "Quantitative\nFused canonical reference",
        "Qualitative only\nNo metric 3D reference",
    )
    accents = (
        JOURNAL_COLORS["blue"],
        JOURNAL_COLORS["teal"],
        JOURNAL_COLORS["grey"],
    )
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 3.9),
        gridspec_kw={"hspace": 0.08, "wspace": 0.035},
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.89, bottom=0.18)

    panel_index = 0
    for column, (title, pair) in enumerate(sources):
        axes[0, column].set_title(title, fontsize=9, weight="bold", pad=4)
        for row, path in enumerate(pair):
            ax = axes[row, column]
            ax.set_box_aspect(9 / 16)
            ax.set_facecolor("#F4F4F4")
            frame = _load_dataset_frame(path)
            if column == 0:
                frame = _crop_to_content(frame)
            ax.imshow(frame, interpolation="antialiased")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(JOURNAL_COLORS["light_grey"])
                spine.set_linewidth(0.8)
            ax.text(
                0.02,
                0.96,
                panel_labels[panel_index],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                weight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 1.4,
                },
            )
            panel_index += 1
    axes[0, 0].set_ylabel("View 1", fontsize=8.5, weight="bold", labelpad=5)
    axes[1, 0].set_ylabel("View 2", fontsize=8.5, weight="bold", labelpad=5)

    for column, (role, accent) in enumerate(zip(roles, accents)):
        position = axes[1, column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.07,
            role,
            ha="center",
            va="center",
            fontsize=7.3,
            weight="bold",
            color=accent,
            linespacing=1.15,
        )
    return fig


def render_dataset_examples(
    unity_views,
    ski_ptz_views,
    author_views,
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Render representative synchronized views from the three data sources."""
    fig = build_dataset_examples_figure(
        unity_views,
        ski_ptz_views,
        author_views,
    )
    output_pdf = Path(output_pdf)
    output_png = Path(output_png)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


OCCLUSION_EXAMPLE_SOURCE = {
    "person_id": "female",
    "action_id": "Anim_Female_Skier_Skiing",
    "cameras": ("capture_L3_A000", "capture_L3_A210"),
    "root": DATA_ROOT / "skiing_unity_dataset" / "data_pole_ski" / "female" / "Anim_Female_Skier_Skiing",
}
OCCLUSION_PATTERNS = ("random", "distal", "temporal")


def _occlusion_example_frame(setting_cls, key_cls, mask_fn, *, ratio: float,
                             frame_ids: list[int]) -> int:
    """Pick the frame where the temporal pattern masks the most joints in both views."""
    source = OCCLUSION_EXAMPLE_SOURCE
    setting = setting_cls(pattern="temporal", ratio=ratio)
    best_frame, best_count = frame_ids[0], -1
    for frame_id in frame_ids:
        total = 0
        for camera in source["cameras"]:
            key = key_cls(source["person_id"], source["action_id"], camera, frame_id)
            total += int(mask_fn(setting, key, source_frame_ids=frame_ids).sum())
        if total > best_count:
            best_frame, best_count = frame_id, total
    return best_frame


def build_image_occlusion_examples_figure(*, ratio: float = 0.5):
    """Render a 2 x 3 panel: rows are the two views, columns the three mask patterns."""
    from dual2pose.eval.image_occlusion import (
        ImageOcclusionSetting,
        OcclusionFrameKey,
        apply_image_occlusion,
        selected_joint_mask,
    )
    from dual2pose.map_config import filter_unity_kpts

    apply_journal_style()
    source = OCCLUSION_EXAMPLE_SOURCE
    root = Path(source["root"])
    first_camera = source["cameras"][0]
    frame_ids = sorted(
        int(path.stem.split("_")[-1])
        for path in (root / "frames" / first_camera).glob("frame_*.png")
    )
    if not frame_ids:
        raise FileNotFoundError(f"No Unity frames found under {root / 'frames' / first_camera}")
    frame_id = _occlusion_example_frame(
        ImageOcclusionSetting, OcclusionFrameKey, selected_joint_mask,
        ratio=ratio, frame_ids=frame_ids,
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.05))
    letters = "abcdef"
    for row, camera in enumerate(source["cameras"]):
        image_path = root / "frames" / camera / f"frame_{frame_id:06d}.png"
        joint_path = root / "kpt2d" / camera.replace("capture_", "") / "character" / f"kpt2d_{frame_id:06d}.npy"
        frame = _load_dataset_frame(image_path)
        raw_joints = np.asarray(np.load(joint_path, allow_pickle=False), dtype=np.float32)
        joints = filter_unity_kpts(raw_joints, flag="2d", gender=source["person_id"])
        window = _content_window(frame, target_aspect=4 / 3, padding=0.10)
        key = OcclusionFrameKey(source["person_id"], source["action_id"], camera, frame_id)
        for column, pattern in enumerate(OCCLUSION_PATTERNS):
            setting = ImageOcclusionSetting(pattern=pattern, ratio=ratio)
            masked, record = apply_image_occlusion(
                frame, joints, setting, key, source_frame_ids=frame_ids,
            )
            ax = axes[row, column]
            if window is None:
                top = left = 0
                view = masked
            else:
                top, left, height, width = window
                view = masked[top : top + height, left : left + width]
            ax.imshow(view)
            for x0, y0, x1, y1 in record["boxes_xyxy"]:
                ax.add_patch(
                    plt.Rectangle(
                        (x0 - left - 0.5, y0 - top - 0.5), x1 - x0, y1 - y0,
                        fill=False, edgecolor=JOURNAL_COLORS["red"], linewidth=0.7,
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor(JOURNAL_COLORS["grey"])
            ax.text(0.02, 0.96, f"({letters[row * 3 + column]})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, weight="bold", color="white")
            if row == 0:
                ax.set_title(
                    f"{pattern.capitalize()} (ratio {ratio:g}), {record['selected_joint_count']} joints",
                    fontsize=8,
                )
            else:
                ax.set_xlabel(f"{record['selected_joint_count']} joints masked", fontsize=7.5)
        axes[row, 0].set_ylabel(f"View {row + 1}", fontsize=8.5, weight="bold")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.08, wspace=0.05, hspace=0.16)
    return fig, {"frame_id": frame_id, "ratio": ratio, "cameras": source["cameras"],
                 "action_id": source["action_id"]}


def render_image_occlusion_examples(output: Path, *, ratio: float = 0.5) -> dict:
    fig, meta = build_image_occlusion_examples_figure(ratio=ratio)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return meta


def _crops(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width = image.shape[:2]
    left_frame = image[int(0.055 * height) : int(0.405 * height), : int(0.395 * width)]
    right_frame = image[int(0.50 * height) : int(0.86 * height), : int(0.395 * width)]
    pose_comparison = image[int(0.015 * height) : int(0.965 * height), int(0.39 * width) :]
    return left_frame, right_frame, pose_comparison


def render_realworld_qualitative(outdoor: Path, indoor: Path, output: Path) -> None:
    """Recompose the two qualitative outputs into a journal-specific panel layout."""
    apply_journal_style()
    rows = ((_crops(outdoor), "Outdoor"), (_crops(indoor), "Indoor"))
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.4),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.72), "hspace": 0.08, "wspace": 0.035},
    )
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    panel_index = 0
    for row_index, (panels, scene) in enumerate(rows):
        for column_index, panel in enumerate(panels):
            ax = axes[row_index, column_index]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(JOURNAL_COLORS["light_grey"])
                spine.set_linewidth(0.7)
            ax.text(
                0.02,
                0.96,
                panel_labels[panel_index],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )
            panel_index += 1
        axes[row_index, 0].set_ylabel(scene, fontsize=8.5, weight="bold", labelpad=5)
    axes[0, 0].set_title("Left view", fontsize=8.5, weight="bold", pad=3)
    axes[0, 1].set_title("Right view", fontsize=8.5, weight="bold", pad=3)
    axes[0, 2].set_title("3D pose comparison", fontsize=8.5, weight="bold", pad=3)
    _save(fig, output)


def main() -> None:
    output = PAPER / "figures" / "journal"
    original = PAPER / "figures" / "original" / "experiment3"
    dataset_sources = DATASET_EXAMPLE_SOURCES
    render_dataset_examples(
        dataset_sources["unity"],
        dataset_sources["ski_ptz"],
        dataset_sources["author"],
        output / "dataset_examples.pdf",
        output / "dataset_examples.png",
    )
    render_study_overview(output / "study_overview.pdf")
    render_method_pipeline(output / "method_pipeline.pdf")
    meta = render_image_occlusion_examples(output / "image_occlusion_examples.pdf")
    print("image occlusion examples:", meta)
    outdoor = original / "pro_1_frame0000_real_compare.png"
    indoor = original / "run_3_frame0000_real_compare.png"
    if outdoor.is_file() and indoor.is_file():
        # Legacy recomposition of the conference qualitative panels; the
        # manuscript now uses realworld_qualitative_refresh.pdf instead.
        render_realworld_qualitative(outdoor, indoor, output / "realworld_qualitative.pdf")
    else:
        print("skipping legacy realworld_qualitative.pdf: conference source panels are archived")
    print(output)


if __name__ == "__main__":
    main()
