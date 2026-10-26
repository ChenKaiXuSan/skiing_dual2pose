"""Build the Elsevier graphical abstract for the IVC submission.

Elsevier requires a separate image of at least 531 x 1328 px (h x w) that stays
readable at 5 x 13 cm. This script lays out a 13 x 5 cm canvas with three
columns (dual-view input, CanonFuse3D, refined output) and reuses the skeleton
renderings from figures/journal/study_overview_visio.pdf so the visual language
matches Figure 1. Run from the repository root:

    python3 -m paper.ivc_draft_20260821.scripts.generate_graphical_abstract
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from .journal_figure_style import JOURNAL_COLORS, apply_journal_style  # noqa: E402

PAPER_ROOT = Path(__file__).resolve().parents[1]
OVERVIEW_PDF = PAPER_ROOT / "figures" / "journal" / "study_overview_visio.pdf"
OUTPUT_STEM = PAPER_ROOT / "graphical_abstract"

CANVAS_CM = (13.0, 5.0)
RENDER_DPI = 300
# Crop boxes in points of the overview PDF page (1553.9 x 558.7 pt), measured
# on the rendered page; the source page is 80 dpi-equivalent to 1727 x 621 px.
CROPS_80DPI = {
    "left": (45, 333, 205, 478),
    "right": (212, 333, 352, 478),
    "fused": (1470, 225, 1670, 470),
}


def _crop_skeletons(tmp: Path) -> dict[str, Image.Image]:
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(RENDER_DPI), "-singlefile", str(OVERVIEW_PDF), str(tmp / "overview")],
        check=True,
    )
    page = Image.open(tmp / "overview.png")
    scale = RENDER_DPI / 80
    return {
        key: page.crop(tuple(int(v * scale) for v in box)) for key, box in CROPS_80DPI.items()
    }


def _place_image(fig, image: Image.Image, box: tuple[float, float, float, float]) -> None:
    ax = fig.add_axes(box)
    ax.imshow(image)
    ax.axis("off")


def _arrow(fig, x0: float, x1: float, y: float) -> None:
    fig.patches.append(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            color=JOURNAL_COLORS["grey"],
        )
    )


def build(output_stem: Path = OUTPUT_STEM) -> list[Path]:
    apply_journal_style()
    with tempfile.TemporaryDirectory() as tmpdir:
        skel = _crop_skeletons(Path(tmpdir))

    width_in, height_in = (c / 2.54 for c in CANVAS_CM)
    # Elsevier prefers Arial/Helvetica; Liberation Sans and Nimbus Sans are their metric clones.
    rc = {
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(width_in, height_in))
        fig.patch.set_facecolor("white")
        black, grey = JOURNAL_COLORS["black"], JOURNAL_COLORS["grey"]
        orange = JOURNAL_COLORS["orange"]

        fig.text(
            0.5, 0.93,
            "CanonFuse3D: calibration-free fusion of two monocular 3D pose streams",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=black,
        )

        # Column 1: dual-view input.
        fig.text(0.16, 0.80, "Two monocular 3D pose streams", ha="center", va="center", fontsize=6.0, fontweight="bold", color=black)
        _place_image(fig, skel["left"], (0.04, 0.33, 0.115, 0.42))
        _place_image(fig, skel["right"], (0.17, 0.33, 0.115, 0.42))
        fig.text(0.0975, 0.28, "view L", ha="center", va="center", fontsize=5.4, color=JOURNAL_COLORS["blue"])
        fig.text(0.2275, 0.28, "view R", ha="center", va="center", fontsize=5.4, color=JOURNAL_COLORS["teal"])
        fig.text(
            0.16, 0.13,
            "approximately synchronized; no images,\n2D keypoints, or camera parameters at fusion",
            ha="center", va="center", fontsize=5.0, color=grey, linespacing=1.3,
        )

        _arrow(fig, 0.305, 0.34, 0.53)

        # Column 2: the fusion model.
        fig.patches.append(
            FancyBboxPatch(
                (0.35, 0.10), 0.30, 0.70,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                transform=fig.transFigure, linewidth=0.9,
                edgecolor=orange, facecolor="#FFF6EE",
            )
        )
        fig.text(0.50, 0.72, "CanonFuse3D", ha="center", va="center", fontsize=7.0, fontweight="bold", color=orange)
        steps = [
            (0.59, "1  Human-centric canonicalization"),
            (0.48, "2  Cross-view pose alignment"),
            (0.37, "3  Cross-view attention +"),
            (0.305, "    temporal refinement"),
            (0.195, "4  Joint-frame gate + residual"),
        ]
        for y, step in steps:
            fig.text(0.365, y, step, ha="left", va="center", fontsize=5.6, color=black)

        _arrow(fig, 0.66, 0.695, 0.53)

        # Column 3: refined output and headline result.
        fig.text(0.845, 0.80, "Refined canonical 3D pose", ha="center", va="center", fontsize=6.0, fontweight="bold", color=black)
        _place_image(fig, skel["fused"], (0.705, 0.30, 0.105, 0.46))
        fig.text(0.89, 0.63, "MPJPE vs. canonical averaging", ha="center", va="center", fontsize=5.0, color=grey)
        fig.text(0.89, 0.52, "\u221244.5%  Unity", ha="center", va="center", fontsize=6.6, fontweight="bold", color=black)
        fig.text(0.89, 0.41, "\u221248.6%  Ski-PTZ-Pose", ha="center", va="center", fontsize=6.6, fontweight="bold", color=black)
        fig.text(
            0.845, 0.14,
            "robust to camera separation,\ntiming error, and occlusion;\nadapts to other pose estimators;\nextends to 3\u20134 cameras",
            ha="center", va="center", fontsize=5.0, color=grey, linespacing=1.25,
        )

        outputs = []
        for suffix in ("pdf", "png"):
            target = output_stem.with_suffix(f".{suffix}")
            fig.savefig(target, dpi=RENDER_DPI, facecolor="white")
            outputs.append(target)
        plt.close(fig)
    return outputs


if __name__ == "__main__":
    for path in build():
        print(path)
