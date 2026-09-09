"""Shared publication style for IVC experiment figures."""

from __future__ import annotations

import matplotlib as mpl


JOURNAL_COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#8A8A8A",
    "light_grey": "#E8E8E8",
    "black": "#222222",
    "white": "#FFFFFF",
}


def apply_journal_style() -> None:
    """Apply one colorblind-safe, print-readable style to every paper figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": JOURNAL_COLORS["black"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax, *, grid_axis: str = "y") -> None:
    """Add a restrained reference grid without relying on color alone."""
    ax.grid(axis=grid_axis, color=JOURNAL_COLORS["light_grey"], linewidth=0.6)
    ax.set_axisbelow(True)
