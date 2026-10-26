#!/usr/bin/env python3
"""Generate the complete journal front-end adaptation matrix table and heatmap."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual2pose.experiments.run_frontend_adaptation_matrix import build_evaluation_cells
from paper.ivc_draft_20260821.scripts.journal_figure_style import (
    JOURNAL_COLORS,
    apply_journal_style,
)


PAPER = Path(__file__).resolve().parents[1]
INPUT = ROOT / "logs" / "ivc_p1" / "frontend_adaptation" / "frontend_adaptation_matrix.csv"
ROW_END = r"\\"
MODEL_ORDER = (
    "mmsports",
    "videopose3d_heads_only",
    "videopose3d_full",
    "poseformer_heads_only",
    "poseformer_full",
    "motionbert_heads_only",
    "motionbert_full",
    "mixed_full",
)
FRONTEND_ORDER = ("sam3d", "videopose3d", "poseformer", "motionbert")
MODEL_LABELS = {
    "mmsports": "Frozen reference",
    "videopose3d_heads_only": "VideoPose3D/heads",
    "videopose3d_full": "VideoPose3D/full",
    "poseformer_heads_only": "PoseFormer/heads",
    "poseformer_full": "PoseFormer/full",
    "motionbert_heads_only": "MotionBERT/heads",
    "motionbert_full": "MotionBERT/full",
    "mixed_full": "Mixed/full",
}
FRONTEND_LABELS = {
    "sam3d": "SAM3D",
    "videopose3d": "VideoPose3D",
    "poseformer": "PoseFormer",
    "motionbert": "MotionBERT",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def validate_matrix_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected = {(cell.model_name, cell.test_frontend) for cell in build_evaluation_cells()}
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for raw in rows:
        row = dict(raw)
        key = (str(row["model_name"]), str(row["test_frontend"]))
        if key in indexed:
            raise ValueError(f"Duplicate front-end adaptation row: {key}")
        row["common13_mpjpe"] = float(row["common13_mpjpe"])
        row["sample_count"] = int(row["sample_count"])
        if row["sample_count"] != 64_440:
            raise ValueError(f"Every evaluation cell must contain all 64,440 test camera-pair sequences: {key}")
        for field in (
            "common13_acceleration_error",
            "recovery_vs_mmsports",
            "relative_recovery_percent",
            "sam3d_retention_delta",
        ):
            row[field] = float(row[field])
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(
            f"Front-end adaptation matrix mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    frontend_rank = {name: index for index, name in enumerate(FRONTEND_ORDER)}
    return sorted(indexed.values(), key=lambda row: (
        model_rank[str(row["model_name"])], frontend_rank[str(row["test_frontend"])]
    ))


def _indexed(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], Mapping[str, object]]:
    return {(str(row["model_name"]), str(row["test_frontend"])): row for row in rows}


def render_table(rows: Sequence[Mapping[str, object]]) -> str:
    indexed = _indexed(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Common-13 MPJPE after front-end-specific or mixed adaptation. Each row is one frozen model evaluated on all four front ends; lower is better.}",
        r"\label{tab:frontend_adaptation}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Adaptation model & SAM3D & VideoPose3D & PoseFormer & MotionBERT " + ROW_END,
        r"\midrule",
    ]
    for model_name in MODEL_ORDER:
        values = [float(indexed[(model_name, frontend)]["common13_mpjpe"]) for frontend in FRONTEND_ORDER]
        lines.append(
            f"{MODEL_LABELS[model_name]} & " + " & ".join(f"{value:.4f}" for value in values) + " " + ROW_END
        )
        if model_name == "mmsports":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def render_figure(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    indexed = _indexed(rows)
    apply_journal_style()
    matrix = np.asarray(
        [[float(indexed[(model, frontend)]["relative_recovery_percent"]) for frontend in FRONTEND_ORDER]
         for model in MODEL_ORDER],
        dtype=np.float64,
    )
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    diverging = LinearSegmentedColormap.from_list(
        "journal_diverging",
        [JOURNAL_COLORS["blue"], JOURNAL_COLORS["white"], JOURNAL_COLORS["orange"]],
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.imshow(matrix, cmap=diverging, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(FRONTEND_ORDER)), [FRONTEND_LABELS[name] for name in FRONTEND_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[name] for name in MODEL_ORDER])
    ax.set_xticks(np.arange(-0.5, len(FRONTEND_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color=JOURNAL_COLORS["white"], linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = JOURNAL_COLORS["white"] if abs(value) > 0.55 * limit else JOURNAL_COLORS["black"]
            ax.text(
                column_index, row_index, f"{value:+.1f}",
                ha="center", va="center", fontsize=8, color=text_color,
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Recovery vs. frozen reference (%)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    rows = validate_matrix_rows(read_csv(INPUT))
    table = PAPER / "tables" / "frontend_adaptation.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(render_table(rows), encoding="utf-8")
    render_figure(rows, PAPER / "figures" / "extension" / "frontend_adaptation.pdf")
    print(table)


if __name__ == "__main__":
    main()
