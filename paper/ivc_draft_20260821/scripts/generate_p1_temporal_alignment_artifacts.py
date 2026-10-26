#!/usr/bin/env python3
"""Generate P1 automatic temporal-alignment publication artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

try:
    from .journal_figure_style import JOURNAL_COLORS, apply_journal_style, style_axes
except ImportError:  # Direct script execution.
    from journal_figure_style import JOURNAL_COLORS, apply_journal_style, style_axes


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "logs" / "ivc_p1" / "temporal_alignment"
ROW_END = r"\\"
INJECTED_OFFSETS = (-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
CONDITIONS = ("uncorrected", "automatic", "oracle")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def validate_summary_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected = {(offset, condition) for offset in INJECTED_OFFSETS for condition in CONDITIONS}
    indexed: dict[tuple[float, str], dict[str, object]] = {}
    counts: set[int] = set()
    numeric_fields = (
        "mpjpe_mean",
        "mpjpe_std",
        "acceleration_error_mean",
        "degradation_vs_zero_reference",
        "recovery_fraction",
        "offset_mae",
        "offset_signed_bias",
        "offset_accuracy_exact",
        "offset_accuracy_within_0p5",
        "correction_activation_rate",
        "gate_error_correlation",
        "view_preference_accuracy",
    )
    for raw in rows:
        row = dict(raw)
        key = (float(row["injected_offset"]), str(row["condition"]))
        if key in indexed:
            raise ValueError(f"Duplicate temporal-alignment row: {key}")
        row["injected_offset"] = key[0]
        row["sample_count"] = int(row["sample_count"])
        counts.add(row["sample_count"])
        for field in numeric_fields:
            row[field] = float(row[field])
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(
            f"Temporal-alignment matrix mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError(f"Inconsistent temporal-alignment sample count: {sorted(counts)}")
    order = {condition: index for index, condition in enumerate(CONDITIONS)}
    return sorted(indexed.values(), key=lambda row: (float(row["injected_offset"]), order[str(row["condition"])]))


def _indexed(rows: Sequence[Mapping[str, object]]) -> dict[tuple[float, str], Mapping[str, object]]:
    return {(float(row["injected_offset"]), str(row["condition"])): row for row in rows}


def render_table(rows: Sequence[Mapping[str, object]]) -> str:
    indexed = _indexed(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Automatic temporal alignment under injected right-stream offsets, evaluated on the fixed central 20-frame interval. Oracle uses the original unperturbed right stream. Negative $\Delta_{\mathrm{auto}}$ means that automatic correction improves over the uncorrected input. The confidence threshold is selected on validation data only.}",
        r"\label{tab:temporal_alignment}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Offset & Uncorrected & Automatic & Oracle & $\Delta_{\mathrm{auto}}$ & Lag MAE & Exact (\%) & Active (\%) " + ROW_END,
        r"\midrule",
    ]
    for offset in INJECTED_OFFSETS:
        uncorrected = indexed[(offset, "uncorrected")]
        automatic = indexed[(offset, "automatic")]
        oracle = indexed[(offset, "oracle")]
        automatic_delta = float(automatic["mpjpe_mean"]) - float(uncorrected["mpjpe_mean"])
        lines.append(
            f"{offset:+g} & {float(uncorrected['mpjpe_mean']):.4f} & "
            f"{float(automatic['mpjpe_mean']):.4f} & {float(oracle['mpjpe_mean']):.4f} & "
            f"{automatic_delta:+.5f} & {float(automatic['offset_mae']):.3f} & "
            f"{100.0 * float(automatic['offset_accuracy_exact']):.1f} & "
            f"{100.0 * float(automatic['correction_activation_rate']):.1f} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def render_figure(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    indexed = _indexed(rows)
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    styles = {
        "uncorrected": ("s", JOURNAL_COLORS["grey"], "--", "Uncorrected"),
        "automatic": ("o", JOURNAL_COLORS["blue"], "-", "Automatic"),
        "oracle": ("^", JOURNAL_COLORS["orange"], ":", "Oracle"),
    }
    for condition, (marker, color, linestyle, label) in styles.items():
        ax.plot(
            INJECTED_OFFSETS,
            [float(indexed[(offset, condition)]["mpjpe_mean"]) for offset in INJECTED_OFFSETS],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    ax.axvline(0.0, color=JOURNAL_COLORS["black"], linewidth=0.8, linestyle="--")
    ax.set(xlabel="Injected right-stream offset (frames)", ylabel="MPJPE")
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    rows = validate_summary_rows(read_csv(INPUT_ROOT / "temporal_alignment_summary.csv"))
    table = PAPER / "tables" / "temporal_alignment.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(render_table(rows), encoding="utf-8")
    render_figure(rows, PAPER / "figures" / "extension" / "temporal_alignment.pdf")
    print(table)


if __name__ == "__main__":
    main()
