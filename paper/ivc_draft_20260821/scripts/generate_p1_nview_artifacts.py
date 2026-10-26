#!/usr/bin/env python3
"""Generate the P1 N-view scaling table and figure from verified CSV output."""

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
INPUT_ROOT = ROOT / "logs" / "ivc_p1" / "nview"
ROW_END = r"\\"

EXPECTED_KEYS = {(1, "single_view")} | {
    (n_views, method)
    for n_views in (2, 3, 4)
    for method in (
        "nview_canonical_mean",
        "pairwise_canonfuse_mean",
        "pairwise_oracle_select",
    )
}
METHOD_LABELS = {
    "single_view": "Single view",
    "nview_canonical_mean": "Canonical mean",
    "pairwise_canonfuse_mean": "Pairwise CanonFuse3D",
    "pairwise_oracle_select": "Pairwise oracle (upper bound)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_summary_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    indexed: dict[tuple[int, str], dict[str, object]] = {}
    counts: set[int] = set()
    for raw in rows:
        row = dict(raw)
        key = (int(row["n_views"]), str(row["method"]))
        if key in indexed:
            raise ValueError(f"Duplicate N-view summary row: {key}")
        row["n_views"] = key[0]
        row["group_count"] = int(row["group_count"])
        for field in (
            "mpjpe_mean",
            "mpjpe_std",
            "mpjpe_ci95_low",
            "mpjpe_ci95_high",
        ):
            row[field] = float(row[field])
        if not row["mpjpe_ci95_low"] <= row["mpjpe_mean"] <= row["mpjpe_ci95_high"]:
            raise ValueError(f"N-view confidence interval does not contain its mean: {key}")
        if row["group_count"] <= 0:
            raise ValueError("N-view group count must be positive")
        counts.add(row["group_count"])
        oracle = key[1] == "pairwise_oracle_select"
        if _is_true(row.get("upper_bound", False)) != oracle:
            raise ValueError("Pairwise oracle rows must be labeled as upper bound only")
        indexed[key] = row
    if set(indexed) != EXPECTED_KEYS:
        raise ValueError(
            f"N-view matrix mismatch; missing={sorted(EXPECTED_KEYS - set(indexed))}, "
            f"extra={sorted(set(indexed) - EXPECTED_KEYS)}"
        )
    if len(counts) != 1:
        raise ValueError(f"Inconsistent accepted group counts: {sorted(counts)}")
    order = {"single_view": 0, "nview_canonical_mean": 1,
             "pairwise_canonfuse_mean": 2, "pairwise_oracle_select": 3}
    return sorted(indexed.values(), key=lambda row: (int(row["n_views"]), order[str(row["method"])]))


def render_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{N-view scaling by frozen pairwise composition. The oracle selects one complete pair per camera group and is reported only as an upper bound.}",
        r"\label{tab:nview_scaling}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"Views & Method & Groups & MPJPE & SD & 95\% bootstrap CI " + ROW_END,
        r"\midrule",
    ]
    previous_n = None
    for row in rows:
        n_views = int(row["n_views"])
        if previous_n is not None and n_views != previous_n:
            lines.append(r"\midrule")
        lines.append(
            f"{n_views} & {METHOD_LABELS[str(row['method'])]} & "
            f"{int(row['group_count'])} & {float(row['mpjpe_mean']):.4f} & "
            f"{float(row['mpjpe_std']):.4f} & "
            f"[{float(row['mpjpe_ci95_low']):.4f}, {float(row['mpjpe_ci95_high']):.4f}] "
            + ROW_END
        )
        previous_n = n_views
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.3em}",
            r"\begin{minipage}{0.96\linewidth}\footnotesize Upper bound rows are diagnostic and are not a deployable method.\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_efficiency_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    indexed: dict[int, dict[str, object]] = {}
    shared_metadata: set[tuple[object, ...]] = set()
    for raw in rows:
        row = dict(raw)
        n_views = int(row["n_views"])
        if n_views in indexed:
            raise ValueError(f"Duplicate N-view efficiency row: {n_views}")
        row["n_views"] = n_views
        row["pair_forward_count"] = int(row["pair_forward_count"])
        row["group_count"] = int(row["group_count"])
        row["warmup_iterations"] = int(row["warmup_iterations"])
        row["sequence_frames"] = int(row["sequence_frames"])
        for field in (
            "mpjpe_mean",
            "latency_mean_ms",
            "latency_std_ms",
            "latency_median_ms",
            "latency_p95_ms",
            "relative_latency_vs_two_view",
            "throughput_groups_per_second",
            "peak_gpu_memory_mib",
            "mpjpe_reduction_vs_two_view_pct",
        ):
            row[field] = float(row[field])
        expected_pairs = n_views * (n_views - 1) // 2
        if row["pair_forward_count"] != expected_pairs:
            raise ValueError(
                f"N={n_views} pair count mismatch: {row['pair_forward_count']}"
            )
        if row["group_count"] <= 0 or row["latency_mean_ms"] <= 0.0:
            raise ValueError("Efficiency group counts and latency must be positive")
        shared_metadata.add(
            (
                row["group_count"],
                row["warmup_iterations"],
                row["execution_mode"],
                row["device_name"],
                row["torch_version"],
                row["cuda_version"],
                row["sequence_frames"],
            )
        )
        indexed[n_views] = row
    if set(indexed) != {2, 3, 4}:
        raise ValueError(
            f"N-view efficiency views mismatch; expected=[2, 3, 4], "
            f"actual={sorted(indexed)}"
        )
    if len(shared_metadata) != 1:
        raise ValueError("N-view efficiency metadata must be consistent across views")
    if abs(float(indexed[2]["relative_latency_vs_two_view"]) - 1.0) > 1e-6:
        raise ValueError("Two-view relative latency must equal 1")
    return [indexed[n_views] for n_views in (2, 3, 4)]


def _latex_escape(value: object) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_")


def render_efficiency_table(rows: Sequence[Mapping[str, object]]) -> str:
    first = rows[0]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Accuracy--cost trade-off for deployable frozen pairwise composition.}",
        r"\label{tab:nview_efficiency}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Views & Pairs & MPJPE & Reduction vs. 2 views & Serial latency (ms) & Relative latency & Groups/s & Peak alloc. (MiB) " + ROW_END,
        r"\midrule",
    ]
    for row in rows:
        reduction = (
            "--"
            if int(row["n_views"]) == 2
            else f"{float(row['mpjpe_reduction_vs_two_view_pct']):.1f}\%"
        )
        relative_latency = float(row["relative_latency_vs_two_view"])
        lines.append(
            f"{int(row['n_views'])} & {int(row['pair_forward_count'])} & "
            f"{float(row['mpjpe_mean']):.4f} & {reduction} & "
            f"{float(row['latency_mean_ms']):.2f} & "
            f"{relative_latency:.2f}" + chr(36) + chr(92) + "times" + chr(36) + " & "
            f"{float(row['throughput_groups_per_second']):.1f} & "
            f"{float(row['peak_gpu_memory_mib']):.2f} " + ROW_END
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.3em}",
            r"\begin{minipage}{0.97\linewidth}\footnotesize "
            + f"Serial latency is the synchronized mean over {int(first['group_count'])} accepted "
            + f"{int(first['sequence_frames'])}-frame camera groups after "
            + f"{int(first['warmup_iterations'])} warm-up executions on one "
            + f"{_latex_escape(first['device_name'])}. It includes pose canonicalization, all unique "
            + r"pair forwards, and output averaging, but excludes monocular front-end inference, data loading, and oracle selection."
            + r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_figure(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    styles = {
        "nview_canonical_mean": ("s", JOURNAL_COLORS["grey"], "--", "Canonical mean"),
        "pairwise_canonfuse_mean": ("o", JOURNAL_COLORS["blue"], "-", "Pairwise CanonFuse3D"),
        "pairwise_oracle_select": ("^", JOURNAL_COLORS["orange"], ":", "Pairwise oracle (upper bound)"),
    }
    single = next(row for row in rows if row["method"] == "single_view")
    ax.scatter(
        [1], [float(single["mpjpe_mean"])], marker="D",
        color=JOURNAL_COLORS["black"], label="Single view", zorder=3,
    )
    for method, (marker, color, linestyle, label) in styles.items():
        subset = [row for row in rows if row["method"] == method]
        ax.errorbar(
            [int(row["n_views"]) for row in subset],
            [float(row["mpjpe_mean"]) for row in subset],
            yerr=[
                [float(row["mpjpe_mean"]) - float(row["mpjpe_ci95_low"]) for row in subset],
                [float(row["mpjpe_ci95_high"]) - float(row["mpjpe_mean"]) for row in subset],
            ],
            marker=marker,
            color=color,
            linestyle=linestyle,
            capsize=3,
            label=label,
        )
    ax.set(xlabel="Number of cameras", ylabel="MPJPE", xticks=[1, 2, 3, 4])
    style_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    rows = validate_summary_rows(read_csv(INPUT_ROOT / "nview_summary.csv"))
    efficiency_rows = validate_efficiency_rows(
        read_csv(INPUT_ROOT / "nview_efficiency.csv")
    )
    table = PAPER / "tables" / "nview_scaling.tex"
    efficiency_table = PAPER / "tables" / "nview_efficiency.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(render_table(rows), encoding="utf-8")
    efficiency_table.write_text(
        render_efficiency_table(efficiency_rows), encoding="utf-8"
    )
    render_figure(rows, PAPER / "figures" / "extension" / "nview_scaling.pdf")
    print(table)
    print(efficiency_table)


if __name__ == "__main__":
    main()
