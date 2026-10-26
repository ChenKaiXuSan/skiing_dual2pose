#!/usr/bin/env python3
"""Generate IVC tables, figures, combined masking CSV, and evidence manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .journal_figure_style import JOURNAL_COLORS, apply_journal_style, style_axes
except ImportError:  # Direct script execution.
    from journal_figure_style import JOURNAL_COLORS, apply_journal_style, style_axes


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "ivc_mmsports_extension"
EXPECTED_CHECKPOINT_SHA256 = "869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f"
EXPECTED_SAMPLE_COUNT = 64440
ROW_END = r"\\"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(rows: Iterable[dict[str, str]], *, per_bin: bool = False) -> None:
    for row in rows:
        if row["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("Unexpected checkpoint hash")
        count_key = "total_sample_count" if per_bin else "sample_count"
        if int(row[count_key]) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(f"Unexpected {count_key}: {row[count_key]}")


def write_tex(name: str, lines: list[str]) -> None:
    path = PAPER / "tables" / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def native_reference_table(rows: list[dict[str, str]]) -> None:
    expanded = PAPER / "evidence/results/main_baselines__results.json"
    if expanded.exists():
        from dual2pose.eval.render_main_baseline_tables import render_package
        render_package(json.loads(expanded.read_text())["records"], PAPER)
        return
    native = next(row for row in rows if row["frontend_name"] == "sam3d")
    all15_gain = (
        number(native, "fusion_gain_mpjpe")
        / number(native, "canonical_avg_mpjpe")
        * 100.0
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Primary held-out Unity result with the native SAM3D front end. Lower MPJPE is better; reduction is relative to canonical averaging.}",
        r"\label{tab:native_reference}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Joint set & Canon. avg. & CanonFuse3D & Reduction (\%) " + ROW_END,
        r"\midrule",
        fr"All 15 & {number(native, 'canonical_avg_mpjpe'):.4f} & "
        fr"{number(native, 'fused_mpjpe'):.4f} & {all15_gain:.1f} " + ROW_END,
        fr"Common 13 & {number(native, 'common13_canonical_avg_mpjpe'):.4f} & "
        fr"{number(native, 'common13_fused_mpjpe'):.4f} & "
        fr"{number(native, 'fusion_gain_common13_percent'):.1f} " + ROW_END,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write_tex("native_reference.tex", lines)


def frontend_table(rows: list[dict[str, str]]) -> None:
    order = {"sam3d": 0, "motionbert": 1, "poseformer": 2, "videopose3d": 3}
    rows = sorted(
        (row for row in rows if row["frontend_name"] != "sam3d"),
        key=lambda row: order[row["frontend_name"]],
    )
    labels = {
        "motionbert": "MotionBERT",
        "poseformer": "PoseFormer",
        "videopose3d": "VideoPose3D",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen transfer to alternative upstream estimators on the common 13-joint subset. Lower MPJPE is better. Gain is relative to canonical averaging; positive values indicate improvement.}",
        r"\label{tab:frontend_generalization}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Front end & Canon. avg. & Fused & Gain (\%) & Gate pref. (\%) " + ROW_END,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            fr"{labels[row['frontend_name']]} & "
            fr"{number(row, 'common13_canonical_avg_mpjpe'):.4f} & "
            fr"{number(row, 'common13_fused_mpjpe'):.4f} & "
            fr"{number(row, 'fusion_gain_common13_percent'):+.1f} & "
            fr"{100.0 * number(row, 'gate_preference_accuracy'):.1f} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_tex("frontend_generalization.tex", lines)


def temporal_sampling_table(
    temporal: list[dict[str, str]], sampling: list[dict[str, str]]
) -> None:
    zero_t = next(row for row in temporal if number(row, "offset_frames") == 0.0)
    selected_t = [
        row for row in temporal if number(row, "offset_frames") in {-2.0, -1.0, 0.0, 1.0, 2.0}
    ]
    selected_s = [
        row for row in sampling if number(row, "rate_error_percent") in {-2.0, 0.0, 2.0}
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sensitivity to temporal offset and sampling-rate drift on all 15 joints. Relative change is measured against the unperturbed fused MPJPE.}",
        r"\label{tab:temporal_sampling}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Study & Perturbation & Canon. avg. & Fused & Rel. change (\%) & Gate corr. & Gate pref. (\%) " + ROW_END,
        r"\midrule",
    ]
    for row in selected_t:
        relative = 100.0 * (
            number(row, "fused_mpjpe") - number(zero_t, "fused_mpjpe")
        ) / number(zero_t, "fused_mpjpe")
        value = number(row, "offset_frames")
        lines.append(
            fr"Offset & {value:+g} frames & {number(row, 'canonical_avg_mpjpe'):.4f} & "
            fr"{number(row, 'fused_mpjpe'):.4f} & {relative:+.2f} & "
            fr"{number(row, 'gate_error_corr'):.3f} & "
            fr"{100.0 * number(row, 'gate_preference_accuracy'):.1f} " + ROW_END
        )
    lines.append(r"\midrule")
    for row in selected_s:
        value = number(row, "rate_error_percent")
        lines.append(
            fr"Rate drift & {value:+g}\% & {number(row, 'canonical_avg_mpjpe'):.4f} & "
            fr"{number(row, 'fused_mpjpe'):.4f} & "
            fr"{number(row, 'relative_mpjpe_degradation_percent'):+.2f} & "
            fr"{number(row, 'gate_error_corr'):.3f} & "
            fr"{100.0 * number(row, 'gate_preference_accuracy'):.1f} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    write_tex("temporal_sampling.tex", lines)


def view_table(rows: list[dict[str, str]]) -> None:
    packaged_results = PAPER / "evidence" / "results"
    packaged_results.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Performance by camera azimuth separation. Bins are mutually exclusive.}",
        r"\label{tab:view_angle}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Angle bin & Windows & Canon. avg. & Fused " + ROW_END,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            fr"{row['angle_bin']}$^\circ$ & {int(row['sample_count']):,} & "
            fr"{number(row, 'canonical_avg_mpjpe'):.4f} & "
            fr"{number(row, 'fused_mpjpe'):.4f} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_tex("view_angle.tex", lines)


def masking_table(rows: list[dict[str, str]]) -> None:
    selected = [row for row in rows if number(row, "ratio") in {0.5, 1.0}]
    view_order = {"left": 0, "right": 1, "both": 2}
    pattern_order = {"random": 0, "distal": 1, "temporal": 2}
    selected.sort(
        key=lambda row: (
            view_order[row["view_mode"]],
            pattern_order[row["pattern"]],
            number(row, "ratio"),
        )
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete held-out-split masking robustness at representative severity levels. Positive gain indicates lower MPJPE than canonical averaging under the same corruption.}",
        r"\label{tab:masking_extension}",
        r"\small",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Affected view(s) & Pattern & Ratio & Canon. avg. & Fused & Gain (\%) " + ROW_END,
        r"\midrule",
    ]
    previous_view = None
    for row in selected:
        if previous_view is not None and row["view_mode"] != previous_view:
            lines.append(r"\midrule")
        baseline = number(row, "canonical_avg_mpjpe")
        fused = number(row, "fused_mpjpe")
        gain = 100.0 * (baseline - fused) / baseline
        lines.append(
            fr"{row['view_mode'].title()} & {row['pattern'].title()} & "
            fr"{number(row, 'ratio'):.2f} & {baseline:.4f} & {fused:.4f} & {gain:+.1f} " + ROW_END
        )
        previous_view = row["view_mode"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    write_tex("masking_summary.tex", lines)


def figures(
    temporal: list[dict[str, str]],
    sampling: list[dict[str, str]],
    view: list[dict[str, str]],
    masking: list[dict[str, str]],
) -> None:
    out = PAPER / "figures" / "extension"
    out.mkdir(parents=True, exist_ok=True)
    apply_journal_style()

    try:
        from .generate_consolidated_artifacts import temporal_sampling_figure
    except ImportError:
        from generate_consolidated_artifacts import temporal_sampling_figure
    temporal_sampling_figure(temporal, sampling, out / "temporal_sampling.pdf")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = [number(row, "mean_separation_deg") for row in view]
    ax.plot(
        x, [number(row, "canonical_avg_mpjpe") for row in view],
        color=JOURNAL_COLORS["grey"], marker="s", markerfacecolor="white",
        linestyle="--", label="Canonical average",
    )
    ax.plot(
        x, [number(row, "fused_mpjpe") for row in view],
        color=JOURNAL_COLORS["blue"], marker="o", label="CanonFuse3D",
    )
    ax.set(xlabel="Mean azimuth separation (degrees)", ylabel="MPJPE")
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "view_angle.pdf")
    plt.close(fig)

    colors = {
        "random": JOURNAL_COLORS["blue"],
        "distal": JOURNAL_COLORS["orange"],
        "temporal": JOURNAL_COLORS["teal"],
    }
    markers = {"random": "o", "distal": "s", "temporal": "^"}
    panels = {"left": "(a)", "right": "(b)", "both": "(c)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True)
    for ax, view_mode in zip(axes, ("left", "right", "both")):
        for pattern in ("random", "distal", "temporal"):
            subset = [
                row for row in masking
                if row["view_mode"] == view_mode and row["pattern"] == pattern
            ]
            subset.sort(key=lambda row: number(row, "ratio"))
            ratios = [number(row, "ratio") for row in subset]
            ax.plot(
                ratios, [number(row, "fused_mpjpe") for row in subset],
                color=colors[pattern], marker=markers[pattern], label=pattern.title(),
            )
            ax.plot(
                ratios, [number(row, "canonical_avg_mpjpe") for row in subset],
                color=colors[pattern], marker=markers[pattern], markerfacecolor="white",
                linestyle="--", alpha=0.85,
            )
        ax.set_title(f"{panels[view_mode]} {view_mode.title()} view")
        ax.set_xlabel("Mask ratio")
        style_axes(ax)
    axes[0].set_ylabel("MPJPE")
    pattern_handles = [
        Line2D([0], [0], color=colors[name], marker=markers[name], label=name.title())
        for name in ("random", "distal", "temporal")
    ]
    method_handles = [
        Line2D([0], [0], color=JOURNAL_COLORS["black"], linestyle="-", label="CanonFuse3D"),
        Line2D([0], [0], color=JOURNAL_COLORS["black"], linestyle="--", label="Canonical average"),
    ]
    fig.legend(
        handles=pattern_handles + method_handles,
        loc="lower center", ncol=5, frameon=False, fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "masking_robustness.pdf")
    plt.close(fig)


def combine_masking(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    if len(rows) != 45:
        raise ValueError(f"Expected 45 masking rows, found {len(rows)}")
    output = LOGS / "masking" / "occlusion_summary_combined.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def evidence(
    paths: list[Path],
    masking_rows: list[dict[str, str]],
    frontend_rows: list[dict[str, str]],
) -> None:
    packaged_results = PAPER / "evidence" / "results"
    packaged_results.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IVC extension evidence manifest",
        "",
        fr"- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`",
        fr"- Fold: `0`",
        fr"- Seed: `42`",
        fr"- Complete test camera-pair sequences: `{EXPECTED_SAMPLE_COUNT}`",
        "- Perturbation metrics use all 15 joints in normalized dataset coordinates.",
        "- Front-end metrics use the shared common-13 subset in normalized dataset coordinates.",
        "",
        "## Source artifacts",
        "",
    ]
    for path in paths:
        relative = path.relative_to(LOGS)
        packaged = packaged_results / "__".join(relative.parts)
        shutil.copy2(path, packaged)
        lines.append(
            f"- `{path.relative_to(ROOT)}` — SHA-256 `{sha256(path)}`; "
            f"packaged copy `{packaged.relative_to(PAPER)}`."
        )
    lines.extend([
        "",
        "## Upstream estimator provenance",
        "",
    ])
    for row in frontend_rows:
        if row["frontend_name"] == "sam3d":
            continue
        lines.append(
            "- `{}`: repository commit `{}`, estimator checkpoint SHA-256 `{}`, "
            "prediction manifest SHA-256 `{}`.".format(
                row["frontend_name"],
                row["official_repo_commit"],
                row["estimator_checkpoint_sha256"],
                row["manifest_sha256"],
            )
        )
    lines.extend([
        "",
        "## Masking grid",
        "",
        fr"- Rows: `{len(masking_rows)}` (3 view modes x 3 patterns x 5 ratios).",
        "- Corruption: 35% zero replacement and 65% Gaussian perturbation with standard deviation 0.08.",
        "",
        "## Author-confirmation items",
        "",
        "- Confirm final MMSports publication metadata (venue, DOI, and pages).",
        "- Confirm CRediT roles, competing-interest declaration, code/data release URLs, ethics wording, and generative-AI disclosure.",
    ])
    (PAPER / "evidence" / "evidence_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frontend_path = LOGS / "frontend_generalization" / "frontend_generalization_comparison.csv"
    temporal_path = LOGS / "temporal_offset" / "temporal_offset_summary_last.csv"
    sampling_path = LOGS / "sampling_rate" / "sampling_rate_summary_last.csv"
    view_path = LOGS / "view_angle" / "view_angle_summary_last.csv"
    masking_paths = [
        LOGS / "masking" / view / "occlusion_summary_last.csv"
        for view in ("left", "right", "both")
    ]
    frontend = read_csv(frontend_path)
    temporal = read_csv(temporal_path)
    sampling = read_csv(sampling_path)
    view = read_csv(view_path)
    masking = combine_masking(masking_paths)
    validate(frontend)
    validate(temporal)
    validate(sampling)
    validate(view, per_bin=True)
    validate(masking)
    if sum(int(row["sample_count"]) for row in view) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("View bins are not mutually exclusive and exhaustive")
    native_reference_table(frontend)
    frontend_table(frontend)
    temporal_sampling_table(temporal, sampling)
    view_table(view)
    masking_table(masking)
    figures(temporal, sampling, view, masking)
    evidence([frontend_path, temporal_path, sampling_path, view_path, *masking_paths], masking, frontend)


if __name__ == "__main__":
    main()
