"""Render the independent native/pre-Canon result table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROWS = (
    ("Native monocular front ends", "sam3d_native_view_pool", r"SAM 3D Body~\cite{yang2026sam}", "No"),
    ("Native monocular front ends", "motionbert_native_view_pool", r"MotionBERT~\cite{zhu2023motionbert}", "No"),
    ("Native monocular front ends", "poseformer_native_view_pool", r"PoseFormer~\cite{zheng2021poseformer}", "No"),
    ("Native monocular front ends", "videopose3d_native_view_pool", r"VideoPose3D~\cite{pavllo2019videopose3d}", "No"),
    ("Pre-Canon controls", "left_native", "Left native", "No"),
    ("Pre-Canon controls", "right_native", "Right native", "No"),
    ("Pre-Canon controls", "unaligned_native_average", "Unaligned native average", "No"),
    ("Pre-Canon controls", "native_sequence_aligned_average", r"Sequence-aligned average~\cite{umeyama1991least}", "No"),
    ("Pre-Canon controls", "native_aligned_quality_weighted", "Aligned quality-weighted fusion", "No"),
    ("Pre-Canon controls", "native_aligned_average_smoothnet", r"Aligned average + SmoothNet~\cite{zeng2022smoothnet}", "No"),
    ("Published external methods", "stride_native_view_pool", r"STRIDE (per-view adaptation)~\cite{lal2025stride}", "No"),
    ("Published external methods", "deciwatch_native_view_pool", r"DeciWatch (per-view adaptation)~\cite{zeng2022deciwatch}", "No"),
    ("Published external methods", "metapose_native", r"MetaPose (adapted)~\cite{usman2022metapose}", "No"),
    ("Calibrated geometry", "calibrated_dlt_native", r"Calibrated DLT~\cite{hartley2004multiple}", "No"),
    ("Calibrated geometry", "reprojection_gated_dlt_native", r"Reprojection-gated DLT~\cite{hartley2004multiple}", "No"),
    ("Calibrated geometry", "dlt_residual_mlp_native", r"DLT-residual MLP~\cite{hartley2004multiple}", "No"),
)
EXPECTED_ROW_LABELS = tuple(row[2] for row in ROWS)


def _cell(row: dict[str, Any] | None, metric: str) -> str:
    if not row or row.get("status") != "measured" or row.get(metric) is None:
        return "--"
    return f"{float(row[metric]):.4f}"


def render_native_output_table(report: dict[str, Any]) -> str:
    unity = report.get("unity", {}).get("methods", {})
    ski = report.get("ski", {}).get("methods", {})
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Independent comparison before the shared body-coordinate normalization. All listed methods operate on their native outputs and do not use CanonFuse3D's canonical transform. Position error removes only per-frame pelvis translation; Procrustes alignment is a separate reflection-free diagnostic. A double dash denotes that no target-independent common coordinate frame or admissible output gauge was available.}",
        r"\label{tab:native_output_comparison}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Unity} & \multicolumn{3}{c}{Ski-PTZ-Pose} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Method & Canon & MPJPE $\downarrow$ & PA-MPJPE $\downarrow$ & Accel. $\downarrow$ & MPJPE $\downarrow$ & PA-MPJPE $\downarrow$ & Accel. $\downarrow$ \\",
        r"\midrule",
    ]
    previous_group = None
    for group, method, label, canon in ROWS:
        if group != previous_group:
            if previous_group is not None:
                lines.append(r"\addlinespace[2pt]")
            lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
            previous_group = group
        urow, srow = unity.get(method), ski.get(method)
        values = [_cell(urow, name) for name in ("mpjpe", "pa_mpjpe", "acceleration_error")]
        values += [_cell(srow, name) for name in ("mpjpe", "pa_mpjpe", "acceleration_error")]
        lines.append(f"{label} & {canon} & " + " & ".join(values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--ski", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "unity": json.loads(args.unity.read_text(encoding="utf-8")),
        "ski": json.loads(args.ski.read_text(encoding="utf-8")),
    }
    text = render_native_output_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()


__all__ = ["EXPECTED_ROW_LABELS", "ROWS", "render_native_output_table"]
