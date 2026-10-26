#!/usr/bin/env python3
"""Generate the P1 repeated-training table from the verified six-run summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
INPUT = ROOT / "logs" / "ivc_p1" / "multiseed" / "multiseed_summary.json"
EXPECTED_CELLS = {(fold, seed) for fold in (0, 1) for seed in (13, 42, 73)}
ROW_END = r"\\"


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    runs = [dict(row) for row in output.get("runs", [])]
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in runs:
        key = (int(row["fold"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate repeated-training cell: {key}")
        if int(row["sample_count"]) != 64_440:
            raise ValueError(f"Every repeated-training run must contain all 64,440 test camera-pair sequences: {key}")
        row["fold"], row["seed"] = key
        row["mpjpe"] = float(row["mpjpe"])
        indexed[key] = row
    if set(indexed) != EXPECTED_CELLS:
        raise ValueError(
            f"Repeated-training matrix mismatch; missing={sorted(EXPECTED_CELLS - set(indexed))}, "
            f"extra={sorted(set(indexed) - EXPECTED_CELLS)}"
        )
    summary = dict(output.get("summary", {}))
    aggregate = dict(summary.get("aggregate", {}))
    if int(aggregate.get("run_count", -1)) != 6 or "df=5" not in str(aggregate.get("interval", "")):
        raise ValueError("Repeated-training aggregate must be a six-run df=5 t interval")
    per_fold = dict(summary.get("per_fold", {}))
    if set(per_fold) != {"0", "1"} or any(int(per_fold[key].get("run_count", -1)) != 3 for key in per_fold):
        raise ValueError("Repeated-training per-fold summary must contain three runs per fold")
    output["runs"] = [indexed[key] for key in sorted(indexed)]
    output["summary"] = summary
    return output


def render_table(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    aggregate = summary["aggregate"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training variability over six fresh training runs. The aggregate interval is a two-sided Student-\(t\) interval over runs (df=5), not over frames.}",
        r"\label{tab:multiseed_summary}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scope & Runs & MPJPE & SD & 95\% CI " + ROW_END,
        r"\midrule",
    ]
    for fold in (0, 1):
        row = summary["per_fold"][str(fold)]
        lines.append(
            f"Fold {fold} & {int(row['run_count'])} & {float(row['mpjpe_mean']):.4f} & "
            f"{float(row['mpjpe_std']):.4f} & -- " + ROW_END
        )
    lines.append(r"\midrule")
    lines.append(
        f"Both folds & {int(aggregate['run_count'])} & {float(aggregate['mpjpe_mean']):.4f} & "
        f"{float(aggregate['mpjpe_std']):.4f} & "
        f"[{float(aggregate['mpjpe_ci95_low']):.4f}, {float(aggregate['mpjpe_ci95_high']):.4f}] " + ROW_END
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.3em}",
            r"\begin{minipage}{0.94\linewidth}\footnotesize Each run is selected by validation MPJPE and evaluated once on all 64,440 test camera-pair sequences. Interval: df=5.\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = validate_payload(json.loads(INPUT.read_text(encoding="utf-8")))
    output = PAPER / "tables" / "multiseed_summary.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
