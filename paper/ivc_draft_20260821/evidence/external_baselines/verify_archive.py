#!/usr/bin/env python3
"""Verify the archived IVC external-baseline evidence package."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ARCHIVE = SCRIPT.parent
PAPER = SCRIPT.parents[2]
ROOT = SCRIPT.parents[4]

SOURCE_MAP = {
    "results/stride_unity_metrics.json": "logs/ivc_mmsports_extension/external_baselines/20260910/stride_unity_test_full_seed42/metrics.json",
    "results/stride_ski_metrics.json": "logs/ivc_mmsports_extension/external_baselines/20260910/stride_ski_test_full_seed42/metrics.json",
    "results/metapose_unity_metrics.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_unity_full_seed42/metrics.json",
    "results/metapose_ski_metrics.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_ski_full_seed42/metrics.json",
    "results/deciwatch_unity_metrics.json": "logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_unity_official_protocol_full_seed4321/metrics.json",
    "results/deciwatch_ski_metrics.json": "logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_ski_official_protocol_full_seed4321/metrics.json",
    "results/native_unity_scores.json": "logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/scores.json",
    "results/native_ski_scores.json": "logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/scores.json",
    "provenance/stride_unity_report.json": "logs/ivc_mmsports_extension/external_baselines/20260910/stride_unity_test_full_seed42/report.json",
    "provenance/stride_ski_report.json": "logs/ivc_mmsports_extension/external_baselines/20260910/stride_ski_test_full_seed42/report.json",
    "provenance/metapose_unity_report.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_unity_full_seed42/report.json",
    "provenance/metapose_ski_report.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_ski_full_seed42/report.json",
    "provenance/metapose_unity_config.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_unity_full_seed42/config.json",
    "provenance/metapose_ski_config.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_ski_full_seed42/config.json",
    "provenance/metapose_unity_progress.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_unity_full_seed42/progress.json",
    "provenance/metapose_ski_progress.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_ski_full_seed42/progress.json",
    "provenance/metapose_unity_selected_stages.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_unity_full_seed42/selected_stages.json",
    "provenance/metapose_ski_selected_stages.json": "logs/ivc_mmsports_extension/external_baselines/20260911/metapose_ski_full_seed42/selected_stages.json",
    "provenance/deciwatch_unity_report.json": "logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_unity_official_protocol_full_seed4321/report.json",
    "provenance/deciwatch_ski_report.json": "logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_ski_official_protocol_full_seed4321/report.json",
    "provenance/deciwatch_unity_config.json": "logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_unity_official_protocol_full_seed4321/config.json",
    "provenance/deciwatch_ski_config.json": "logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_ski_official_protocol_full_seed4321/config.json",
}

EXPECTED = {
    ("stride", "unity"): (0.2799222415524485, 0.11736991393327023, 0.03157572966423695),
    ("stride", "ski"): (0.34827979103566553, 0.1860614763567546, 0.03867885560383655),
    ("metapose", "unity"): (0.26545565695547213, 0.1089323299670668, 0.03997513398136868),
    ("metapose", "ski"): (0.42525706215340714, 0.14252038652083346, 0.06344784429882552),
    ("deciwatch", "unity"): (0.11336497942908902, 0.06946216342782602, 0.01320843282677357),
    ("deciwatch", "ski"): (0.2239018529925384, 0.15489959672654635, 0.016788053691389228),
}

COVERAGE = {
    "unity": {"samples": 64440, "points": 25131600, "frames": 1933200},
    "ski": {"samples": 30, "points": 11700, "frames": 900},
}

CANONICAL_ROWS = (
    r"Avg. + STRIDE (adapted)~\cite{lal2025stride} & 0.2799 & 0.1174 & 0.0316 & 0.3483 & 0.1861 & 0.0387",
    r"DeciWatch (official-protocol adaptation)~\cite{zeng2022deciwatch} & 0.1134 & 0.0695 & 0.0132 & 0.2239 & 0.1549 & 0.0168",
    r"MetaPose (adapted)~\cite{usman2022metapose} & 0.2655 & 0.1089 & 0.0400 & 0.4253 & 0.1425 & 0.0634",
)

NATIVE_ROWS = (
    r"STRIDE (per-view adaptation)~\cite{lal2025stride} & No & 0.1072 & 0.0776 & 0.0357 & -- & -- & --",
    r"DeciWatch (per-view adaptation)~\cite{zeng2022deciwatch} & No & 0.6315 & 0.3033 & 0.0119 & -- & -- & --",
    r"MetaPose (adapted)~\cite{usman2022metapose} & No & 0.2171 & 0.1088 & 0.0382 & -- & -- & --",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(relative: str):
    with (ARCHIVE / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_metric_record(record: dict, dataset: str, expected: tuple[float, float, float]) -> None:
    actual = (record["mpjpe"], record["pa_mpjpe"], record["acceleration_error"])
    assert all(math.isfinite(value) for value in actual)
    assert actual == expected, (dataset, actual, expected)
    expected_coverage = COVERAGE[dataset]
    assert record["sample_count"] == expected_coverage["samples"]
    assert record["point_count"] == expected_coverage["points"]
    assert record["pa_point_count"] == expected_coverage["points"]
    assert record["pa_frame_count"] == expected_coverage["frames"]
    assert record["pa_degenerate_prediction_frames"] == 0


def main() -> None:
    manifest_files = []
    for archived_name, source_name in sorted(SOURCE_MAP.items()):
        archived_path = ARCHIVE / archived_name
        source_path = ROOT / source_name
        assert archived_path.is_file(), archived_path
        assert source_path.is_file(), source_path
        archived_hash = sha256(archived_path)
        source_hash = sha256(source_path)
        assert archived_hash == source_hash, archived_name
        manifest_files.append(
            {
                "archive_path": archived_name,
                "source_path": source_name,
                "sha256": archived_hash,
                "bytes": archived_path.stat().st_size,
            }
        )

    for (method, dataset), expected in EXPECTED.items():
        key = "stride_adapted" if method == "stride" else method
        record = load(f"results/{method}_{dataset}_metrics.json")
        assert_metric_record(record[key], dataset, expected)
        delta = record.get("control_delta", record.get("archived_average_control_delta"))
        if delta is not None:
            assert delta == {"mpjpe": 0.0, "pa_mpjpe": 0.0, "acceleration_error": 0.0}

    for dataset in ("unity", "ski"):
        progress = load(f"provenance/metapose_{dataset}_progress.json")
        assert progress["status"] == "complete_evaluated"
        assert progress["windows"] == COVERAGE[dataset]["samples"]
        assert progress["frames"] == COVERAGE[dataset]["frames"]

    native_unity = load("results/native_unity_scores.json")
    native_ski = load("results/native_ski_scores.json")
    unity_statuses = [value["status"] for value in native_unity["methods"].values()]
    ski_statuses = [value["status"] for value in native_ski["methods"].values()]
    assert unity_statuses.count("measured") == 13
    assert unity_statuses.count("unavailable") == 4
    assert ski_statuses.count("measured") == 0
    assert ski_statuses.count("unavailable") == 17

    canonical_tex = (PAPER / "tables/native_reference.tex").read_text(encoding="utf-8")
    native_tex = (PAPER / "tables/native_output_comparison.tex").read_text(encoding="utf-8")
    for row in CANONICAL_ROWS:
        assert row in canonical_tex, row
    for row in NATIVE_ROWS:
        assert row.removesuffix(" & -- & -- & --").replace(" & No & ", " & ") in native_tex, row

    references = (PAPER / "references.bib").read_text(encoding="utf-8")
    for key in ("lal2025stride", "zeng2022deciwatch", "usman2022metapose"):
        assert re.search(r"@\w+\{" + re.escape(key) + r",", references), key

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "methods must not be ranked across the two tables" in main_tex

    multiseed = (ROOT / "docs/superpowers/plans/2026-08-21-ivc-p1-multiseed-crossfold.md").read_text(encoding="utf-8")
    assert "Retired from the current IVC submission scope" in multiseed
    assert "six full training cells were not launched" in multiseed

    smoke_logs = (
        ROOT / "logs/ivc_p1/multiseed_smoke/fold_1/seed_13/train_unity.log",
        ROOT / "logs/ivc_p1/multiseed_smoke_retry/fold_1/seed_13/train_unity.log",
    )
    for smoke_log in smoke_logs:
        assert smoke_log.is_file(), smoke_log
        smoke_text = smoke_log.read_text(encoding="utf-8", errors="replace")
        assert "No valid frames after filtering" in smoke_text, smoke_log
        assert not list(smoke_log.parent.rglob("*.ckpt")), smoke_log.parent
    assert not (ROOT / "logs/ivc_p1/multiseed").is_dir()

    log_path = PAPER / "main.log"
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        for forbidden in ("LaTeX Warning: There were undefined references.", "Overfull \\hbox", "! LaTeX Error:"):
            assert forbidden not in log, forbidden

    pages = {}
    aux_path = PAPER / "main.aux"
    if aux_path.is_file():
        aux = aux_path.read_text(encoding="utf-8", errors="replace")
        for label in ("tab:native_reference", "tab:native_output_comparison"):
            match = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{[^}]+\}\{([^}]+)\}", aux)
            assert match, label
            pages[label] = match.group(1)

    manifest = {
        "archive_date": "2026-09-13",
        "status": "complete",
        "file_count": len(manifest_files),
        "files": manifest_files,
    }
    verification = {
        "archive_date": "2026-09-13",
        "status": "passed",
        "external_matrix": "3 methods x 2 datasets; complete",
        "external_result_cells": 6,
        "paper_metric_cells_verified": 36,
        "native_unity": {"measured": 13, "unavailable": 4},
        "native_ski": {"measured": 0, "unavailable": 17},
        "table_pages": pages,
        "multiseed_crossfold": {
            "status": "retired_not_run",
            "planned_full_training_cells": 6,
            "admitted_result_cells": 0,
            "launch_smoke_logs": 2,
            "training_epochs_completed": 0,
            "formal_checkpoints": 0,
        },
    }
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARCHIVE / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
