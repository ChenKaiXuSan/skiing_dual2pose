"""Plot the seven archived calibration conditions without smoothing or selection."""
from pathlib import Path
import hashlib
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from journal_figure_style import apply_journal_style, JOURNAL_COLORS

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "evidence/c07_matched_occlusion_20260917/results"
CONDITIONS = (
    ("clean", "Clean"), ("random_0p5_left", "Left 0.5"),
    ("random_0p5_right", "Right 0.5"), ("random_0p5_both", "Both 0.5"),
    ("random_1_left", "Left 1.0"), ("random_1_right", "Right 1.0"),
    ("random_1_both", "Both 1.0"),
)


def load_data():
    """Validate archived full-precision scores against all 35 old table cells."""
    table = (PAPER / "tables/c07_calibration_mpjpe.tex").read_text()
    rows = []
    for stem, label in CONDITIONS:
        raw = json.loads((SOURCE / f"{stem}.json").read_text())
        assert raw["status"] == "complete"
        scores = {(r["method"], r["angle_degrees"]): r["metrics"]["mpjpe"]
                  for r in raw["results"]}
        assert all(r["metrics"]["sample_count"] == 64440 for r in raw["results"])
        row = {"condition": stem, "label": label, "angles": [0, 1, 3],
               "dlt_mlp": [scores["dlt_mlp", a] for a in (0, 1, 3)],
               "canonfuse3d": scores["canonfuse3d", 0],
               "canonical_avg": scores["canonical_avg", 0]}
        for method in ("canonfuse3d", "canonical_avg"):
            assert all(scores[method, a] == row[method] for a in (0, 1, 3))
        cells = [row["canonfuse3d"], row["canonical_avg"], *row["dlt_mlp"]]
        old = next(line for line in table.splitlines() if line.startswith(label + " &"))
        assert re.findall(r"\d+\.\d+", old.split(" & ", 1)[1]) == [f"{v:.4f}" for v in cells]
        rows.append(row)
    return rows


def make_figure(rows):
    apply_journal_style()
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.0), sharex=True, sharey=True)
    for index, (ax, row) in enumerate(zip(axes.flat, rows)):
        ax.plot(row["angles"], row["dlt_mlp"], "o-", color=JOURNAL_COLORS["orange"],
                markersize=4, label="DLT-residual")
        ax.plot(row["angles"], [row["canonfuse3d"]] * 3, "--",
                color=JOURNAL_COLORS["blue"], label="CanonFuse3D")
        ax.plot(row["angles"], [row["canonical_avg"]] * 3, ":",
                color=JOURNAL_COLORS["black"], label="Canonical average")
        ax.set_title(f"({chr(97 + index)}) {row['label']}", fontsize=10)
        ax.set_ylim(0, .65)
        ax.set_xlim(-.12, 3.12)
        ax.set_xticks([0, 1, 3])
        ax.set_yticks([0, .2, .4, .6])
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes.flat[-1].set_axis_off()
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),
                         loc="center", fontsize=9.5, handlelength=2.0)
    fig.supxlabel("Supplied right-camera rotation error (degrees)", fontsize=10)
    fig.supylabel("Common-13 MPJPE (dataset coordinate units)", fontsize=10)
    fig.subplots_adjust(left=.105, right=.99, bottom=.19, top=.89, hspace=.62, wspace=.17)
    return fig


def main():
    rows = load_data()
    fig = make_figure(rows)
    path = PAPER / "figures/extension/c07_calibration_sensitivity.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    evidence = PAPER / "evidence/compact_layout_20260917"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"verified_original_table_cells": 35, "conditions": rows,
                "source_hashes": {str(SOURCE / f"{s}.json"): sha(SOURCE / f"{s}.json")
                                  for s, _ in CONDITIONS},
                "script_sha256": sha(Path(__file__)), "figure_sha256": sha(path),
                "figure_table_trace": {
                    "artifact_id": "fig:c07_calibration",
                    "source_data": "Seven archived C07 JSON results; all 35 former Table 8 cells checked",
                    "transformation": "Full-precision MPJPE at 0/1/3 degrees; unchanged pose-only scores repeated as references",
                    "caption_claim": "Calibration sensitivity under all seven matched input conditions",
                    "supported_manuscript_claims": [
                        "DLT-residual retains a lower MPJPE on clean and ratio-0.5 inputs at 1 degree",
                        "The three displayed methods change ranking conditionally; averaging remains best under severe masks at 3 degrees"],
                    "limitations": ["Fixed checkpoint and partitions, not independent training repeats",
                                    "No measured intermediate angles; line segments only guide the eye",
                                    "Additional 2D and calibration inputs for DLT-residual",
                                    "Artificial calibration errors, not physical camera movement"]}}
    (evidence / "calibration_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified 35 original table cells; saved {path}")


if __name__ == "__main__":
    main()
