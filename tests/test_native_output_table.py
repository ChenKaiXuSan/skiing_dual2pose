"""Regression tests for the independent pre-Canon manuscript table."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from dual2pose.eval.evaluate_native_output_comparison import METHOD_IDS
from dual2pose.eval.render_native_output_table import EXPECTED_ROW_LABELS, render_native_output_table


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NATIVE_REFERENCE_SHA256 = "af44bb8ac6adc0b5e5b2ab2e74cf7a65bbf1442b4a16f0082c8643195a5eaa44"


def fixture_report(unavailable: str | None = None):
    methods = {}
    for index, method in enumerate(METHOD_IDS):
        value = 0.1 + index * 0.001
        methods[method] = {
            "status": "measured",
            "mpjpe": value,
            "pa_mpjpe": value / 2,
            "acceleration_error": value / 4,
        }
    if unavailable:
        methods[unavailable] = {
            "status": "unavailable",
            "reason": "no supplied frame mapping",
            "mpjpe": None,
            "pa_mpjpe": None,
            "acceleration_error": None,
        }
    return {"unity": {"methods": methods}, "ski": {"methods": methods}}


class NativeOutputTableTest(unittest.TestCase):
    def test_contains_all_native_rows_in_declared_order(self) -> None:
        tex = render_native_output_table(fixture_report())
        offsets = [tex.index(label) for label in EXPECTED_ROW_LABELS]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(len(offsets), 16)

    def test_has_all_three_metrics_for_both_datasets(self) -> None:
        tex = render_native_output_table(fixture_report())
        self.assertEqual(tex.count("MPJPE"), 4)
        self.assertGreaterEqual(tex.count("Accel."), 2)
        self.assertIn("Unity", tex)
        self.assertIn("Ski-PTZ-Pose", tex)

    def test_unavailable_cells_render_as_double_dash(self) -> None:
        tex = render_native_output_table(fixture_report("metapose_native"))
        self.assertIn("--", tex)
        self.assertNotIn("nan", tex.lower())

    def test_diagnostic_unaligned_average_is_never_bold(self) -> None:
        row = next(
            line
            for line in render_native_output_table(fixture_report()).splitlines()
            if "Unaligned native average" in line
        )
        self.assertNotIn("\\textbf", row)

@unittest.skipUnless(
    (ROOT / "paper/ivc_draft_20260821").is_dir(),
    "Optional manuscript integration test: paper/ is not distributed with this repository",
)
class NativeOutputManuscriptTest(unittest.TestCase):
    def test_table_is_input_immediately_after_native_reference(self) -> None:
        main = (ROOT / "paper/ivc_draft_20260821/main.tex").read_text(encoding="utf-8")
        expected = (
            "\\input{tables/native_reference.tex}\n"
            "\\input{tables/native_output_comparison.tex}"
        )
        self.assertIn(expected, main)

    def test_integrated_canonical_reference_hash_is_stable(self) -> None:
        path = ROOT / "paper/ivc_draft_20260821/tables/native_reference.tex"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), EXPECTED_NATIVE_REFERENCE_SHA256)


    def test_paper_separates_canonical_and_native_protocols(self) -> None:
        canonical = (
            ROOT / "paper/ivc_draft_20260821/tables/native_reference.tex"
        ).read_text(encoding="utf-8")
        native = (
            ROOT / "paper/ivc_draft_20260821/tables/native_output_comparison.tex"
        ).read_text(encoding="utf-8")
        self.assertIn("after the shared body-coordinate normalization", canonical)
        self.assertIn("before the shared body-coordinate normalization", native)
        self.assertEqual(canonical.count(r"\begin{table*}"), 1)
        self.assertEqual(native.count(r"\begin{table*}"), 1)
        self.assertIn("CanonFuse3D (ours)", canonical)
        self.assertNotIn("CanonFuse3D (ours)", native)

    def test_canonical_table_contains_cited_external_results(self) -> None:
        canonical = (
            ROOT / "paper/ivc_draft_20260821/tables/native_reference.tex"
        ).read_text(encoding="utf-8")
        expected_rows = (
            r"Avg. + STRIDE (adapted)~\cite{lal2025stride} & 0.2799 & 0.1174 & 0.0316 & 0.3483 & 0.1861 & 0.0387",
            r"DeciWatch (official-protocol adaptation)~\cite{zeng2022deciwatch} & 0.1134 & 0.0695 & 0.0132 & 0.2239 & 0.1549 & 0.0168",
            r"MetaPose (adapted)~\cite{usman2022metapose} & 0.2655 & 0.1089 & 0.0400 & 0.4253 & 0.1425 & 0.0634",
        )
        for row in expected_rows:
            self.assertIn(row, canonical)

    def test_native_table_contains_unity_scores_and_withholds_ski(self) -> None:
        native = (
            ROOT / "paper/ivc_draft_20260821/tables/native_output_comparison.tex"
        ).read_text(encoding="utf-8")
        expected_rows = (
            r"SAM 3D Body~\cite{yang2026sam} & No & 0.1019 & 0.0732 & 0.0369 & -- & -- & --",
            r"STRIDE (per-view adaptation)~\cite{lal2025stride} & No & 0.1072 & 0.0776 & 0.0357 & -- & -- & --",
            r"DeciWatch (per-view adaptation)~\cite{zeng2022deciwatch} & No & 0.6315 & 0.3033 & 0.0119 & -- & -- & --",
            r"MetaPose (adapted)~\cite{usman2022metapose} & No & 0.2171 & 0.1088 & 0.0382 & -- & -- & --",
        )
        for row in expected_rows:
            self.assertIn(row.removesuffix(" & -- & -- & --").replace(" & No & ", " & "), native)
        self.assertNotIn("Ski-PTZ-Pose", native)

    def test_results_text_cites_and_distinguishes_external_adaptations(self) -> None:
        main = (ROOT / "paper/ivc_draft_20260821/main.tex").read_text(
            encoding="utf-8"
        )
        for citation in (
            r"\cite{lal2025stride}",
            r"\cite{zeng2022deciwatch}",
            r"\cite{usman2022metapose}",
        ):
            self.assertIn(citation, main)
        self.assertIn("must not be ranked across the two tables", main)


if __name__ == "__main__":
    unittest.main()
