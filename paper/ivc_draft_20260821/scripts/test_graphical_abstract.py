"""Checks that the graphical abstract meets the Elsevier size requirement."""
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from paper.ivc_draft_20260821.scripts.generate_graphical_abstract import CANVAS_CM, build

ELSEVIER_MIN_PX = (1328, 531)  # width x height


@unittest.skipUnless(shutil.which("pdftoppm"), "pdftoppm (poppler) is required to crop the Fig. 1 panels")
class GraphicalAbstractTest(unittest.TestCase):
    def test_outputs_are_at_least_elsevier_minimum_and_keep_13_by_5_cm(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf, png = build(Path(tmp) / "ga")
            self.assertTrue(pdf.exists() and pdf.stat().st_size > 0)
            with Image.open(png) as image:
                width, height = image.size
            self.assertGreaterEqual(width, ELSEVIER_MIN_PX[0])
            self.assertGreaterEqual(height, ELSEVIER_MIN_PX[1])
            self.assertAlmostEqual(width / height, CANVAS_CM[0] / CANVAS_CM[1], places=2)


if __name__ == "__main__":
    unittest.main()
