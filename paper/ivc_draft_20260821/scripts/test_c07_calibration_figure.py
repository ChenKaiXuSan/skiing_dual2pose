"""Source-to-plot checks: retain every condition and the exact measured angles."""
import unittest
import numpy as np
from generate_c07_calibration_figure import load_data, make_figure


class CalibrationFigureTest(unittest.TestCase):
    def test_archived_values_and_all_conditions(self):
        data = load_data()
        self.assertEqual(len(data), 7)
        np.testing.assert_array_equal(data[0]["angles"], [0, 1, 3])
        self.assertEqual([round(x, 4) for x in data[0]["dlt_mlp"]], [.0988, .1548, .3747])
        self.assertEqual([round(x, 4) for x in data[-1]["dlt_mlp"]], [.4155, .4257, .5007])
        self.assertAlmostEqual(data[-1]["canonical_avg"], .4654, places=4)
        self.assertAlmostEqual(data[-1]["canonfuse3d"], .5986, places=4)

    def test_every_panel_plots_real_angles_and_unchanged_references(self):
        data = load_data()
        fig = make_figure(data)
        for ax, row in zip(fig.axes[:7], data):
            lines = ax.get_lines()
            self.assertEqual(len(lines), 3)
            np.testing.assert_array_equal(lines[0].get_xdata(), [0, 1, 3])
            np.testing.assert_array_equal(lines[0].get_ydata(), row["dlt_mlp"])
            np.testing.assert_array_equal(lines[1].get_ydata(), [row["canonfuse3d"]] * 3)
            np.testing.assert_array_equal(lines[2].get_ydata(), [row["canonical_avg"]] * 3)
            self.assertEqual(ax.get_ylim(), (0., .65))


if __name__ == "__main__":
    unittest.main()
