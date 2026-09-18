"""Unity-only presentation must retain values and omit empty dataset columns."""
import unittest
from dual2pose.eval.render_native_output_table import render_native_output_table


class UnityOnlyTableTest(unittest.TestCase):
    def test_unity_only_preserves_values_without_ski_columns(self):
        report = {'unity': {'methods': {'sam3d_native_view_pool': {
            'status': 'measured', 'mpjpe': .1234, 'pa_mpjpe': .0678,
            'acceleration_error': .0091}}}}
        tex = render_native_output_table(report, unity_only=True)
        self.assertIn('Unity', tex)
        self.assertNotIn('Ski-PTZ-Pose', tex)
        row = next(x for x in tex.splitlines() if x.startswith('SAM 3D Body'))
        self.assertEqual(row.count('&'), 3)
        self.assertIn('0.1234 & 0.0678 & 0.0091', row)
        missing = next(x for x in tex.splitlines() if x.startswith('MetaPose'))
        self.assertIn('& -- & -- & --', missing)
        self.assertNotIn('& No &', tex)
        self.assertNotIn('Method & Canon', tex)
        self.assertNotIn(r'\multicolumn{3}{c}{Unity}', tex)

    def test_legacy_two_dataset_view_is_still_available(self):
        tex = render_native_output_table({})
        self.assertIn('Ski-PTZ-Pose', tex)


if __name__ == '__main__':
    unittest.main()
