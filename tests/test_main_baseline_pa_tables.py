import json
from pathlib import Path
import tempfile
import unittest
from dual2pose.eval.render_main_baseline_tables import DIRECT, GEOMETRY, load_records, render_table


class PATableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for dataset, count in [('unity', 64440), ('ski', 30)]:
            (self.root/dataset).mkdir()
            for method in DIRECT+GEOMETRY:
                metrics = {}
                for subset in (['all15', 'common13'] if dataset == 'unity' else ['common13']):
                    joints = 15 if subset == 'all15' else 13
                    metrics[subset] = {'mpjpe': .2, 'pa_mpjpe': .1, 'acceleration_error': .03,
                        'sample_count': count, 'point_count': count*30*joints,
                        'acceleration_point_count': count*28*joints,
                        'pa_point_count': count*30*joints, 'pa_frame_count': count*30,
                        'pa_degenerate_prediction_frames': 0}
                (self.root/dataset/f'{method}.json').write_text(json.dumps({'method': method,
                    'metrics': metrics, 'provenance': {'ski_prediction_root_centered': dataset=='ski'}}))

    def tearDown(self): self.temp.cleanup()

    def mutate(self, key, value=None, remove=False):
        path = self.root/'unity/mlp.json'
        data = json.loads(path.read_text())
        if remove: del data['metrics']['common13'][key]
        else: data['metrics']['common13'][key] = value
        path.write_text(json.dumps(data))

    def test_missing_pa_value_is_rejected(self):
        self.mutate('pa_mpjpe', remove=True)
        with self.assertRaisesRegex(ValueError, 'PA|metric'):
            load_records(self.root)

    def test_pa_cannot_omit_frames(self):
        self.mutate('pa_frame_count', 1)
        with self.assertRaisesRegex(ValueError, 'PA|denominator'):
            load_records(self.root)

    def test_nonfinite_pa_cannot_enter_paper(self):
        self.mutate('pa_mpjpe', float('nan'))
        with self.assertRaises(ValueError): load_records(self.root)

    def test_each_joint_group_has_three_metric_columns(self):
        records = load_records(self.root)
        unity, ski = render_table('unity', records), render_table('ski', records)
        self.assertEqual(unity.count('PA-MPJPE'), 2)
        self.assertEqual(ski.count('PA-MPJPE'), 1)
        self.assertIn(r'\multicolumn{3}{c}{All 15 joints}', unity)
        for text, count in [(unity, 6), (ski, 3)]:
            row = next(x for x in text.splitlines() if x.startswith('Cross-view MLP &'))
            self.assertEqual(row.count(' & '), count)


if __name__ == '__main__': unittest.main()
