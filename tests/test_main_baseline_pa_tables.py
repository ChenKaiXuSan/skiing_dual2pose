import json
from pathlib import Path
import tempfile
import unittest
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
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
                    'metrics': metrics, 'provenance': {'canonical_mode': 'per_frame', 'evaluation_protocol': EVALUATION_PROTOCOL, 'ski_prediction_root_centered': dataset=='ski'}}))

    def tearDown(self): self.temp.cleanup()

    def test_approved_local_results_need_no_retired_mlp(self):
        from dual2pose.eval import evaluate_main_baseline_pa as pa
        from dual2pose.experiments.workflow_external import LOCAL
        for dataset in ('unity', 'ski'):
            (self.root / dataset / 'mlp.json').unlink(missing_ok=True)
        records = load_records(self.root)
        self.assertEqual(len(records), 22)
        self.assertEqual(set(DIRECT + GEOMETRY), set(LOCAL))
        self.assertEqual(set(pa.DIRECT + pa.GEOMETRY), set(LOCAL))
        self.assertNotIn('Cross-view MLP', render_table('unity', records))

    def test_cli_manifest_reports_current_method_counts(self):
        from unittest.mock import patch
        from dual2pose.eval.render_main_baseline_tables import main
        for dataset in ('unity', 'ski'):
            (self.root / dataset / 'mlp.json').unlink(missing_ok=True)
        paper = self.root / 'rendered'
        with patch('sys.argv', ['render', '--results', str(self.root), '--paper', str(paper)]):
            main()
        manifest = json.loads((paper / 'evidence/main_baselines_manifest.json').read_text())
        self.assertEqual(manifest['method_count_per_dataset'], 11)
        self.assertEqual(manifest['direct_group_rows'], 8)
        self.assertEqual(manifest['record_count'], 22)

    def mutate(self, key, value=None, remove=False):
        path = self.root/'unity/tcn.json'
        data = json.loads(path.read_text())
        if remove: del data['metrics']['common13'][key]
        else: data['metrics']['common13'][key] = value
        path.write_text(json.dumps(data))

    def test_historical_records_are_rejected(self):
        path = self.root / "unity/tcn.json"
        record = json.loads(path.read_text())
        record["provenance"].pop("evaluation_protocol")
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "protocol"):
            load_records(self.root)

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
        self.assertEqual(unity.count('PA-MPJPE'), 1)
        self.assertEqual(ski.count('PA-MPJPE'), 1)
        self.assertNotIn('All 15 joints', unity)
        self.assertIn('common 13 joints', unity)
        for text, count in [(unity, 3), (ski, 3)]:
            row = next(x for x in text.splitlines() if x.startswith('Temporal fusion (local TCN)'))
            self.assertEqual(row.count(' & '), count)

    def test_method_citations_preserve_attribution_boundaries(self):
        text = render_table('unity', load_records(self.root))
        expected = {
            'Aligned average': 'umeyama1991least',
            'Temporal fusion (local TCN)': 'bai2018empirical',
            'Avg. + SmoothNet': 'zeng2022smoothnet',
            'Calibrated DLT': 'hartley2004multiple',
            'Reprojection-gated DLT': 'hartley2004multiple',
            'DLT-residual MLP': 'hartley2004multiple',
        }
        for label, key in expected.items():
            self.assertIn(label + r'~\cite{' + key + '}', text)
        self.assertIn('technical components or background', text)
        self.assertIn('not reproductions', text)


if __name__ == '__main__': unittest.main()
