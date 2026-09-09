import importlib
import importlib.util
import unittest

from tests.paper_support import require_local_paper

require_local_paper()


class ConsolidatedArtifactTests(unittest.TestCase):
    def api(self):
        name='paper.ivc_draft_20260821.scripts.generate_consolidated_artifacts'
        self.assertIsNotNone(importlib.util.find_spec(name), 'Consolidated artifact generator is required')
        return importlib.import_module(name)

    def test_relative_change_uses_each_methods_own_zero(self):
        api=self.api()
        rows=[{'offset':1,'fused':2.02,'avg':3.9}, {'offset':0,'fused':2.,'avg':4.}]
        a=api.relative_series(rows,'offset','fused')
        b=api.relative_series(rows,'offset','avg')
        self.assertEqual(a['x'],[0.,1.])
        self.assertAlmostEqual(a['relative_percent'][1],1.)
        self.assertAlmostEqual(b['relative_percent'][1],-2.5)

    def test_relative_change_rejects_missing_or_duplicate_zero(self):
        api=self.api()
        for rows in [[{'x':1,'m':1}], [{'x':0,'m':1},{'x':0,'m':1}], [{'x':0,'m':0}]]:
            with self.assertRaises(ValueError): api.relative_series(rows,'x','m')

    def test_angle_merge_requires_matching_bins(self):
        api=self.api()
        with self.assertRaises(ValueError): api.merged_angle_rows([{'angle_bin':'0-30'}], [])

    def test_angle_merge_retains_measured_interval_and_rejects_inconsistent_means(self):
        api=self.api()
        view=[{'angle_bin':'0-30','sample_count':8,'canonical_avg_mpjpe':.3,'fused_mpjpe':.1}]
        stats=[{'angle_bin':'0-30','cluster_count':2,'canonical_avg_mpjpe_mean':.3,
                'fused_mpjpe_mean':.1,'mean_gain_mpjpe':.2,'mean_gain_ci95_low':.19,
                'mean_gain_ci95_high':.21,'p_holm':.0001,'rank_biserial':1.}]
        merged=api.merged_angle_rows(view,stats)
        self.assertEqual(merged[0]['ci_low'],.19)
        self.assertEqual(merged[0]['ci_high'],.21)
        stats[0]['fused_mpjpe_mean']=.15
        with self.assertRaises(ValueError): api.merged_angle_rows(view,stats)


if __name__=='__main__': unittest.main()
