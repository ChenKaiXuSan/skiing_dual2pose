import unittest
import numpy as np
import torch
from dual2pose.eval.evaluate_metapose import canonical_prediction_and_control, evaluate
from dual2pose.eval.metapose_inputs import sha256
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


class MetaPoseEvaluationTest(unittest.TestCase):
    def test_uses_native_left_transform_before_selection(self):
        rng = np.random.default_rng(33)
        for ds,j in [('unity',15),('ski',13)]:
            left = rng.normal(size=(3,30,j,3)).astype(np.float32)
            right = rng.normal(size=(3,30,j,3)).astype(np.float32)
            pred,avg = canonical_prediction_and_control(left[:,:,-13:],left,right,ds)
            kw = {} if ds == 'unity' else dict(left_hip=4,right_hip=5,neck=12)
            lc = canonicalize_pose_torch(torch.tensor(left),**kw)[0][:,:,-13:]
            rc = canonicalize_pose_torch(torch.tensor(right),**kw)[0][:,:,-13:]
            torch.testing.assert_close(pred,lc,rtol=0,atol=0)
            torch.testing.assert_close(avg,(lc+rc)/2,rtol=0,atol=0)

    def test_complete_saved_prediction_evaluation_keeps_partial_batch(self):
        import json
        import tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        import io
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache'; run=root/'run'
            cache.mkdir(); run.mkdir()
            arrays={k:rng.normal(size=(5,30,13,3)).astype(np.float32) for k in ('left','right','target')}
            for name,a in arrays.items(): np.save(cache/f'{name}.npy',a)
            m=dict(dataset='ski',split='test',sample_count=5,
                   files={f'{k}.npy':sha256(cache/f'{k}.npy') for k in arrays})
            (cache/'manifest.json').write_text(json.dumps(m))
            pred_hashes={}
            for name in ('s1','metapose'):
                path=run/f'{name}_raw_left.npy'; np.save(path,arrays['left']); pred_hashes[path.name]=sha256(path)
            report=dict(status='complete_predictions_not_yet_scored',finite_coverage=1.,dataset='ski',windows=5,
                        native_test_cache=str(cache),native_test_manifest_sha256=sha256(cache/'manifest.json'),
                        predictions=pred_hashes)
            (run/'report.json').write_text(json.dumps(report))
            control=PAMetricAccumulator()
            for sl in (slice(0,4),slice(4,5)):
                l,r,t=[canonicalize_pose_torch(torch.tensor(arrays[k][sl]),left_hip=4,right_hip=5,neck=12)[0]
                       for k in ('left','right','target')]
                control.update((l+r)/2,t)
            control_path=root/'control.json'
            control_path.write_text(json.dumps(dict(average_control=control.result())))
            with redirect_stdout(io.StringIO()): result=evaluate(run,control_path)
            self.assertEqual(result['metapose']['sample_count'],5)
            self.assertEqual(result['metapose']['pa_frame_count'],150)
            self.assertEqual(result['metapose']['point_count'],1950)
            self.assertEqual(result['metapose']['acceleration_point_count'],1820)
            self.assertTrue(all(v==0 for v in result['archived_average_control_delta'].values()))
            with self.assertRaises(FileExistsError): evaluate(run,control_path)


if __name__ == '__main__':
    unittest.main()
