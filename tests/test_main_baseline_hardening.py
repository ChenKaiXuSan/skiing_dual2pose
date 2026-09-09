import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dual2pose.eval.main_baseline_data import load_cache, sha256
from dual2pose.models.main_baselines import quality_weighted_fusion

class MainBaselineHardeningTests(unittest.TestCase):
    def make_cache(self,root):
        p=Path(root)
        for name in ('left','right','target'):
            np.save(p/f'{name}.npy',np.ones((2,30,13,3),dtype=np.float32))
        manifest={'sample_count':2,'time_window':30,'num_joints':13,
                  'files':{f'{name}.npy':sha256(p/f'{name}.npy') for name in ('left','right','target')}}
        (p/'manifest.json').write_text(json.dumps(manifest))
        return p,manifest

    def test_rejects_same_shape_cache_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            p,_=self.make_cache(root)
            np.save(p/'left.npy',np.zeros((2,30,13,3),dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                load_cache(p)

    def test_rejects_wrong_joint_shape_even_with_current_digest(self):
        with tempfile.TemporaryDirectory() as root:
            p,m=self.make_cache(root)
            np.save(p/'left.npy',np.zeros((2,30,15,3),dtype=np.float32))
            m['files']['left.npy']=sha256(p/'left.npy'); (p/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(RuntimeError,'shape'):
                load_cache(p)

    def test_collapsed_view_is_not_rewarded_as_stable(self):
        generator=torch.Generator().manual_seed(46)
        right=torch.randn(2,30,13,3,generator=generator)
        left=torch.zeros_like(right)
        torch.testing.assert_close(quality_weighted_fusion(left,right),right)
        torch.testing.assert_close(quality_weighted_fusion(right,left),right)

    def test_resume_refuses_changed_objective_and_microbatch(self):
        from dual2pose.experiments.run_main_baselines import validate_resume_config
        saved={'loss':'L1+.1acc','microbatch_size':2048,'seed':42}
        for key,value in [('loss','L1+1acc'),('microbatch_size',4096)]:
            current={**saved,key:value}
            with self.assertRaisesRegex(RuntimeError,'Resume configuration mismatch'):
                validate_resume_config(saved,current)

if __name__=='__main__': unittest.main()
