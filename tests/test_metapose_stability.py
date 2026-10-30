import unittest
import numpy as np
from dual2pose.eval import metapose_stability as health


class MetaPoseStabilityTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.random.default_rng(71).normal(0, .2, (100,2,13,3))
        self.scales = np.ones((100,2))

    def test_healthy_rigid_translation_passes(self):
        report = health.geometry_health(self.ref + [4,-3,8], self.ref, self.scales)
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['radius_ratio_median'], 1.)

    def test_finite_collapsed_pose_is_rejected(self):
        report = health.geometry_health(self.ref * 1e-5, self.ref, self.scales * 1e-6)
        self.assertFalse(report['passed'])
        self.assertEqual(report['bad_frame_fraction'], 1.)

    def test_localized_collapse_cannot_hide_behind_median(self):
        pred = self.ref.copy(); pred[:2] = 0
        self.assertFalse(health.geometry_health(pred,self.ref,self.scales)['passed'])

    def test_nonfinite_and_empty_reference_rejected(self):
        for pred in (self.ref*np.nan, self.ref[:0]):
            with self.assertRaises(ValueError):
                health.geometry_health(pred,self.ref,self.scales)
        with self.assertRaises(ValueError):
            health.geometry_health(self.ref,self.ref*0,self.scales)

    def test_training_loss_rebound_fails_even_with_finite_geometry(self):
        good = health.geometry_health(self.ref,self.ref,self.scales)
        verdict = health.training_gate(good, .08, .04, .0004)
        self.assertFalse(verdict['passed'])
        self.assertTrue(health.training_gate(good,.001,.04,.0004)['passed'])

    def test_nonfinite_loss_cannot_pass(self):
        good = health.geometry_health(self.ref,self.ref,self.scales)
        with self.assertRaises(ValueError):
            health.training_gate(good,float('nan'),.04,.0004)

class ExportGateTests(unittest.TestCase):
    def test_pilot_and_failed_fits_cannot_be_exported(self):
        from dual2pose.experiments.run_metapose_stable import require_exportable
        base = dict(status='complete_train_quality_passed', frames=4350, stages=3,
                    epochs=300, pilot=False, quality=[{'passed':True}]*3)
        require_exportable(base)
        for patch in ({'pilot':True}, {'status':'failed'}, {'epochs':20},
                      {'quality':[{'passed':True},{'passed':False},{'passed':True}]}):
            with self.assertRaises(ValueError): require_exportable({**base,**patch})

    def test_train_parser_does_not_accept_test_data(self):
        from dual2pose.experiments.run_metapose_stable import parser
        with self.assertRaises(SystemExit):
            parser().parse_args(['train','--train-dir','train','--output-dir','out',
                                 '--metapose-repo','src','--test-dir','secret'])

class SavedFitVerificationTests(unittest.TestCase):
    def setUp(self):
        import tempfile,json,pathlib
        from dual2pose.eval.metapose_inputs import sha256
        from dual2pose.experiments import run_metapose_stable as runner
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=pathlib.Path(self.temp.name);self.json=json;self.sha=sha256;self.runner=runner
        train_manifest=self.root/'manifest.json';train_manifest.write_text('{}')
        cfg=dict(epochs=300,pilot=False,max_frames=None,batch_size=128,seed=42,
                 identity_stage_initialization=True,learning_rate=1e-5,gradient_clip_norm=1.,policy=health.POLICY,
                 code_sha256={str(p):sha256(p) for p in [pathlib.Path(runner.__file__),
                     pathlib.Path(runner.__file__).with_name('run_metapose.py'),pathlib.Path(health.__file__)]})
        self.write('config.json',cfg);self.write('uncertainty.json',{})
        np.save(self.root/'training_frame_indices.npy',np.arange(4350))
        shape=np.random.default_rng(3).normal(size=(4350,2,13,3))
        geometry=health.geometry_health(shape,shape,np.ones((4350,2)))
        gate=health.training_gate(geometry,.001,.04,.001)
        rows=[dict(stage=s,epoch=e,quality=gate,full_train_2d_mse=.001) for s in (1,2,3) for e in range(1,301)]
        (self.root/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        selected=[]
        for s in (1,2,3):
            p=self.root/f'stage_{s}.weights.h5';p.write_bytes(b'fixture checkpoint')
            selected.append(dict(stage=s,epoch=300,checkpoint_sha256=sha256(p)))
        report=dict(status='complete_train_quality_passed',dataset='ski',frames=4350,stages=3,epochs=300,
                    pilot=False,quality=[gate]*3,selected_stages=selected,training_manifest_path=str(train_manifest),
                    training_manifest_sha256=sha256(train_manifest),test_data_opened=False,
                    config_sha256=sha256(self.root/'config.json'),uncertainty_sha256=sha256(self.root/'uncertainty.json'),
                    history_sha256=sha256(self.root/'epochs.jsonl'),
                    frame_indices_sha256=sha256(self.root/'training_frame_indices.npy'))
        self.write('training_report.json',report)

    def write(self,name,value):
        (self.root/name).write_text(self.json.dumps(value))

    def test_complete_artifacts_pass_and_missing_checkpoint_fails(self):
        self.runner.verified_fit(self.root)
        (self.root/'stage_2.weights.h5').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.runner.verified_fit(self.root)

    def test_report_quality_cannot_disagree_with_history(self):
        report=self.json.loads((self.root/'training_report.json').read_text())
        report['quality'][2]['full_train_2d_mse']=.08
        self.write('training_report.json',report)
        with self.assertRaises(ValueError):self.runner.verified_fit(self.root)

    def test_export_rejects_changed_adapter_code_provenance(self):
        cfg=self.json.loads((self.root/'config.json').read_text())
        cfg['code_sha256'][str(self.runner.__file__)]='0'*64
        self.write('config.json',cfg)
        report=self.json.loads((self.root/'training_report.json').read_text())
        report['config_sha256']=self.sha(self.root/'config.json');self.write('training_report.json',report)
        with self.assertRaises(ValueError):self.runner.verified_fit(self.root)

    def test_export_rejects_partial_training_indices(self):
        np.save(self.root/'training_frame_indices.npy',np.arange(128))
        with self.assertRaises(ValueError):self.runner.verified_fit(self.root)

class RepairResourceTests(unittest.TestCase):
    def setUp(self):
        self.jobs=[dict(id='metapose_unity_train',resource='gpu'),dict(id='short_eval',resource='gpu'),
                   dict(id='efficiency',resource='exclusive_gpu')]
        self.states={'metapose_unity_train':dict(status='running',gpu=0),
                     'short_eval':dict(status='complete'),'efficiency':dict(status='pending')}
        self.inventory={0:dict(processes=[1],free_mb=47000),1:dict(processes=[],free_mb=48000)}

    def test_waits_for_queued_work_even_when_gpu_temporarily_empty(self):
        from dual2pose.experiments.run_metapose_repair import resource_ready
        self.states['short_eval']['status']='pending'
        self.assertFalse(resource_ready(self.jobs,self.states,self.inventory))

    def test_idle_gpu_one_is_safe_while_unity_is_on_zero(self):
        from dual2pose.experiments.run_metapose_repair import resource_ready
        self.assertTrue(resource_ready(self.jobs,self.states,self.inventory))
        self.inventory[1]['processes']=[999]
        self.assertFalse(resource_ready(self.jobs,self.states,self.inventory))

    def test_does_not_race_exclusive_profile_or_unity_on_gpu_one(self):
        from dual2pose.experiments.run_metapose_repair import resource_ready
        self.states['metapose_unity_train']['gpu']=1
        self.assertFalse(resource_ready(self.jobs,self.states,self.inventory))
        self.states['metapose_unity_train']['status']='complete'
        self.assertFalse(resource_ready(self.jobs,self.states,self.inventory))
        self.states['efficiency']['status']='complete'
        self.assertTrue(resource_ready(self.jobs,self.states,self.inventory))
