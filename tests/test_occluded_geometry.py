import importlib
import tempfile
import unittest
from pathlib import Path
import numpy as np


class MatchedOcclusionTest(unittest.TestCase):
    def frontend(self):
        name = 'dual2pose.eval.run_matched_occlusion_frontend'
        self.assertIsNotNone(importlib.util.find_spec(name), 'matched frontend missing')
        return importlib.import_module(name)

    def geometry(self):
        name = 'dual2pose.eval.evaluate_occluded_geometry'
        self.assertIsNotNone(importlib.util.find_spec(name), 'matched geometry evaluator missing')
        return importlib.import_module(name)

    def test_same_masked_image_yields_paired_predictions_and_exact_ids(self):
        from tests.test_image_occlusion_frontend import _stream_fixture, _joint_loader
        from dual2pose.eval.image_occlusion import ImageOcclusionSetting, OcclusionFrameKey, apply_image_occlusion
        from dual2pose.map_config import filter_unity_kpts, filter_sam3d_body_kpts
        m = self.frontend()
        image = np.arange(200*200*3,dtype=np.uint8).reshape(200,200,3)
        setting = ImageOcclusionSetting('random',.5)
        expected,_ = apply_image_occlusion(image,filter_unity_kpts(_joint_loader(None),flag='2d',gender='female'),
                                           setting,OcclusionFrameKey('female','turn','capture_L0_A000',3),source_frame_ids=[3])
        raw3 = np.arange(70*3,dtype=np.float32).reshape(70,3)
        raw2 = np.arange(70*2,dtype=np.float32).reshape(70,2)+7
        calls=[]
        def predict(img):
            calls.append(img.copy())
            return dict(pred_keypoints_3d=raw3,pred_keypoints_2d=raw2)
        with tempfile.TemporaryDirectory() as tmp:
            p=m.infer_matched_stream(_stream_fixture(),predictor=predict,setting=setting,output_root=Path(tmp),
                                    image_loader=lambda _:image.copy(),joint_loader=_joint_loader)
            with np.load(p) as d:
                np.testing.assert_array_equal(calls[0],expected)
                np.testing.assert_array_equal(d['pose'][0],filter_sam3d_body_kpts(raw3))
                # SAM3D exports full-image pixel coordinates: never rescale/crop twice.
                np.testing.assert_array_equal(d['keypoints_2d'][0],filter_sam3d_body_kpts(raw2))
                self.assertEqual(d['frame_indices'].tolist(),[3])
                self.assertEqual(d['image_size_hw'].tolist(),[[200,200]])
            self.assertTrue(m.validate_matched_stream(p,[3],setting))
            self.assertFalse(m.validate_matched_stream(p,[4],setting))
            self.assertFalse(m.validate_matched_stream(p,[3],ImageOcclusionSetting('random',1.)))

    def test_failed_detection_is_missing_2d_not_origin_pixel(self):
        from tests.test_image_occlusion_frontend import _stream_fixture, _image_loader
        m=self.frontend()
        with tempfile.TemporaryDirectory() as tmp:
            p=m.infer_matched_stream(_stream_fixture(),predictor=lambda _:None,setting=None,output_root=Path(tmp),
                                    image_loader=_image_loader,joint_loader=lambda _:self.fail('clean must not need GT mask joints'))
            with np.load(p) as d:
                self.assertTrue(d['detection_failed'][0])
                self.assertTrue(np.isnan(d['keypoints_2d']).all())
                self.assertFalse(d['valid_2d'].any())
                self.assertFalse(d['pose'].any())

    def test_resume_does_not_accept_legacy_pose_only_cache(self):
        from tests.test_image_occlusion_frontend import _stream_fixture, _image_loader
        from dual2pose.eval.run_unity_image_occlusion_frontend import stream_output_path
        m=self.frontend()
        with tempfile.TemporaryDirectory() as tmp:
            p=stream_output_path(Path(tmp),_stream_fixture());p.parent.mkdir(parents=True)
            np.savez_compressed(p,pose=np.zeros((1,15,3)),frame_indices=[3],detection_failed=[False])
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):
                m.infer_matched_stream(_stream_fixture(),predictor=lambda _:None,setting=None,output_root=Path(tmp),image_loader=_image_loader)
            self.assertEqual(before,p.read_bytes())

    def test_rotation_zero_identity_and_camera_center_fixed(self):
        m=self.geometry()
        k=np.array([[800.,0,320],[0,800,240],[0,0,1]])
        center=np.array([2.,3.,4.]);p=k@np.column_stack([np.eye(3),-center])
        np.testing.assert_array_equal(m.rotate_projection(p,k,0),p)
        rotated=m.rotate_projection(p,k,90)
        np.testing.assert_allclose(rotated@np.r_[center,1],0,atol=1e-10)
        local=np.linalg.solve(k,rotated[:,:3])
        np.testing.assert_allclose(local,np.array([[0,0,1],[0,1,0],[-1,0,0]]),atol=1e-12)

    def test_no_gt_geometry_and_missing_point_not_silently_dropped(self):
        from dual2pose.eval.main_baseline_geometry import project_points
        m=self.geometry()
        p1=np.column_stack([np.eye(3),[0,0,0]])
        p2=np.column_stack([np.eye(3),[-1,0,0]])
        xyz=np.tile(np.array([[[.2,.3,3.],[.5,.2,4.]]]),(4,1,1))
        uv1=project_points(p1,xyz);uv2=project_points(p2,xyz)
        uv1[1,0]=np.nan
        result=m.geometry_predictions(p1,p2,uv1,uv2)
        self.assertEqual(result['dlt'].shape,(4,2,3))
        self.assertFalse(result['dlt_valid'][1,0])
        np.testing.assert_array_equal(result['dlt'][1,0],[0,0,0])
        self.assertTrue(np.isfinite(result['robust_dlt']).all())
        np.testing.assert_allclose(result['dlt'][0],xyz[0],atol=1e-6)
        self.assertEqual(result['counts']['dlt_invalid_points'],1)

    def test_stream_frame_lookup_preserves_repeated_frame_positions(self):
        m=self.geometry()
        d=dict(frame_indices=np.array([2,5]),pose=np.stack([np.full((15,3),2),np.full((15,3),5)]),
               keypoints_2d=np.zeros((2,15,2)),valid_2d=np.ones((2,15),bool),detection_failed=np.zeros(2,bool))
        out=m.select_frames(d,[5,2,5])
        self.assertEqual(out['pose'][:,0,0].tolist(),[5,2,5])
        with self.assertRaises(ValueError):m.select_frames(d,[6])


class AdditionalMatchedOcclusionTest(unittest.TestCase):
    frontend = MatchedOcclusionTest.frontend
    geometry = MatchedOcclusionTest.geometry
    def test_all_missing_is_retained_as_zero_not_successful_reconstruction(self):
        m = self.geometry()
        p = np.column_stack([np.eye(3), [0, 0, 0]])
        uv = np.full((2, 30, 15, 2), np.nan)
        result = m.geometry_predictions(p, p, uv, uv)
        self.assertEqual(result['counts']['dlt_invalid_points'], 900)
        self.assertEqual(result['counts']['dlt_invalid_points_common13'], 780)
        self.assertEqual(result['counts']['robust_invalid_points'], 900)
        for key in ('dlt', 'robust_dlt', 'reprojection_error_px'):
            self.assertTrue(np.isfinite(result[key]).all())
            self.assertFalse(result[key].any())

    def test_same_contract_resume_skips_predictor_changed_contract_rejected(self):
        from tests.test_image_occlusion_frontend import _stream_fixture, _image_loader
        m = self.frontend()
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(setting=None, output_root=Path(tmp), image_loader=_image_loader, run_signature='a')
            path = m.infer_matched_stream(_stream_fixture(), predictor=lambda _: None, **kwargs)
            again = m.infer_matched_stream(_stream_fixture(), predictor=lambda _: self.fail('must resume'), **kwargs)
            self.assertEqual(path, again)
            kwargs['run_signature'] = 'b'
            with self.assertRaises(FileExistsError):
                m.infer_matched_stream(_stream_fixture(), predictor=lambda _: None, **kwargs)

    def test_matrix_has_seven_conditions_and_camera_free_reuse_is_defined(self):
        m = self.geometry()
        self.assertEqual(len(m.EVALUATION_CONDITIONS), 7)
        self.assertEqual(m.ANGLES, (0., 1., 3.))
        self.assertEqual(m.EVALUATION_CONDITIONS[1], ('random_0p5_left', 'random_0p5', 'clean'))
        self.assertEqual(m.EVALUATION_CONDITIONS[-1], ('random_1_both', 'random_1', 'random_1'))


    def test_resume_validator_rejects_incomplete_or_nonfinite_result(self):
        import copy
        m = self.geometry()
        self.assertTrue(hasattr(m, 'validate_condition_result'))
        cells = [dict(angle_degrees=a, method=method, reused_no_camera_result=a != 0 and method in ('canonfuse3d', 'canonical_avg'),
                      metrics=dict(mpjpe=.1, pa_mpjpe=.05, acceleration_error=.01, sample_count=2,
                                   point_count=780, pa_point_count=780, acceleration_point_count=728))
                 for a in m.ANGLES for method in m.METHODS]
        result = dict(status='complete', condition='clean', protocol_sha256='abc', results=cells,
                      detection_coverage=dict(sample_count=2, frame_positions=60),
                      geometry_coverage={str(a):dict(point_count_all15=900) for a in m.ANGLES})
        self.assertTrue(m.validate_condition_result(result, 'abc', 'clean', 2))
        bad = copy.deepcopy(result); bad['results'].pop()
        self.assertFalse(m.validate_condition_result(bad, 'abc', 'clean', 2))
        bad = copy.deepcopy(result); bad['results'][0]['metrics']['mpjpe'] = float('nan')
        self.assertFalse(m.validate_condition_result(bad, 'abc', 'clean', 2))
        self.assertFalse(m.validate_condition_result(result, 'changed', 'clean', 2))


if __name__=='__main__':
    unittest.main()
