"""Input-contract tests; synthetic caches deliberately contain no GT file."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dual2pose.eval.external_baseline_inputs import load_estimated_2d, load_pose_probe
from dual2pose.eval.main_baseline_data import sha256
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


class ExternalBaselineInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.rng = np.random.default_rng(42)

    def tearDown(self):
        self.temp.cleanup()

    def make_cache(self, dataset='unity', split='val'):
        joints = 15 if dataset == 'unity' else 13
        self.left = self.rng.normal(size=(4, 30, joints, 3)).astype('float32')
        self.right = self.left * 2 + np.array([1, -2, 3], dtype='float32')
        records = []
        sources = []
        self.frames = np.tile(np.arange(10, 40), (4, 1))
        for i in range(4):
            if dataset == 'unity':
                record = dict(index=i, person_id='female', action_id=f'action{i}',
                              cam1_id='left', cam2_id='right')
                source = {**record, 'sam3d_cam1_kpt2d_dir': str(self.root / 'left'),
                          'sam3d_cam2_kpt2d_dir': str(self.root / 'right')}
            else:
                record = dict(index=i, subj=0, seq=i, cam1=0, cam2=1)
                source = dict(subject_id=0, sequence_id=i, cam1_id=0, cam2_id=1,
                              cam1_sam3d_kpt2d_dir=str(self.root / 'left'),
                              cam2_sam3d_kpt2d_dir=str(self.root / 'right'))
            records.append(record)
            sources.append(source)
        self.index = self.root / 'index.json'
        self.index.write_text(json.dumps({split: sources}))
        for name, array in [('left', self.left), ('right', self.right), ('frames', self.frames)]:
            np.save(self.cache / f'{name}.npy', array)
        (self.cache / 'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        self.manifest = dict(dataset=dataset, split=split, sample_count=4, num_joints=joints,
                             time_window=30, representation='raw_pre_batch_canonicalization',
                             index=str(self.index), index_sha256=sha256(self.index),
                             files={n: sha256(self.cache / n) for n in
                                    ['left.npy', 'right.npy', 'frames.npy', 'samples.jsonl']})
        self.save_manifest()

    def save_manifest(self):
        (self.cache / 'manifest.json').write_text(json.dumps(self.manifest))

    def write_2d(self, dataset):
        for view_id, view in enumerate(['left', 'right']):
            (self.root / view).mkdir()
            for frame in range(10, 40):
                points = np.stack([np.arange(70)+frame, np.arange(70)+100*view_id], axis=-1)
                if dataset == 'unity':
                    np.save(self.root / view / f'kpt2d_{frame:04d}.npy', points)
                else:
                    np.savez(self.root / view / f'{frame:06d}_sam3d_body.npz',
                             output={'pred_keypoints_2d': points})

    def test_reads_no_gt_and_preserves_raw_views_frames(self):
        self.make_cache()
        result = load_pose_probe(self.cache, count=2)
        self.assertFalse((self.cache / 'target.npy').exists())
        self.assertEqual(result.raw_views.shape, (2, 30, 2, 13, 3))
        np.testing.assert_array_equal(result.raw_views[:, :, 0], self.left[:2, :, 2:])
        np.testing.assert_array_equal(result.frame_ids, self.frames[:2])
        self.assertEqual([s['index'] for s in result.samples], [0, 1])

    def test_native_canonicalization_precedes_common_joint_selection(self):
        for dataset in ['unity', 'ski']:
            with self.subTest(dataset=dataset):
                self.make_cache(dataset, 'train')
                result = load_pose_probe(self.cache, count=2)
                kwargs = {} if dataset == 'unity' else dict(left_hip=4, right_hip=5, neck=12)
                indices = list(range(2, 15)) if dataset == 'unity' else list(range(13))
                for view, original in enumerate([self.left, self.right]):
                    expected = canonicalize_pose_torch(torch.tensor(original), **kwargs)[0]
                    np.testing.assert_array_equal(result.canonical_views[:, :, view],
                                                  expected.numpy()[:2, :, indices])
                np.testing.assert_allclose(result.average, result.canonical_views.mean(axis=2))

    def test_smoke_refuses_test_split_and_unbounded_count(self):
        self.make_cache(split='test')
        with self.assertRaisesRegex(ValueError, 'train|val|test'):
            load_pose_probe(self.cache)
        self.make_cache()
        for count in [0, 5]:
            with self.assertRaises(ValueError):
                load_pose_probe(self.cache, count=count)

    def test_checksum_mismatch_is_not_silently_accepted(self):
        self.make_cache()
        np.save(self.cache / 'left.npy', self.left + 1)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_pose_probe(self.cache)

    def test_invalid_pose_shape_and_nonfinite_values_are_rejected(self):
        for invalid in [np.zeros((4, 29, 15, 3), dtype='float32'),
                        np.full((4, 30, 15, 3), np.nan, dtype='float32')]:
            self.make_cache()
            np.save(self.cache / 'left.npy', invalid)
            self.manifest['files']['left.npy'] = sha256(self.cache / 'left.npy')
            self.save_manifest()
            with self.assertRaises(ValueError):
                load_pose_probe(self.cache)

    def test_sample_order_mismatch_is_rejected(self):
        self.make_cache()
        path = self.cache / 'samples.jsonl'
        path.write_text('\n'.join(reversed(path.read_text().splitlines()))+'\n')
        self.manifest['files']['samples.jsonl'] = sha256(path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'order|index'):
            load_pose_probe(self.cache)

    def test_estimated_2d_matches_common13_without_calibration(self):
        self.make_cache()
        self.write_2d('unity')
        points, provenance = load_estimated_2d(load_pose_probe(self.cache), self.root)
        self.assertEqual(points.shape, (2, 30, 2, 13, 2))
        # Native SAM3D indices: shoulder L=5, shoulder R=6, neck=69.
        np.testing.assert_array_equal(points[0, 0, 0, [0, 1, 12]], [[15, 5], [16, 6], [79, 69]])
        np.testing.assert_array_equal(points[0, 0, 1, 0], [15, 105])
        self.assertEqual(provenance['coordinate_space'], 'image_pixels')
        self.assertFalse(provenance['supplied_calibration'])

    def test_ski_2d_matches_frames_without_labels_h5(self):
        self.make_cache('ski', 'train')
        self.write_2d('ski')
        points, _ = load_estimated_2d(load_pose_probe(self.cache), self.root)
        np.testing.assert_array_equal(points[0, -1, 0, 12], [108, 69])

    def test_changed_source_index_is_rejected(self):
        self.make_cache()
        probe = load_pose_probe(self.cache)
        self.index.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_estimated_2d(probe, self.root)

    def test_invalid_frame_order_is_rejected(self):
        self.make_cache()
        np.save(self.cache / 'frames.npy', self.frames[:, ::-1])
        self.manifest['files']['frames.npy'] = sha256(self.cache / 'frames.npy')
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Frame'):
            load_pose_probe(self.cache)

    def test_missing_estimated_frame_is_not_filled_from_gt(self):
        self.make_cache()
        (self.root / 'left').mkdir()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing estimated 2D'):
            load_estimated_2d(load_pose_probe(self.cache), self.root)

    def test_input_only_smoke_is_not_a_publication_result(self):
        from dual2pose.experiments.smoke_external_baselines import run_probe
        self.make_cache()
        self.write_2d('unity')
        output = self.root / 'probe'
        report = run_probe(self.cache, self.root, output)
        self.assertFalse(report['publication_eligible'])
        self.assertEqual(report['components']['stride']['status'], 'not_requested')
        self.assertEqual(report['components']['metapose']['status'], 'not_requested')
        self.assertNotIn('mpjpe', report)
        with np.load(output / 'inputs.npz', allow_pickle=False) as inputs:
            self.assertEqual(set(inputs.files), {'raw_views', 'canonical_views',
                                                'average', 'frame_ids', 'estimated_2d_px'})
        self.assertEqual(json.loads((output / 'report.json').read_text()), report)

    def test_smoke_refuses_existing_output_directory(self):
        from dual2pose.experiments.smoke_external_baselines import run_probe
        output = self.root / 'probe'
        output.mkdir()
        marker = output / 'user_file.txt'
        marker.write_text('preserve me')
        with self.assertRaises(FileExistsError):
            run_probe(self.cache, self.root, output)
        self.assertEqual(marker.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
