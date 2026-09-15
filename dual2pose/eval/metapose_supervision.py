"""Separately guarded non-test 2D annotations; never used by inference."""
from functools import lru_cache
import numpy as np
from dual2pose.eval.metapose_inputs import sha256, finite


def require_supervised_split(split):
    if split not in ('train', 'val'):
        raise ValueError('2D supervision is restricted to train/val; never test')


class Supervision2D:
    def __init__(self, native_inputs):
        require_supervised_split(native_inputs.split)
        self.inputs = native_inputs
        self.hashes = {}

    @lru_cache(maxsize=400000)
    def unity_frame(self, path, gender):
        from dual2pose.map_config import filter_unity_kpts
        self.hashes[str(path)] = sha256(path)
        return finite(filter_unity_kpts(np.load(path, allow_pickle=False), '2d', gender)[2:15, :2], str(path))

    @lru_cache(maxsize=2)
    def ski_labels(self, path):
        import h5py
        from dual2pose.map_config import H36M_17_TO_COMMON_13_INDICES
        # Intentionally never access 3D/cam_intrinsic/cam_position/R_cam_2_world.
        with h5py.File(path, 'r') as f:
            ids = np.stack([np.asarray(f[k], dtype=np.int64) for k in ('subj', 'seq', 'cam', 'frame')], axis=-1)
            labels = np.asarray(f['2D']).reshape(-1, 17, 2)[:, H36M_17_TO_COMMON_13_INDICES] * 256
        keys = [tuple(x) for x in ids]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate Ski 2D annotation identity')
        self.hashes[str(path)] = sha256(path)
        return dict(zip(keys, finite(labels, 'training 2D annotations')))

    def batch(self, start, stop):
        from dual2pose.eval.main_baseline_geometry import _resolve_dataset_path
        inp = self.inputs
        require_supervised_split(inp.split)
        out = np.empty((stop-start, 30, 2, 13, 2), dtype=np.float32)
        for i in range(start, stop):
            source = inp.sources[i]
            if inp.dataset == 'ski':
                path = _resolve_dataset_path(str(source['labels_h5']), inp.root)
                labels = self.ski_labels(path)
            for view in range(2):
                if inp.dataset == 'unity':
                    paths = inp.directory(str(source[f'cam{view+1}_kpt2d_dir']))
                    out[i-start, :, view] = [self.unity_frame(paths[int(f)], str(source['person_id'])) for f in inp.frames[i]]
                else:
                    key = (int(source['subject_id']), int(source['sequence_id']), int(source[f'cam{view+1}_id']))
                    out[i-start, :, view] = [labels[(*key, int(f))] for f in inp.frames[i]]
        return finite(out, '2D supervision')
