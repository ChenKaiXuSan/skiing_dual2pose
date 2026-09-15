"""Full MetaPose export; add --supervision only for train/validation labels."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from dual2pose.eval.metapose_inputs import NativeInputs, normalize_observations, sha256


def prepare(cache_dir, dataset_root, output_dir, supervision=False):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    inp = NativeInputs(cache_dir, dataset_root)
    label_reader = None
    if supervision:
        from dual2pose.eval.metapose_supervision import Supervision2D
        label_reader = Supervision2D(inp)  # refuse test before creating outputs
    output.mkdir(parents=True)
    shapes = dict(xyz=(2,13,3), uv=(2,13,2), q=(2,), t=(2,3), native_left=(13,3))
    if supervision:
        shapes['labels2d'] = (2,13,2)
    arrays = {k: np.lib.format.open_memmap(output/f'{k}.npy', mode='w+', dtype=np.float32,
              shape=(inp.count*30, *shape)) for k, shape in shapes.items()}
    started = time.monotonic()
    for start in range(0, inp.count, 256):
        stop = min(start+256, inp.count)
        raw, uv = inp.batch(start, stop)
        xyz, uv, q, t, origin, side = normalize_observations(raw, uv)
        values = dict(xyz=xyz, uv=uv, q=q, t=t, native_left=raw[:,:,0])
        if label_reader:
            values['labels2d'] = (label_reader.batch(start,stop)-origin[...,None,:])/side[...,None,None]
        for name, value in values.items():
            arrays[name][start*30:stop*30] = value.reshape(-1, *shapes[name])
        progress = dict(completed_windows=stop, total_windows=inp.count, seconds=time.monotonic()-started)
        (output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        if start == 0 or stop == inp.count or stop % 8192 == 0:
            print(json.dumps(progress), flush=True)
    for a in arrays.values():
        a.flush()
    metadata = dict(dataset=inp.dataset, split=inp.split, windows=inp.count, frames=inp.count*30,
                    native_cache=str(inp.cache), native_manifest_sha256=sha256(inp.cache/'manifest.json'),
                    ground_truth_3d_loaded=False, supplied_calibration=False,
                    labels2d_exported=supervision, seconds=time.monotonic()-started,
                    files={f'{k}.npy':sha256(output/f'{k}.npy') for k in arrays},
                    code_sha256={str(Path(p).resolve()):sha256(p) for p in
                        [__file__, 'dual2pose/eval/metapose_inputs.py','dual2pose/eval/metapose_supervision.py']})
    (output/'observation_hashes.json').write_text(json.dumps(inp.observation_hashes, sort_keys=True)+'\n')
    metadata['observations_sha256'] = sha256(output/'observation_hashes.json')
    if label_reader:
        (output/'supervision_hashes.json').write_text(json.dumps(label_reader.hashes, sort_keys=True)+'\n')
        metadata['supervision_sha256'] = sha256(output/'supervision_hashes.json')
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir','dataset-root','output-dir'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--supervision', action='store_true')
    args = p.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))
