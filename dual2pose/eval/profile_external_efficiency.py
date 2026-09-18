"""External-adaptation timing audit; no accuracy selection or artifact overwrite.

Run MetaPose in its isolated TensorFlow environment, the others in dual2pose.
The first archived Unity window is fixed before timing. These heterogeneous
method contracts are intentionally not presented as a common-input ranking.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / 'logs/ivc_mmsports_extension/external_baselines'


def time_calls(call, synchronize, warmup=20, repeats=100):
    if warmup < 1 or repeats < 2:
        raise ValueError('Positive warmup and at least two repeats required')
    for _ in range(warmup):
        call()
        synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        call()
        synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return dict(samples_ms=samples, median_ms=statistics.median(samples),
                mean_ms=statistics.mean(samples), warmup=warmup, repeats=repeats)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def torch_profile(method):
    import numpy as np
    import torch
    from dual2pose.eval.profile_fusion_efficiency import parameter_summary
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    sync = lambda: torch.cuda.synchronize(0)
    if method == 'stride':
        from dual2pose.eval.stride_external import load_official_stride, COMMON
        root = EXTERNAL/'20260910/stride_unity_test_full_seed42'
        old = read(root/'report.json')
        checkpoint = ROOT/'ckpt/external/stride_latest_epoch.bin'
        assert sha(checkpoint) == old['model']['checkpoint_sha256']
        adapter, info = load_official_stride(ROOT/'.cache/external_baselines/STRIDE', checkpoint, 'cuda:0')
        assert info['config_sha256'] == old['model']['config_sha256']
        arr = np.load(root/'inputs17.npy', mmap_mode='r')[:1].copy()
        x = torch.as_tensor(arr, device='cuda:0')
        pred, _ = adapter.refine(x)
        expected = np.load(root/'predictions_common13.npy', mmap_mode='r')[:1]
        delta = float(np.max(np.abs(pred.numpy()[:,:,COMMON,:]-expected)))
        # A reset and all 30 gradient updates are repeated for every timed call.
        details = []
        def run():
            _, timing = adapter.refine(x)
            details.append(timing)
        total = time_calls(run, sync, 3, 10)
        model = adapter.model
        def inference():
            with torch.no_grad():
                out = (model(x)+adapter.flip_fn(model(adapter.flip_fn(x))))*.5
                out[:,:,0,:] = 0
                return out+x[:,:,0:1,:]
        forward = time_calls(inference, sync)
        extra = dict(total_with_ttt=total, forward_only=forward,
                     timed_ttt_details=details[3:], updates=adapter.config.epochs,
                     scope='prepared canonical average17 on GPU; reset, AdamW creation, 30 updates, flip inference and CPU output; forward-only excludes reset/updates/CPU output',
                     checkpoint_sha256=sha(checkpoint), config_sha256=info['config_sha256'])
    else:
        from dual2pose.experiments.run_deciwatch import DeciWatchWindow
        root = EXTERNAL/'20260911/deciwatch_unity_official_protocol_full_seed4321'
        config = read(root/'config.json')
        checkpoint = root/'best.pth'
        assert sha(checkpoint) == read(root/'report.json')['checkpoint_sha256']
        model = DeciWatchWindow(ROOT/'.cache/external_baselines/DeciWatch',39,
                               config['interval'],config['hidden'],config['layers']).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        arr = np.load(root/'prepared/test/input.npy',mmap_mode='r')[:1].copy().reshape(1,30,39)
        x = torch.as_tensor(arr,device='cuda:0')
        def inference():
            with torch.no_grad():
                return model(x)[0]
        prediction = inference().cpu().numpy().reshape(1,30,13,3)
        expected = np.load(root/'recovered.npy',mmap_mode='r')[:1].reshape(1,30,13,3)
        delta = float(np.max(np.abs(prediction-expected)))
        extra = dict(forward_only=time_calls(inference,sync), checkpoint_sha256=sha(checkpoint),
                     scope='prepared canonical average13 on GPU; repeat-last to 31 tokens, official forward, both output heads, crop to 30; excludes input canonicalization/averaging')
    if not np.isfinite(delta) or delta > 2e-3:
        raise ValueError(f'Archived prediction parity failed: {delta}')
    return dict(method=method, **parameter_summary(model), **extra,
                archived_first_window_max_abs_difference=delta, input_shape=list(arr.shape),
                input_slice_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                dtype='float32', framework=torch.__version__, gpu=torch.cuda.get_device_name(0),
                cpu_threads=4, tf32=False)


def metapose_profile():
    import numpy as np
    from dual2pose.experiments.run_metapose import OfficialMetaPose
    from dual2pose.eval.metapose_inputs import normalize_observations, point_mixtures, restore_left_gauge
    root = EXTERNAL/'20260911/metapose_unity_full_seed42'
    data = EXTERNAL/'20260911/metapose_data/unity/test'
    arrays = {name: np.load(data/f'{name}.npy',mmap_mode='r')[:30].copy()
              for name in ('xyz','uv','q','t','native_left')}
    native = (arrays['xyz']-arrays['t'][...,None,:])/arrays['q'][...,None,None]
    uncertainty = read(root/'uncertainty.json')
    model = OfficialMetaPose(ROOT/'.cache/external_baselines/google-research',42)
    tf = model.tf
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Refusing CPU fallback')
    hashes = []
    for i, selection in enumerate(read(root/'selected_stages.json'),1):
        path = root/f'stage_{i}.weights.h5'
        assert sha(path) == selection['checkpoint_sha256']
        model.add_stage(); model.load_stage(path); hashes.append(sha(path))
    x = model.initialize(arrays['xyz'])
    task = point_mixtures(arrays['uv'],uncertainty).reshape(30,-1)
    def full():
        xyz, uv, q, t, _, _ = normalize_observations(native, arrays['uv'])
        state = model.initialize(xyz)
        mixture = point_mixtures(uv, uncertainty).reshape(30,-1)
        state = model.predict(state, mixture)
        projected = model.project(state)[:,0]
        return restore_left_gauge(projected,q[:,0],t[:,0],arrays['native_left'])
    prediction = full()
    expected = np.load(root/'metapose_raw_left.npy',mmap_mode='r')[0]
    delta = float(np.max(np.abs(prediction-expected)))
    if not np.isfinite(delta) or delta > 2e-3:
        raise ValueError(f'MetaPose archived prediction parity failed: {delta}')
    # Public wrappers materialize numpy outputs, synchronizing TF device work.
    forward = time_calls(lambda: model.predict(x,task),lambda: None)
    total = time_calls(full,lambda: None)
    weights = [v for stage in model.stages for v in stage.weights]
    return dict(method='metapose', total_parameters=sum(int(np.prod(v.shape)) for v in weights),
                weight_bytes=sum(int(np.prod(v.shape))*v.dtype.size for v in weights),
                learned_solver=forward, total_with_adapters=total,
                dtype='float64 learned solver; float32 S1 initialization', framework=tf.__version__,
                gpu=tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0]).get('device_name'),
                cpu_threads=dict(intra=8,inter=2), input_shape=[30,2,13,3],
                checkpoint_sha256=hashes, input_slice_sha256=hashlib.sha256(native.tobytes()+arrays['uv'].tobytes()).hexdigest(),
                archived_first_window_max_abs_difference=delta,
                scope='30 independent frames batched; predicted weak projection fit, S1, point mixtures, three-stage solver, left-gauge restoration; CPU/GPU transfers included; normalized 2D box units; excludes RGB front end, disk I/O and final evaluation canonicalization')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['stride','deciwatch','metapose'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result = metapose_profile() if args.method=='metapose' else torch_profile(args.method)
    result.update(status='complete_timing_audit',source_sha256=sha(__file__),
                  selected_input='first archived Unity test window, fixed before timing; no GT input',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  limitation='method-specific adaptation contracts; not common-input speed ranking or full video pipeline')
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('timed_ttt_details',)},indent=2),flush=True)


if __name__=='__main__':
    main()
