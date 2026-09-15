"""Audited native-input STRIDE runner; predictions and evaluation are separate.

Non-test --limit is for execution preflight, not accuracy claims. A test run must
cover the entire archived split. Every output directory must be new.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from dual2pose.eval.main_baseline_data import EXPECTED, batch_slices, sha256, write_json
from dual2pose.eval.main_baseline_geometry import _load_index, _match_source_samples, _resolve_dataset_path
from dual2pose.eval.stride_external import COMMON, canonical_average17, load_official_stride
from dual2pose.map_config import filter_sam3d_body_kpts
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


def _read_native_frame(item):
    directory, frame = item
    path = Path(directory)/f'{frame:06d}_sam3d_body.npz'
    with np.load(path, allow_pickle=True) as archive:
        output = archive['output'].item()  # trusted local SAM3D archive format
        points = np.asarray(filter_sam3d_body_kpts(output['pred_keypoints_3d']),np.float32)
        bones = np.asarray(output['pred_joint_coords'],np.float32)
    if points.shape != (15,3) or bones.shape != (127,3):
        raise ValueError(f'Invalid native prediction shape: {path}')
    if not np.isfinite(points).all() or not np.isfinite(bones).all():
        raise ValueError(f'Nonfinite native prediction: {path}')
    # Hash the exact arrays consumed, not unused image/mesh payloads.
    digest = hashlib.sha256(points.tobytes()+bones.tobytes()).hexdigest()
    return directory, frame, points, bones, digest


def prepare_inputs(cache_dir, dataset_root, limit=0, workers=4):
    """Prepare input-only average17; never opens target.npy or camera metadata."""
    cache, root = Path(cache_dir).resolve(), Path(dataset_root).resolve()
    manifest = json.loads((cache/'manifest.json').read_text())
    dataset, split = manifest['dataset'],manifest['split']
    if dataset not in EXPECTED or split not in EXPECTED[dataset] or limit < 0:
        raise ValueError('Unsupported dataset/split/limit')
    if split == 'test' and limit:
        raise ValueError('Partial test runs are forbidden')
    if manifest['representation'] != 'raw_pre_batch_canonicalization':
        raise ValueError('Wrong cache representation')
    total = manifest['sample_count']; count = min(total,limit) if limit else total
    if split == 'test' and total != EXPECTED[dataset][split]:
        raise ValueError('Incomplete archived test split')
    for name in ('left.npy','right.npy','frames.npy','samples.jsonl'):
        if sha256(cache/name) != manifest['files'][name]:
            raise ValueError(f'Cache checksum mismatch: {name}')
    index = Path(manifest['index'])
    if sha256(index) != manifest['index_sha256']:
        raise ValueError('Source index checksum mismatch')
    samples = [json.loads(row) for row in (cache/'samples.jsonl').read_text().splitlines()]
    if len(samples) != total or [x['index'] for x in samples] != list(range(total)):
        raise ValueError('Cache sample order/count mismatch')
    samples = samples[:count]
    sources = _match_source_samples(dataset,samples,_load_index(index,split))
    frames = np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    native = [np.load(cache/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in ('left','right')]
    joints = 15 if dataset == 'unity' else 13
    if (manifest['num_joints'] != joints or manifest['time_window'] != 30
            or frames.shape != (total,30) or not np.issubdtype(frames.dtype,np.integer)
            or any(a.shape != (total,30,joints,3) or a.dtype != np.float32 for a in native)):
        raise ValueError('Cache shape/dtype mismatch')
    if np.any(frames[:count] < 0) or np.any(np.diff(frames[:count],axis=1) < 0):
        raise ValueError('Invalid frame order')
    directories, needed = [], {}
    for sample,source,fs in zip(samples,sources,frames[:count]):
        pair = []
        for view in (1,2):
            if dataset == 'unity':
                d = root/'sam3d_body_results/inference'/sample['person_id']/sample['action_id']/'frames'/sample[f'cam{view}_id']
            else:
                d = _resolve_dataset_path(source[f'cam{view}_sam3d_kpt3d_dir'],root)
            name = str(d);pair.append(name)
            needed.setdefault(name,set()).update(map(int,fs))
        directories.append(pair)
    jobs = [(d,f) for d,fs in sorted(needed.items()) for f in sorted(fs)]
    predictions, evidence = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(d,f,points,bones,digest) in enumerate(pool.map(_read_native_frame,jobs),1):
            predictions[d,f] = (points,bones)
            evidence[str(Path(d)/f'{f:06d}_sam3d_body.npz')] = digest
            if i % 1000 == 0:
                print(f'Raw input frames {i}/{len(jobs)}',flush=True)
    average = np.empty((count,30,17,3),np.float32)
    batch_size = 256 if dataset == 'unity' else 4
    for start,stop in batch_slices(count,batch_size):
        extra = []
        for view in (0,1):
            batch_bones = []
            for i in range(start,stop):
                sequence = [predictions[directories[i][view],int(f)] for f in frames[i]]
                raw_points = np.stack([x[0] for x in sequence])
                if dataset == 'ski':
                    raw_points = raw_points[:,2:15]
                if not np.allclose(raw_points,native[view][i],rtol=0,atol=1e-6):
                    raise ValueError(f'Raw/cached prediction mismatch at sample {i}, view {view}')
                batch_bones.append(np.stack([x[1] for x in sequence]))
            extra.append(np.stack(batch_bones))
        average[start:stop] = canonical_average17(
            native[0][start:stop],native[1][start:stop],*extra,dataset)
    return average, dict(dataset=dataset,split=split,sample_count=count,
        full_split_count=total,protocol_batch_size=batch_size,cache_dir=str(cache),
        cache_manifest_sha256=sha256(cache/'manifest.json'),source_index_sha256=sha256(index),
        raw_array_sha256=evidence,raw_hash_definition='float32 filtered15 xyz bytes followed by MHR127 xyz bytes',
        frame_ids=frames[:count].tolist(),samples=samples,
        ground_truth_loaded=False,supplied_calibration=False)


def evaluate_predictions(cache, prediction, average, dataset):
    """GT is opened only after all predictions are fixed and saved."""
    from dual2pose.eval.pa_mpjpe import PAMetricAccumulator, PA_PROTOCOL
    manifest=json.loads((cache/'manifest.json').read_text())
    if sha256(cache/'target.npy') != manifest['files']['target.npy']:
        raise ValueError('Target checksum mismatch')
    target=np.load(cache/'target.npy',mmap_mode='r',allow_pickle=False)
    joints=15 if dataset=='unity' else 13
    if target.shape != (len(prediction),30,joints,3):
        raise ValueError('Prediction/target coverage mismatch')
    accumulators = [PAMetricAccumulator(),PAMetricAccumulator()]
    kwargs={} if dataset=='unity' else dict(left_hip=4,right_hip=5,neck=12)
    for start,stop in batch_slices(len(prediction),256 if dataset=='unity' else 4):
        truth=canonicalize_pose_torch(torch.tensor(target[start:stop]),**kwargs)[0]
        if dataset=='unity':
            truth=truth[:,:,2:15]
        for acc,values in zip(accumulators,(prediction,average[:,:,COMMON])):
            acc.update(torch.tensor(values[start:stop]),truth)
    return dict(stride_adapted=accumulators[0].result(),average_control=accumulators[1].result(),pa_protocol=PA_PROTOCOL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stride-repo',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-seconds',type=float,default=86400,
                        help='Hard wall-clock budget for this full run (default 24 hours)')
    args=parser.parse_args()
    if args.max_seconds <= 0 or args.workers < 1:
        parser.error('Positive wall-clock budget and worker count required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.time();report=dict(status='preparing',publication_eligible=False,command=sys.argv,
        method='Avg. + STRIDE (adapted; MHR17; 30-frame TTT)',start_unix=start,
        code_sha256={str(p):sha256(p) for p in (Path(__file__),Path(__file__).parents[1]/'eval/stride_external.py')})
    report_path=args.output_dir/'report.json'
    write_json(report_path,report)
    try:
        torch.set_num_threads(2)
        average,provenance=prepare_inputs(args.cache_dir,args.dataset_root,args.limit,args.workers)
        np.save(args.output_dir/'inputs17.npy',average)
        write_json(args.output_dir/'input_provenance.json',provenance)
        adapter,model_info=load_official_stride(args.stride_repo,args.checkpoint,args.device)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
            report['gpu_name']=torch.cuda.get_device_name(args.device)
        model_info['source_commit']=subprocess.check_output(['git','-C',str(args.stride_repo),'rev-parse','HEAD'],text=True).strip()
        model_info['source_dirty']=bool(subprocess.check_output(['git','-C',str(args.stride_repo),'status','--porcelain'],text=True).strip())
        if model_info['source_dirty']:
            raise ValueError('Official source checkout is modified')
        report.update(status='running',model=model_info,dataset=provenance['dataset'],split=provenance['split'],
            sample_count=len(average),completed_count=0,preparation_seconds=time.time()-start,
            torch_version=torch.__version__,device=args.device)
        report['hard_timeout_seconds']=args.max_seconds
        write_json(report_path,report)
        pred=np.lib.format.open_memmap(args.output_dir/'predictions_common13.npy',mode='w+',
            dtype=np.float32,shape=(len(average),30,13,3));pred[:]=np.nan
        with (args.output_dir/'per_window.jsonl').open('x') as log:
            for i in range(len(average)):
                if time.time()-start > args.max_seconds:
                    raise TimeoutError('Declared full-run wall-clock budget exceeded')
                output,timing=adapter.refine(average[i:i+1])
                pred[i]=output.numpy()[0][:,COMMON,:]
                log.write(json.dumps(dict(index=i,**timing),allow_nan=False)+'\n');log.flush()
                report['completed_count']=i+1
                if i<2 or (i+1)%25==0 or i+1==len(average):
                    pred.flush();report['elapsed_seconds']=time.time()-start
                    write_json(report_path,report)
                    print(f'Windows {i+1}/{len(average)}; TTT {timing["total_seconds"]:.3f}s; elapsed {report["elapsed_seconds"]:.1f}s',flush=True)
        pred.flush()
        if not np.isfinite(pred).all():
            raise FloatingPointError('Final prediction coverage is not finite')
        report.update(prediction_sha256=sha256(args.output_dir/'predictions_common13.npy'),
            elapsed_seconds=time.time()-start,status='preflight_complete')
        if torch.device(args.device).type == 'cuda':
            report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
        if provenance['split']=='test':
            metrics=evaluate_predictions(args.cache_dir,pred,average,provenance['dataset'])
            write_json(args.output_dir/'metrics.json',metrics)
            report.update(status='complete',publication_eligible=True,metrics=metrics)
        write_json(report_path,report)
        print(json.dumps({k:v for k,v in report.items() if k in ('status','completed_count','elapsed_seconds','metrics')},indent=2),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-start)
        write_json(report_path,report)
        raise


if __name__=='__main__':
    main()
