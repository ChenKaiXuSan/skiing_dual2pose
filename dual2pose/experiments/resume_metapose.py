"""Export Unity predictions from completed training without reopening its output.

Use the frozen runner from the original run. Incomplete S1/test caches are never
reused, and the selected checkpoints are fixed before any test data is opened.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def reserve_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def verify_bundle(path):
    path=Path(path); cfg=read(path/'config.json')
    if (cfg['stages'],cfg['epochs'],cfg['seed'])!=(3,10,42):
        raise ValueError('Expected completed three-stage Unity training budget')
    if not cfg['no_3d_supervision'] or not cfg['no_supplied_calibration'] or not cfg['val_dir']:
        raise ValueError('Training supervision or validation mismatch')
    history=[json.loads(l) for l in (path/'epochs.jsonl').read_text().splitlines()]
    if [(r['stage'],r['epoch']) for r in history]!=[(s,e) for s in (1,2,3) for e in range(1,11)]:
        raise ValueError('Incomplete training history')
    if any(not math.isfinite(r[k]) for r in history for k in ('train_2d_mse','val_2d_mse')):
        raise ValueError('Nonfinite training history')
    selected=read(path/'selected_stages.json')
    if len(selected)!=3:raise ValueError('Incomplete checkpoint selection')
    for stage,sel in enumerate(selected,1):
        best=min((r for r in history if r['stage']==stage),key=lambda r:r['val_2d_mse'])
        if sel['stage']!=stage or sel['epoch']!=best['epoch']:
            raise ValueError('Wrong validation checkpoint selection')
        if sha(path/f'stage_{stage}.weights.h5')!=sel['checkpoint_sha256']:
            raise ValueError('Changed selected checkpoint')
    return cfg,selected


def load_frozen_runner(path, cfg):
    path=Path(path).resolve()
    # Training config records the exact runner, including the original path.
    expected=[v for k,v in cfg['code_sha256'].items() if Path(k).name=='run_metapose.py']
    if expected!=[sha(path)]:raise ValueError('Frozen runner differs from training')
    spec=importlib.util.spec_from_file_location('frozen_metapose_training',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name,digest in cfg['code_sha256'].items():
        if Path(name).name=='run_metapose.py':continue
        if sha(name)!=digest:raise ValueError('Changed training dependency: '+name)
    revision=subprocess.check_output(['git','-C',cfg['metapose_repo'],'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',cfg['metapose_repo'],'status','--porcelain','--','metapose'],text=True)
    if revision!=cfg['source_commit'] or dirty:raise ValueError('Official MetaPose source changed')
    return module


def export(train_run, output_dir, frozen_runner, preflight=False):
    import numpy as np
    train_run=Path(train_run).resolve();out=Path(output_dir).resolve()
    if out.exists():raise FileExistsError(out)
    cfg,selected=verify_bundle(train_run)
    mp=load_frozen_runner(frozen_runner,cfg)
    model=mp.OfficialMetaPose(Path(cfg['metapose_repo']),cfg['seed'])
    tf=model.tf;tf.config.set_soft_device_placement(False)
    if not tf.config.list_physical_devices('GPU'):raise RuntimeError('GPU required; no CPU fallback')
    reserve_output(out)
    started=time.monotonic();last=[0.]
    def progress(status, **extra):
        record=dict(status=status,elapsed_seconds=time.monotonic()-started,pid=os.getpid(),**extra)
        mp.write_json(out/'progress.json',record)
        if record['elapsed_seconds']-last[0]>=30:
            print(json.dumps(record),flush=True);last[0]=record['elapsed_seconds']
    uncertainty=read(train_run/'uncertainty.json')
    with tf.device('/GPU:0'):
        for stage in range(1,4):
            model.add_stage();model.load_stage(train_run/f'stage_{stage}.weights.h5')
        if preflight:
            # Fixed training input, no GT or test-driven selection. Check
            # partition invariance and strict loading of all selected stages.
            data=mp.load_split(cfg['train_dir'],'train')
            xyz=data['xyz'][:64];uv=data['uv'][:64]
            x=model.initialize(xyz)
            task=mp.point_mixtures(uv,uncertainty).reshape(64,-1)
            whole=model.predict(x,task)
            chunks=np.concatenate([model.predict(x[i:i+16],task[i:i+16]) for i in range(0,64,16)])
            delta=float(np.max(np.abs(whole-chunks)))
            if not np.allclose(whole,chunks,atol=2e-5,rtol=1e-5):
                raise ValueError('Prediction changes with batch partition: '+str(delta))
            mp.write_json(out/'preflight.json',dict(status='passed',frames=64,split='train',
                batch_partition_max_abs_delta=delta,selected_stages=selected,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),frozen_runner_sha256=sha(frozen_runner)))
            return
        for name in ['config.json','epochs.jsonl','selected_stages.json','uncertainty.json',
                     *[f'stage_{s}.weights.h5' for s in (1,2,3)]]:
            shutil.copy2(train_run/name,out/name)
        mp.write_json(out/'recovery.json',dict(original_training_run=str(train_run),
            original_config_sha256=sha(train_run/'config.json'),original_history_sha256=sha(train_run/'epochs.jsonl'),
            frozen_runner=str(Path(frozen_runner).resolve()),frozen_runner_sha256=sha(frozen_runner),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),retrained=False,
            incomplete_test_cache_reused=False))
        test=mp.load_split(cfg['test_dir'],'test')
        if test['manifest']['dataset']!='unity' or test['manifest']['frames']!=1933200:
            raise ValueError('Wrong Unity test cohort')
        mp.initialize_split(model,test,out,'test',progress)
        count=len(test['x']);bs=cfg['batch_size']
        predictions={n:np.lib.format.open_memmap(out/f'{n}_raw_left.npy',mode='w+',dtype=np.float32,
                      shape=(count//30,30,13,3)) for n in ('s1','metapose')}
        infer_started=time.monotonic()
        for start in range(0,count,bs):
            stop=min(start+bs,count);x=test['x'][start:stop]
            task=mp.point_mixtures(test['uv'][start:stop],uncertainty).reshape(stop-start,-1)
            for name,state in [('s1',x),('metapose',model.predict(x,task))]:
                projected=model.project(state)[:,0]
                raw=mp.restore_left_gauge(projected,test['q'][start:stop,0],test['t'][start:stop,0],test['native_left'][start:stop])
                predictions[name].reshape(-1,13,3)[start:stop]=raw
            if start%(bs*100)==0:progress('inference',completed_frames=stop,total_frames=count)
        for value in predictions.values():value.flush();mp.finite(value,'complete predictions')
        report=dict(status='complete_predictions_not_yet_scored',dataset='unity',windows=count//30,
            frames=count,finite_coverage=1.,inference_seconds=time.monotonic()-infer_started,
            total_seconds=time.monotonic()-started,selected_stages=selected,seed=cfg['seed'],
            split_manifests={k:sha(Path(cfg[k+'_dir'])/'manifest.json') for k in ('train','val','test')},
            native_test_cache=test['manifest']['native_cache'],
            native_test_manifest_sha256=test['manifest']['native_manifest_sha256'],
            uncertainty_sha256=sha(out/'uncertainty.json'),
            predictions={f'{k}_raw_left.npy':sha(out/f'{k}_raw_left.npy') for k in predictions},
            recovery=read(out/'recovery.json'))
        # Training selection and input weights must still match on completion.
        verify_bundle(train_run)
        mp.write_json(out/'report.json',report)
        progress('complete_predictions_not_yet_scored',frames=count,windows=count//30)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--frozen-runner',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    export(**vars(p.parse_args()))
