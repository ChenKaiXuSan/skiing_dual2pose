"""Bounded external-method diagnostics, never a full experimental rerun."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['deciwatch','stride','metapose']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(os.environ['DUAL2POSE_REPO']);py=os.environ['DUAL2POSE_PYTHON'];tf=os.environ['DUAL2POSE_TF_PYTHON']
    raw=repo/'logs/ivc_mmsports_extension/main_baselines/20260906/cache'
    env=os.environ.copy();commands=[]
    if a.method=='deciwatch':
        commands=[[py,'-m','dual2pose.experiments.run_deciwatch','--deciwatch-repo',str(repo/'.cache/external_baselines/DeciWatch'),'--train-cache',str(raw/'ski/train'),'--test-cache',str(raw/'ski/test'),'--output-dir',str(out/'fit'),'--device','cuda:0','--epochs','1','--batch-size','32']]
    elif a.method=='stride':
        commands=[[py,'-m','dual2pose.experiments.run_stride_external','--cache-dir',str(raw/'ski/train'),'--dataset-root',os.environ['DUAL2POSE_SKI_ROOT'],'--output-dir',str(out/'tta'),'--stride-repo',str(repo/'.cache/external_baselines/STRIDE'),'--checkpoint',str(repo/'ckpt/external/stride_latest_epoch.bin'),'--device','cuda:0','--limit','1','--workers','2','--max-seconds','900']]
    else:
        site=Path(tf).parent.parent/'lib/python3.11/site-packages'
        env['LD_LIBRARY_PATH']=':'.join([str(x) for x in sorted((site/'nvidia').glob('*/lib'))]+[env.get('LD_LIBRARY_PATH','')]);env['XLA_FLAGS']='--xla_gpu_cuda_data_dir='+str(site/'nvidia/cuda_nvcc')
        source=repo/'logs/per_frame_common13_v1/metapose_stable_20261029T052359Z/predictions'
        commands=[[tf,str(repo/'pegasus/runtime/metapose_smoke.py'),str(out/'train_check')],[py,'-m','dual2pose.eval.evaluate_metapose','--run-dir',str(source),'--output-dir',str(out/'archived_score'),'--prediction-origin','archived']]
    state=dict(method=a.method,status='running',publication_eligible=False,started=time.time(),steps=[])
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'status.json')
    save()
    for i,cmd in enumerate(commands):
        step=dict(command=cmd,status='running');state['steps'].append(step);save()
        try:
            with (out/f'{i}.log').open('x') as log:
                run=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            step.update(returncode=run.returncode,status='passed' if run.returncode==0 else 'failed')
        except subprocess.TimeoutExpired:
            step.update(status='failed',error='bounded diagnostic exceeded 1800 seconds')
        save()
        if step['status']!='passed':break
    state.update(status='passed' if all(x['status']=='passed' for x in state['steps']) else 'failed',finished=time.time());save()
    raise SystemExit(0 if state['status']=='passed' else 1)


if __name__=='__main__': main()
