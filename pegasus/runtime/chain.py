"""Prepare-only multi-node NQSV plan; explicitly submit one ready task at a time.

No command submits the full DAG. Receipts are immutable and dependencies are
validated before submission and again inside each allocated job.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
import uuid

OLD='/home/kaixu_chen/skiing/Skiing_Canonical_DualView_3D_Pose_PyTorch'
AUTO='logs/per_frame_common13_v1/auto_20261028T061053Z'
MAIN='logs/per_frame_common13_v1/main_20261028T053836Z'
CORE='logs/per_frame_common13_v1/core_20261028T055152Z'
SUPP='logs/per_frame_common13_v1/completion_20261029T042127Z'


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    # Publish complete JSON atomically, without replacing another writer's file.
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.',suffix='.tmp',delete=False) as f:
        temporary=Path(f.name)
        try:
            json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
            os.link(temporary,path)
        finally:temporary.unlink(missing_ok=True)


def remap(value,mapping):
    if isinstance(value,dict):return {k:remap(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [remap(v,mapping) for v in value]
    if not isinstance(value,str):return value
    # One substitution pass prevents a newly mapped prefix being mapped twice.
    pattern='|'.join(re.escape(k) for k in sorted(mapping,key=len,reverse=True))
    return re.sub('('+pattern+r')(?=/|["\s:}]|$)',lambda m:str(mapping[m[1]]),value)


def dependencies(job,plan_dir):
    metadata={}
    for dep in job.get('deps',[]):
        receipt=Path(plan_dir)/'receipts'/f'{dep}.json'
        if not receipt.is_file():raise RuntimeError(f'Unfinished dependency: {dep}')
        r=read(receipt)
        if r['status']!='complete':raise RuntimeError(f'Failed dependency: {dep}')
        for path,digest in r.get('artifacts',{}).items():
            if sha(path)!=digest:raise RuntimeError(f'Changed dependency artifact: {path}')
        metadata[dep]=r.get('metadata',{})
    # Checkpoint tokens can refer to transitive dependencies.
    for p in (Path(plan_dir)/'receipts').glob('*.json'):
        r=read(p)
        if r['status']=='complete':metadata[p.stem]=r.get('metadata',{})
    return metadata


def prepare(repo,out,runtime):
    repo=Path(repo).resolve();out=Path(out).resolve();runtime=Path(runtime).resolve()
    out.mkdir(parents=True,exist_ok=False)
    root=out/'results';main=out/'main';core=out/'core';supp=out/'supplement'
    py=str(runtime/'envs/torch/bin/python');tf=str(runtime/'envs/metapose/bin/python')
    mapping={OLD+'/'+AUTO:str(root),OLD+'/'+MAIN:str(main),OLD+'/'+CORE:str(core),OLD+'/'+SUPP:str(supp),OLD+'/.cache/external_baselines/metapose_env':str(runtime/'envs/metapose'),'/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python':py,OLD:str(repo),'/home/kaixu_chen/skiing/data':str(repo.parent/(repo.name+'_data')),'/home/kaixu_chen/data/skiing':str(repo.parent/(repo.name+'_data'))}
    jobs=remap(read(repo/AUTO/'manifest.json')['jobs'],mapping)
    by={j['id']:j for j in jobs};launch=read(repo/MAIN/'launch_manifest.json')['jobs']
    for ds in ('unity','ski'):
        j=by['main_'+ds];j.update(commands=[remap(launch[ds]['command'],mapping)],resource='gpu',scientific_training=True)
        # Module entrypoints work from the frozen source directory.
        j['commands'][0][1:2]=['-m','dual2pose.train_unity' if ds=='unity' else 'dual2pose.train_ski_poseptz']
    for j in jobs:
        j.pop('blocked_reason',None);j.pop('missing_inputs',None);j.pop('external_name',None)
        j.setdefault('env',{})
        for k in ('CUDA_VISIBLE_DEVICES','PATH','LD_LIBRARY_PATH'):j['env'].pop(k,None)
        if j['family']=='core':
            j['scientific_training']=j['validation']['kind']!='reference'
            j['deps']=list(set(j['deps']+['external_prepare_cache']))
        elif j['family']=='training':j['scientific_training']=bool(j['validation'].get('training'))
    # Core training must see the same immutable raw cache as all other methods.
    raw=repo/'logs/ivc_mmsports_extension/main_baselines/20260906/cache'
    for ds in ('unity','ski'):
        for cmd in by['external_stride_'+ds]['commands']:
            cmd[cmd.index('--dataset-root')+1]=str(repo.parent/(repo.name+'_data')/('skiing_unity_dataset' if ds=='unity' else 'Ski-PosePTZ-CameraDataset-png'))
    by['core_ski_reference']['commands'][0][-1]='{ski_checkpoint}'
    # Replace archived MetaPose reuse with the two approved current recipes.
    data=repo/'logs/ivc_mmsports_extension/external_baselines/20260911/metapose_data'
    source=repo/'.cache/external_baselines/google-research'
    prior={j['id']:j for j in remap(read(repo/SUPP/'manifest.json')['jobs'],mapping)}
    unity=prior['metapose_unity_train'];unity['validation']['path']=str(root/'metapose/unity')
    unity['commands']=remap(unity['commands'],{str(supp/'metapose/unity'):str(root/'metapose/unity')})
    unity.update(scientific_training=True,deps=['external_prepare_metapose_env']);jobs.append(unity)
    ski_fit=root/'metapose/ski_train';ski_pred=root/'metapose/ski_predictions'
    jobs.append(dict(id='metapose_ski_train',family='cluster',resource='gpu',deps=['external_prepare_metapose_env'],scientific_training=True,env={},commands=[[tf,'-m','dual2pose.experiments.run_metapose_stable','train','--train-dir',str(data/'ski/train'),'--output-dir',str(ski_fit),'--metapose-repo',str(source),'--require-gpu']],validation={'kind':'stable_train','path':str(ski_fit)}))
    jobs.append(dict(id='metapose_ski_export',family='cluster',resource='gpu',deps=['metapose_ski_train'],env={},commands=[[tf,'-m','dual2pose.experiments.run_metapose_stable','export','--train-run',str(ski_fit),'--test-dir',str(data/'ski/test'),'--output-dir',str(ski_pred)]],validation={'kind':'stable_export','path':str(ski_pred)}))
    for ds,native,dep in [('unity',root/'metapose/unity','metapose_unity_train'),('ski',ski_pred,'metapose_ski_export')]:
        j=by['external_metapose_'+ds];j['deps']=[dep];j['validation']['native']=str(native)
        j['commands']=[[py,'-m','dual2pose.eval.evaluate_metapose','--run-dir',str(native),'--output-dir',str(root/'external/metapose'/ds),'--prediction-origin','fresh_training']]
    j=by['external_profile_metapose'];cmd=j['commands'][0];cmd[cmd.index('--method-run-dir')+1]=str(root/'metapose/unity')
    by['external_prepare_metapose_env']['commands']=[[tf,'-m','dual2pose.experiments.workflow_external','--context',str(root/'context.json'),'--action','environment']]
    for key in ('ski_gate','pocket_shift','realworld','gate_figures'):
        j=prior[key];j['deps']={'ski_gate':['main_ski'],'pocket_shift':['main_unity','main_ski'],'realworld':['main_unity','main_ski','pocket_forward_unity_seed0'],'gate_figures':['ski_gate','eval_gate_interventions']}[key];jobs.append(j)
    # gen_S has a hard 24-hour request limit. Split only independent STRIDE
    # windows; preserve native input preparation and full-cohort scoring.
    from pegasus.runtime.stride_shards import ranges
    prepared=root/'stride_inputs/unity';shards=root/'stride_shards/unity'
    def stride_command(action,*args):return [py,'-m','pegasus.runtime.stride_shards',action,*map(str,args)]
    jobs.append(dict(id='stride_unity_prepare',family='cluster',resource='cpu',deps=['external_prepare_cache'],env={},commands=[stride_command('prepare','--cache',raw/'unity/test','--dataset-root',repo.parent/(repo.name+'_data')/'skiing_unity_dataset','--output',prepared)],validation={'kind':'stride_prepared','path':str(prepared)}))
    shard_ids=[]
    for index,(start,end) in enumerate(ranges(64440,2048)):
        key=f'stride_unity_shard_{index:02d}';shard_ids.append(key);dest=shards/f'{index:02d}'
        jobs.append(dict(id=key,family='cluster',resource='gpu',deps=['stride_unity_prepare'],env={},commands=[stride_command('shard','--prepared',prepared,'--start',start,'--end',end,'--stride-repo',repo/'.cache/external_baselines/STRIDE','--checkpoint',repo/'ckpt/external/stride_latest_epoch.bin','--output',dest)],validation={'kind':'stride_shard','path':str(dest),'start':start,'end':end}))
    merged=by['external_stride_unity'];merged['deps']=shard_ids;merged['resource']='cpu';merged['commands']=[stride_command('merge','--prepared',prepared,'--shards-root',shards,'--cache',raw/'unity/test','--output',root/'external/stride/unity')]
    # One final receipt accounts for all experiments, not historic recovery jobs.
    jobs.append(dict(id='final_coverage',family='cluster',resource='cpu',deps=[j['id'] for j in jobs],env={},commands=[[py,'-m','pegasus.runtime.chain','summarize','--plan',str(out)]],validation={'kind':'final','path':str(out/'final/coverage.json')}))
    from dual2pose.experiments.run_dependency_workflow import validate_graph
    validate_graph(jobs)
    assert sum(bool(j.get('scientific_training')) for j in jobs)==38
    for d in (root,main,core,supp,out/'pbs',out/'receipts',out/'tasks',out/'submissions'):d.mkdir(parents=True,exist_ok=True)
    (core/'cache').symlink_to(raw,target_is_directory=True)
    ctx=dict(repo=str(repo),root=str(root),main_root=str(main),core_root=str(core),raw_cache=str(raw),python=py)
    write_new(root/'context.json',ctx)
    write_new(supp/'context.json',{**ctx,'root':str(supp),'old_root':str(root),'opencv_runtime':str(runtime/'envs/torch/lib/python3.11/site-packages')})
    plan=dict(protocol='common13_direct_per_frame_v1',account='HP260146',queue='gen_S',repo=str(repo),runtime=str(runtime),root=str(out),context=ctx,training_configurations=38,submitted_jobs=[],jobs=jobs,created=time.time(),status='prepared_not_submitted')
    write_new(out/'plan.json',plan)
    frozen=out/'source';frozen.mkdir()
    code_root=Path(__file__).resolve().parents[2]
    for directory in ('dual2pose','configs','pegasus'):
        shutil.copytree(code_root/directory,frozen/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('logs','ckpt','.cache','data','paper'):
        if (repo/name).exists():(frozen/name).symlink_to(repo/name,target_is_directory=True)
    write_new(out/'source_hashes.json',{str(p.relative_to(frozen)):sha(p) for d in ('dual2pose','configs','pegasus') for p in (frozen/d).rglob('*') if p.is_file()})
    for j in jobs:
        wall='24:00:00'
        script=['#!/bin/bash','#PBS -A HP260146','#PBS -q gen_S','#PBS -N cf_'+j['id'][:10],'#PBS -l elapstim_req='+wall,'set -euo pipefail','export DUAL2POSE_REPO='+shlex.quote(str(repo)),'source '+shlex.quote(str(frozen/'pegasus/runtime/env.sh')),'export PYTHONPATH='+shlex.quote(str(frozen)),'cd '+shlex.quote(str(frozen)), 'exec "$DUAL2POSE_PYTHON" -m pegasus.runtime.chain run-one --plan '+shlex.quote(str(out))+' --job '+shlex.quote(j['id'])]
        (out/'pbs'/f'{j["id"]}.pbs').write_text('\n'.join(script)+'\n')
    return plan


def run_one(plan_dir,job_id):
    plan_dir=Path(plan_dir).resolve();plan=read(plan_dir/'plan.json');job=next(j for j in plan['jobs'] if j['id']==job_id)
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Execution requires an allocated NQSV job')
    meta=dependencies(job,plan_dir)
    for name,digest in read(plan_dir/'source_hashes.json').items():
        if sha(plan_dir/'source'/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    task=plan_dir/'tasks'/job_id;task.mkdir(exist_ok=False)
    from dual2pose.experiments import run_dependency_workflow as flow
    flow.verify_checkpoint_tokens(job,meta);job=flow.resolve_tokens(job,meta)
    env=os.environ.copy();env.update(job.get('env',{}));site=Path(plan['runtime'])/'envs/metapose/lib/python3.11/site-packages'
    # Only TF subprocesses get the CUDA 12.2 libraries; PyTorch keeps its cu126 set.
    started=time.time()
    try:
        if job['id'] in ('pocket_shift','realworld'):
            ctx=read(plan_dir/'supplement/context.json')
            mainmeta={k:read(plan_dir/'receipts'/f'{k}.json')['metadata'] for k in ('main_unity','main_ski')}
            # All completion jobs write the same checkpoint-only view atomically.
            tmp=plan_dir/'results'/f'state.{job_id}.tmp';tmp.write_text(json.dumps({'metadata':mainmeta}));tmp.replace(plan_dir/'results/state.json')
        with (task/'run.log').open('x') as log:
            for command in job['commands']:
                cmd_env=env.copy()
                if command[0]==str(Path(plan['runtime'])/'envs/metapose/bin/python'):
                    cmd_env['LD_LIBRARY_PATH']=':'.join(str(p) for p in sorted((site/'nvidia').glob('*/lib')))
                    cmd_env['XLA_FLAGS']='--xla_gpu_cuda_data_dir='+str(site/'nvidia/cuda_nvcc')
                subprocess.run(command,cwd=plan_dir/'source',env=cmd_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if job['family']=='main':
            ds=job['validation']['dataset'];shutil.copy2(task/'run.log',plan_dir/'main'/f'{ds}.log');write_new(plan_dir/'main'/f'{ds}_status.json',{'return_code':0})
        md=validate_job(job,plan,plan_dir)
        artifacts=md.get('artifacts',{}).copy()
        if md.get('checkpoint') and md.get('checkpoint_sha256'):artifacts[md['checkpoint']]=md['checkpoint_sha256']
        write_new(plan_dir/'receipts'/f'{job_id}.json',dict(status='complete',metadata=md,artifacts=artifacts,started=started,finished=time.time(),pbs_job_id=os.environ['PBS_JOBID']))
    except Exception as exc:
        write_new(plan_dir/'receipts'/f'{job_id}.json',dict(status='failed',error=repr(exc),started=started,finished=time.time()));raise


def validate_job(job,plan,plan_dir):
    from dual2pose.experiments import run_dependency_workflow as flow
    ctx=plan['context']
    if job['family']=='main':return flow.verify_main(job,ctx)
    if job['family']=='core':return flow.verify_core(job,ctx)
    if job['family']=='cluster':return validate_cluster(job,plan)
    if job['family']=='completion':return flow.family_module('completion').validate(job,read(Path(plan_dir)/'supplement/context.json'))
    metadata=flow.family_module(job['family']).validate(job,ctx)
    if job['id'].startswith(('eval_mask_','eval_offset_')):
        metadata.setdefault('artifacts',{}).update(alpha_artifacts(Path(job['env']['EVAL_OUTPUT_ROOT'])))
    return metadata


def alpha_artifacts(directory):
    # The corrected 52 exports are required even when new training changes
    # their values; historic rerun-equality checks do not apply to new weights.
    from dual2pose.map_config import FILTERED_KPTS_MAPPING
    files=list(Path(directory).rglob('alpha_summary_*.csv'))
    if len(files)!=1:raise ValueError('Missing or ambiguous alpha export')
    with files[0].open() as f:rows=list(csv.DictReader(f))
    if ([r['joint_name'] for r in rows]!=list(FILTERED_KPTS_MAPPING.values()) or
            [int(r['joint_index']) for r in rows]!=list(range(len(rows)))):
        raise ValueError('Incorrect alpha joint labels or order')
    for row in rows:
        mean,std,right=(float(row[k]) for k in ('alpha_mean','alpha_std','right_mean'))
        if not all(math.isfinite(x) for x in (mean,std,right)) or not (0<=mean<=1 and 0<=right<=1 and std>=0 and math.isclose(mean+right,1,abs_tol=1e-6)):
            raise ValueError('Invalid alpha statistics')
    for pattern in ('alpha_joint_bar_*.png','alpha_time_joint_heatmap_*.png'):
        figures=list(files[0].parent.glob(pattern))
        if len(figures)!=1 or figures[0].stat().st_size==0:raise ValueError('Missing alpha figure')
        files.extend(figures)
    return {str(p):sha(p) for p in files}


def validate_cluster(job,plan):
    v=job['validation'];p=Path(v['path'])
    if v['kind'] in ('stride_prepared','stride_shard'):
        import numpy as np
        r=read(p/'report.json')
        if v['kind']=='stride_prepared':
            assert r['status']=='prepared' and r['sample_count']==64440
            assert sha(p/'inputs17.npy')==r['input_sha256']
            assert sha(p/'input_provenance.json')==r['provenance_sha256']
            files=[p/'report.json',p/'inputs17.npy',p/'input_provenance.json']
        else:
            assert r['status']=='complete_shard' and r['start']==v['start'] and r['end']==v['end'] and r['full_count']==64440
            assert sha(p/'predictions_common13.npy')==r['prediction_sha256']
            a=np.load(p/'predictions_common13.npy',mmap_mode='r')
            assert a.shape==(v['end']-v['start'],30,13,3) and np.isfinite(a).all()
            files=[p/'report.json',p/'predictions_common13.npy']
    elif v['kind']=='stable_train':
        from dual2pose.experiments.run_metapose_stable import verified_fit
        verified_fit(p)
        files=[p/'config.json',p/'training_report.json',*p.glob('stage_*.weights.h5')]
        files=[f for f in files if f.exists()]
    elif v['kind']=='stable_export':
        r=read(p/'report.json');assert r['windows']==30 and r['finite_coverage']==1
        files=[p/'report.json']+[p/name for name,digest in r['predictions'].items() if sha(p/name)==digest]
        assert len(files)==1+len(r['predictions'])
    else:
        assert read(p)['all_required_results_available'];files=[p]
    return {'artifacts':{str(f):sha(f) for f in files}}


def summarize(plan_dir):
    from dual2pose.experiments import run_dependency_workflow as flow
    plan_dir=Path(plan_dir);plan=read(plan_dir/'plan.json');job=next(j for j in plan['jobs'] if j['id']=='final_coverage');metadata=dependencies(job,plan_dir)
    # Some archived validators return only checkpoint hashes. Revalidate every
    # required result/history/selection file before claiming complete coverage.
    for candidate in plan['jobs']:
        if candidate['id'] in job['deps']:validate_job(flow.resolve_tokens(candidate,metadata),plan,plan_dir)
    rows=read(plan_dir/'results/external/main_comparison.json')['rows'];dest=plan_dir/'final';dest.mkdir(exist_ok=False)
    with (dest/'main_comparison_mm.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','method','mpjpe_mm','pa_mpjpe_mm','acceleration_mm_per_frame2']);w.writeheader()
        for r in rows:
            m=r['metrics'];w.writerow(dict(dataset=r['dataset'],method=r['method'],mpjpe_mm=m['mpjpe']*1000,pa_mpjpe_mm=m['pa_mpjpe']*1000,acceleration_mm_per_frame2=m['acceleration_error']*1000))
    write_new(dest/'coverage.json',{'all_required_results_available':True,'validated_jobs':len(job['deps']),'training_configurations':38,'paper_modified':False})


def submit_job(plan_dir,job_id):
    plan_dir=Path(plan_dir);folder=plan_dir/'submissions'/job_id
    folder.mkdir(parents=True,exist_ok=False)
    # An unexpected interruption keeps the reservation: acceptance is unknown
    # and must be reconciled with qstat rather than risking duplicate execution.
    result=subprocess.run(['qsub','-A','HP260146','-q','gen_S','-o',str(folder/'stdout.log'),'-e',str(folder/'stderr.log'),str(plan_dir/'pbs'/f'{job_id}.pbs')],capture_output=True,text=True)
    write_new(folder/'submission.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:
        failures=plan_dir/'submission_failures';failures.mkdir(exist_ok=True)
        folder.rename(failures/(job_id+'.'+uuid.uuid4().hex))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','ready','submit-one','run-one','summarize']);p.add_argument('--plan',required=True,type=Path);p.add_argument('--job');p.add_argument('--repo',type=Path,default=Path(os.environ.get('DUAL2POSE_REPO',Path.cwd())));p.add_argument('--runtime',type=Path,default=Path(os.environ.get('DUAL2POSE_RUNTIME','/nonexistent')));a=p.parse_args()
    if a.action=='prepare':
        plan=prepare(a.repo,a.plan,a.runtime);print(json.dumps({'jobs':len(plan['jobs']),'training_configurations':38,'submitted_jobs':[]}));return
    if a.action=='run-one':run_one(a.plan,a.job);return
    if a.action=='summarize':summarize(a.plan);return
    plan=read(a.plan/'plan.json');ready=[]
    for j in plan['jobs']:
        if (a.plan/'receipts'/f'{j["id"]}.json').exists() or (a.plan/'submissions'/j['id']).exists():continue
        try:dependencies(j,a.plan)
        except RuntimeError:continue
        ready.append(j['id'])
    if a.action=='ready':print(json.dumps(ready));return
    if a.job not in ready:raise RuntimeError('Job is not ready, or already submitted: '+str(a.job))
    result=submit_job(a.plan,a.job);print(result.stdout+result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
