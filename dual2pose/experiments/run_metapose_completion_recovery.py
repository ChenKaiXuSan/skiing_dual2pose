"""Separate GPU-UUID recovery queue with immutable prior outputs and paper."""
from __future__ import annotations
import argparse
import copy
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from dual2pose.experiments.resume_metapose import sha, read, verify_bundle, reserve_output


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def execute_steps(root,steps,*,cwd,env):
    root=Path(root);state_path=root/'state.json'
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status='running',started_at=now(),pid=os.getpid(),
               steps=[dict(name=n,command=c,status='pending') for n,c in steps])
    write(state_path,state)
    for i,step in enumerate(state['steps']):
        step.update(status='running',started_at=now());write(state_path,state)
        try:
            with (root/(step['name']+'.log')).open('x') as f:
                p=subprocess.run(step['command'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
            step.update(return_code=p.returncode,finished_at=now())
            if p.returncode:raise RuntimeError(f"{step['name']} exited {p.returncode}")
            step['status']='complete';write(state_path,state)
        except BaseException as e:
            step.update(status='failed',error=repr(e),finished_at=now())
            for dependent in state['steps'][i+1:]:dependent.update(status='blocked',failed_dependency=step['name'])
            state.update(status='failed',finished_at=now());write(state_path,state)
            raise
    state.update(status='complete',finished_at=now());write(state_path,state)


def replace_metapose_rows(rows,replacements):
    if set(replacements)!={'unity','ski'}:raise ValueError('Require both MetaPose replacements')
    counts={d:0 for d in replacements};result=[]
    for row in rows:
        row=copy.deepcopy(row)
        if row['method']=='metapose':
            dataset=row['dataset']
            if dataset not in counts:raise ValueError('Unexpected MetaPose dataset')
            counts[dataset]+=1;row.update(copy.deepcopy(replacements[dataset]))
        result.append(row)
    if any(v!=1 for v in counts.values()):raise ValueError('Require exactly one MetaPose row per dataset')
    return result


def finalize(root):
    import math
    from dual2pose.experiments import workflow_completion as c
    from dual2pose.experiments import run_dependency_workflow as flow
    root=Path(root);ctx=read(root/'context.json');prior=Path(ctx['prior']);ski=Path(ctx['ski'])
    state=read(prior/'state.json');prior_ctx=read(prior/'context.json')
    checks={};superseded={'metapose_unity_train','metapose_unity_score','metapose_ski_train',
                         'metapose_ski_score','metapose_efficiency','final_coverage'}
    for j in read(prior/'manifest.json')['jobs']:
        if j['id'] in superseded:continue
        if state['jobs'][j['id']]['status']!='complete':raise ValueError('Unfinished prior task '+j['id'])
        flow.verify_checkpoint_tokens(j,state['metadata'])
        checks[j['id']]=c.validate(flow.resolve_tokens(j,state['metadata']),prior_ctx)
    checks['metapose_unity_train']=c.validate_metapose_training(root/'unity','unity')
    checks['metapose_efficiency']=c.validate(dict(validation=dict(kind='profile',path=str(root/'efficiency.json'))),prior_ctx)
    replacements={}
    for ds,metric,export in [('unity',root/'unity/evaluation/metrics.json',root/'unity'),
                              ('ski',root/'ski_evaluation/metrics.json',ski/'predictions')]:
        d=read(metric);report=read(export/'report.json')
        if d['report_sha256']!=sha(export/'report.json') or d['native_prediction_sha256']!=report['predictions']:
            raise ValueError('Scored prediction provenance changed')
        for name,h in report['predictions'].items():
            if sha(export/name)!=h:raise ValueError('Prediction checksum mismatch')
        if d['dataset']!=ds or d['evaluation_protocol']!=c.EVALUATION_PROTOCOL:
            raise ValueError('Wrong scoring protocol')
        expected=64440 if ds=='unity' else 30
        if d['metapose']['sample_count']!=expected or any(not math.isfinite(d['metapose'][k]) for k in ('mpjpe','pa_mpjpe','acceleration_error')):
            raise ValueError('Incomplete/nonfinite scoring')
        if ds=='ski':
            if report['training_report_sha256']!=sha(ski/'training/training_report.json') or not report['training_quality_passed']:
                raise ValueError('Ski stable training provenance changed')
            # Independently rescored frozen predictions must reproduce the
            # completed stable repair, rather than the collapsed old Ski run.
            old=read(ski/'predictions/evaluation/metrics.json')
            for k in ('mpjpe','pa_mpjpe','acceleration_error'):
                if abs(d['metapose'][k]-old['metapose'][k])>1e-8:raise ValueError('Stable Ski rescore mismatch')
            checks['metapose_ski_train']=dict(verified_by='ski_training_validation.log',
                training_report_sha256=report['training_report_sha256'],train_only_quality_gate=True)
        checks['metapose_'+ds+'_score']=dict(metrics=str(metric),sha256=sha(metric))
        replacements[ds]=dict(metrics=d['metapose'],source=str(metric),source_sha256=sha(metric),
            prediction_stage='fresh inference from completed training checkpoints' if ds=='unity' else 'independent rescore of stable Ski training',
            recipe='MetaPose adapted; original Unity recipe, frozen validation-selected checkpoints' if ds=='unity' else 'MetaPose adapted; stable Ski recipe, train-only gates, fixed final epoch',
            training_run=str(prior/'metapose/unity') if ds=='unity' else str(ski/'training'))
    old_table=read(Path(prior_ctx['old_root'])/'external/main_comparison.json')
    table=copy.deepcopy(old_table);table['rows']=replace_metapose_rows(table['rows'],replacements)
    table.update(recovery_root=str(root),updated_at=now(),metapose_results_replaced_for=['unity','ski'])
    # Check each retained row's source, not just the table numbers.
    for row in table['rows']:
        if row.get('source_sha256') and sha(row['source'])!=row['source_sha256']:
            raise ValueError('Comparison source changed: '+row['source'])
    protected_bad=c.verify_protection(read(root/'protected_before.json'))
    source_bad=[p for p,h in read(root/'source_hashes.json').items() if sha(root/'source'/p)!=h]
    if protected_bad or source_bad:raise ValueError('Protected/source files changed: '+str(protected_bad+source_bad))
    final=root/'final';reserve_output(final)
    write(final/'main_comparison_native.json',table)
    fields=['dataset','method','mpjpe_mm','pa_mpjpe_mm','acceleration_mm_per_frame2','source','recipe']
    with (final/'main_comparison_mm.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in table['rows']:
            m=r['metrics'];w.writerow(dict(dataset=r['dataset'],method=r['method'],mpjpe_mm=m['mpjpe']*1000,
                pa_mpjpe_mm=m['pa_mpjpe']*1000,acceleration_mm_per_frame2=m['acceleration_error']*1000,
                source=r['source'],recipe=r.get('recipe','unchanged prior run')))
    write(final/'coverage.json',dict(status='complete',all_required_results_available=True,
        validated_prior_tasks=len(checks)-5,validated_result_tasks=len(checks),checks=checks,
        completed_at=now(),protected_mismatches=protected_bad,source_mismatches=source_bad,
        paper_modified=False,old_results_overwritten=False,old_queue_state_preserved=True,
        superseded_tasks=sorted(superseded),prior_failure='GPU failure after training; old failed state retained',
        unity_retrained=False,ski_stable_result_included=True,
        supplemental_prior_root=str(prior),historical_audit=str(Path(prior_ctx['old_root'])/'final_audit_20261029T041144Z'),
        limitations=['single seed/fixed split','MetaPose adapted; dataset-specific training recipes',
                    'GPU timing is method-specific, not a common-input ranking']))
    print(json.dumps(dict(status='complete',final=str(final),validated_result_tasks=len(checks)),indent=2))


def build_and_run(root,prior,ski,gpu_uuid):
    root=Path(root).resolve();prior=Path(prior).resolve();ski=Path(ski).resolve()
    repo=Path(__file__).resolve().parents[2];cfg,_=verify_bundle(prior/'metapose/unity')
    if not gpu_uuid.startswith('GPU-'):raise ValueError('Use a physical GPU UUID')
    reserve_output(root)
    from dual2pose.experiments import workflow_completion as c
    # Original frozen Unity model and evaluator; overlay only this recovery code.
    source=root/'source';source.mkdir()
    for name in ('dual2pose','configs'):
        shutil.copytree(prior/'source'/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('resume_metapose.py','run_metapose_completion_recovery.py'):
        shutil.copy2(repo/'dual2pose/experiments'/name,source/'dual2pose/experiments'/name)
    for name in ('logs','ckpt','.cache','data'):
        if (repo/name).exists():(source/name).symlink_to(repo/name,target_is_directory=True)
    write(root/'source_hashes.json',{str(p.relative_to(source)):sha(p) for folder in ('dual2pose','configs')
                                   for p in (source/folder).rglob('*') if p.is_file()})
    # Keep large arrays under size/mtime protection; hash all paper files and
    # small records. Selected weights/predictions have separate full SHA checks.
    protected=read(prior/'protected_before.json')
    for parent in (prior,ski,repo/'paper/ieee_access_20261027'):
        for p in parent.rglob('*'):
            if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
            st=p.stat();record=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
            if st.st_size<=4*1024**2 or 'paper' in p.parts:record['sha256']=sha(p)
            protected[str(p.resolve())]=record
    write(root/'protected_before.json',protected)
    ctx=dict(root=str(root),prior=str(prior),ski=str(ski),gpu_uuid=gpu_uuid,created_at=now())
    write(root/'context.json',ctx)
    py=sys.executable;tf=str(repo/'.cache/external_baselines/metapose_env/bin/python')
    common=['--train-run',str(prior/'metapose/unity'),'--frozen-runner',str(source/'dual2pose/experiments/run_metapose.py')]
    # verified_fit requires the exact stable-training implementation, therefore
    # run that validation in its own immutable source tree.
    stable_code='import sys; sys.path.insert(0, '+repr(str(ski/'source'))+'); from dual2pose.experiments.run_metapose_stable import verified_fit; verified_fit('+repr(str(ski/'training'))+'); print("PASS: stable training verified")'
    steps=[('gpu_preflight',[tf,'-u','-m','dual2pose.experiments.resume_metapose',*common,'--output-dir',str(root/'preflight'),'--preflight']),
           ('unity_inference',[tf,'-u','-m','dual2pose.experiments.resume_metapose',*common,'--output-dir',str(root/'unity')]),
           ('unity_evaluation',[py,'-u','-m','dual2pose.eval.evaluate_metapose','--run-dir',str(root/'unity'),'--output-dir',str(root/'unity/evaluation'),'--prediction-origin','fresh_training']),
           ('gpu_efficiency',[tf,'-u','-m','dual2pose.eval.profile_external_efficiency','--method','metapose','--method-run-dir',str(root/'unity'),'--prepared-data-dir',cfg['test_dir'],'--source-repo',cfg['metapose_repo'],'--output',str(root/'efficiency.json')]),
           ('ski_training_validation',[py,'-u','-c',stable_code]),
           ('ski_evaluation',[py,'-u','-m','dual2pose.eval.evaluate_metapose','--run-dir',str(ski/'predictions'),'--output-dir',str(root/'ski_evaluation'),'--prediction-origin','fresh_training']),
           ('final_coverage',[py,'-u','-m','dual2pose.experiments.run_metapose_completion_recovery','--action','final','--root',str(root)])]
    write(root/'commands.json',steps)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,PYTHONPATH=str(source),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='8',TF_NUM_INTEROP_THREADS='2')
    execute_steps(root,steps,cwd=source,env=env)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['run','final'],default='run')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prior',type=Path)
    p.add_argument('--ski',type=Path)
    p.add_argument('--gpu-uuid')
    a=p.parse_args()
    if a.action=='final':finalize(a.root)
    else:
        if not all((a.prior,a.ski,a.gpu_uuid)):p.error('run requires prior, ski and gpu-uuid')
        build_and_run(a.root,a.prior,a.ski,a.gpu_uuid)
