"""Independent, non-overwriting completion of the IEEE Access experiment coverage."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
from dual2pose.experiments import run_dependency_workflow as flow
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL


def read(path):
    return json.loads(Path(path).read_text())


def protection_snapshot(paths):
    result = {}
    for p in paths:
        p = Path(p).resolve(); st = p.stat()
        result[str(p)] = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
                          'sha256': flow.sha(p)}
    return result


def verify_protection(snapshot):
    bad = []
    for name, value in snapshot.items():
        p = Path(name)
        if not p.is_file():
            bad.append(name); continue
        st = p.stat()
        if (st.st_size != value['size'] or st.st_mtime_ns != value['mtime_ns'] or
                ('sha256' in value and flow.sha(p) != value['sha256'])):
            bad.append(name)
    return bad


def shared_shift_mask(ids, reliable, anchors, shifts):
    """Intersection of reference validity on fixed input frames, across all lags."""
    mask = np.ones((*ids.shape, 13), dtype=bool)
    for shift in shifts:
        shifted = ids + shift
        inside = (shifted >= 0) & (shifted < len(reliable))
        rel = reliable[np.clip(shifted, 0, len(reliable)-1)]
        mask &= inside[..., None] & rel[..., anchors].all(-1)[..., None] & rel[..., 2:]
    return mask


def qualitative_window(record):
    starts = record['window_starts']['test']; lo, hi = record['blocks']['test']
    if not starts or any(not lo <= s or s+30 > hi for s in starts):
        raise ValueError('Qualitative windows must stay inside the held-out test block')
    return starts[len(starts)//2]


def validate_metapose_training(path, dataset):
    path = Path(path); config = read(path/'config.json'); report = read(path/'report.json')
    history = [json.loads(s) for s in (path/'epochs.jsonl').read_text().splitlines()]
    budget = 10 if dataset == 'unity' else 300
    if config['stages'] != 3 or config['epochs'] != budget or config['seed'] != 42:
        raise ValueError('Wrong MetaPose training budget')
    if not config['no_3d_supervision'] or not config['no_supplied_calibration']:
        raise ValueError('MetaPose supervision contamination')
    if [(r['stage'], r['epoch']) for r in history] != [(s,e) for s in (1,2,3) for e in range(1,budget+1)]:
        raise ValueError('Incomplete MetaPose training history')
    if report['status'] != 'complete_predictions_not_yet_scored' or report['dataset'] != dataset:
        raise ValueError('Incomplete MetaPose prediction export')
    count = 64440 if dataset == 'unity' else 30
    if report['windows'] != count or report['frames'] != count*30 or report['finite_coverage'] != 1:
        raise ValueError('Incomplete MetaPose test cohort')
    if len(report['selected_stages']) != 3:
        raise ValueError('Missing selected stage')
    for stage, selected in enumerate(report['selected_stages'],1):
        rows = [r for r in history if r['stage'] == stage]
        if any(not math.isfinite(r['train_2d_mse']) or (dataset=='unity' and not math.isfinite(r['val_2d_mse'])) for r in rows):
            raise ValueError('Nonfinite MetaPose loss')
        expected = min(rows,key=lambda r:r['val_2d_mse'])['epoch'] if dataset == 'unity' else budget
        if selected['epoch'] != expected or selected['checkpoint_sha256'] != flow.sha(path/f'stage_{stage}.weights.h5'):
            raise ValueError('Wrong MetaPose selected checkpoint')
    for name, digest in report['predictions'].items():
        p = path/name
        if flow.sha(p) != digest: raise ValueError('MetaPose prediction hash mismatch')
        x = np.load(p, mmap_mode='r')
        if x.shape != (count,30,13,3): raise ValueError('Wrong MetaPose prediction shape')
        for start in range(0,count,1024):
            if not np.isfinite(x[start:start+1024]).all(): raise ValueError('Nonfinite MetaPose prediction')
    return report


def build_jobs(ctx):
    root, repo, old = (Path(ctx[k]) for k in ('root','repo','old_root'))
    python = ctx['python']; tf = str(repo/'.cache/external_baselines/metapose_env/bin/python')
    data = repo/'logs/ivc_mmsports_extension/external_baselines/20260911/metapose_data'
    source = repo/'.cache/external_baselines/google-research'
    jobs = []
    def add(key, commands, kind, deps=(), resource='gpu', priority=0, **spec):
        jobs.append(dict(id=key, commands=commands, family='completion', deps=list(deps),
                         resource=resource, priority=priority, env={}, validation=dict(kind=kind, **spec)))
    def internal(action):
        return [python,'-m','dual2pose.experiments.workflow_completion','--action',action,'--root',str(root)]
    for ds in ('unity','ski'):
        out = root/'metapose'/ds
        cmd = [tf,'-u','-m','dual2pose.experiments.run_metapose','--metapose-repo',str(source),
               '--train-dir',str(data/ds/'train'),'--test-dir',str(data/ds/'test'),
               '--output-dir',str(out),'--stages','3','--epochs','10' if ds=='unity' else '300',
               '--batch-size','1024' if ds=='unity' else '128','--seed','42','--timeout-hours','48']
        if ds=='unity':cmd += ['--val-dir',str(data/ds/'val')]
        add('metapose_'+ds+'_train',[cmd],'metapose_train',priority=100 if ds=='unity' else 90,path=str(out),dataset=ds)
        add('metapose_'+ds+'_score',[[python,'-m','dual2pose.eval.evaluate_metapose',
            '--run-dir',str(out),'--output-dir',str(out/'evaluation'),'--prediction-origin','fresh_training']],'metapose_score',
            deps=['metapose_'+ds+'_train'],resource='cpu',path=str(out/'evaluation'),dataset=ds)
    for original in read(old/'manifest.json')['jobs']:
        if not original['id'].startswith(('eval_mask_','eval_offset_')):continue
        job = json.loads(json.dumps(original).replace(str(old),str(root)))
        job['deps'] = []
        job['validation'] = dict(kind='alpha', original=job['validation'], old_job=original)
        job['family'] = 'completion'
        jobs.append(job)
    add('ski_gate',[[python,'-m','dual2pose.eval.evaluate_gate_interventions','--cache-root',ctx['raw_cache'],
        '--dataset','ski','--split','test','--output',str(root/'ski_gate'),
        '--checkpoint','{ski_checkpoint}','--device','cuda:0']], 'ski_gate',priority=80,path=str(root/'ski_gate'))
    add('pocket_shift',[internal('pocket-shift')],'pocket_shift',priority=75,path=str(root/'pocket_shift'))
    add('realworld',[internal('realworld')],'realworld',priority=70,path=str(root/'realworld'))
    add('gate_figures',[internal('gate-figures')],'gate_figures',deps=['ski_gate'],resource='cpu',path=str(root/'gate_figures'))
    add('metapose_efficiency',[[tf,'-m','dual2pose.eval.profile_external_efficiency','--method','metapose',
        '--method-run-dir',str(root/'metapose/unity'),'--prepared-data-dir',str(data/'unity/test'),
        '--source-repo',str(source),'--output',str(root/'metapose/efficiency.json')]],
        'profile',deps=['metapose_unity_score'],resource='exclusive_gpu',path=str(root/'metapose/efficiency.json'))
    add('final_coverage',[internal('final')],'final',deps=[j['id'] for j in jobs],resource='cpu',path=str(root/'final'))
    flow.validate_graph(jobs)
    return jobs


def validate(job, ctx):
    from dual2pose.experiments import workflow_evaluations as ev
    from dual2pose.experiments import workflow_external as ext
    v=job['validation']; kind=v['kind']; root=Path(ctx['root']); artifacts=[]
    if kind=='alpha':
        wrapped={**job,'validation':v['original']}; md=ev.validate(wrapped,ctx)
        from dual2pose.map_config import FILTERED_KPTS_MAPPING
        directory=Path(job['env']['EVAL_OUTPUT_ROOT'])
        files=list(directory.rglob('alpha_summary_*.csv'))
        if len(files)!=1:raise ValueError('Missing alpha export')
        rows=list(csv.DictReader(files[0].open()))
        if [r['joint_name'] for r in rows]!=list(FILTERED_KPTS_MAPPING.values()):raise ValueError('Wrong joint labels')
        olddir=Path(v['old_job']['env']['EVAL_OUTPUT_ROOT'])
        oldfile=next(olddir.rglob('alpha_summary_*.csv')); oldrows=list(csv.DictReader(oldfile.open()))
        for new,old in zip(rows,oldrows):
            if new['joint_index']!=old['joint_index']:raise ValueError('Changed alpha tensor order')
            for key in ('alpha_mean','alpha_std','right_mean'):
                if not math.isclose(float(new[key]),float(old[key]),abs_tol=1e-6,rel_tol=1e-5):raise ValueError('Alpha values unexpectedly changed')
        # Primary metrics remain identical up to GPU numerical precision.
        new_summary=next(directory.glob(Path(v['original']['path']).name))
        old_summary=next(olddir.glob(Path(v['old_job']['validation']['path']).name))
        a=list(csv.DictReader(new_summary.open())); b=list(csv.DictReader(old_summary.open()))
        if len(a)!=len(b):raise ValueError('Changed scoring coverage')
        for x,y in zip(a,b):
            if not math.isclose(float(x['fused_mpjpe']),float(y['fused_mpjpe']),abs_tol=1e-6,rel_tol=1e-5):raise ValueError('Changed primary MPJPE')
        artifacts += [files[0],*files[0].parent.glob('*.png')]
        md['artifacts'].update({str(p):flow.sha(p) for p in artifacts});return md
    path=Path(v['path'])
    if kind=='metapose_train':
        validate_metapose_training(path,v['dataset'])
        artifacts=[path/'report.json',path/'config.json',path/'epochs.jsonl',*path.glob('stage_*.weights.h5')]
    elif kind=='metapose_score':
        d=read(path/'metrics.json'); ext.current(d); ext.check_metrics(d['metapose'],v['dataset'])
        report=read(path.parent/'report.json')
        if d.get('prediction_origin')!='fresh_training' or d['native_predictions_reused']:
            raise ValueError('Expected newly trained MetaPose predictions')
        if d['native_prediction_sha256']!=report['predictions']:raise ValueError('Scoring used wrong predictions')
        # The scorer reads saved predictions; fresh training provenance is a separate fact.
        artifacts=[path/'metrics.json']
    elif kind=='ski_gate':
        cfg=read(path/'config.json');pr=read(path/'progress.json');ev._protocol(cfg)
        if cfg['dataset']!='ski' or cfg['checkpoint_sha256']!=flow.sha(cfg['checkpoint']):raise ValueError('Ski checkpoint mismatch')
        if pr['status']!='complete_full_test' or pr['samples']!=30 or tuple(pr['conditions'])!=ev.GATE_CONDITIONS:raise ValueError('Missing Ski conditions')
        expected={f'{g}_{r}' for g in ('dynamic','mean','left','right') for r in ('residual','no_residual')}|{'canonical_average'}
        for condition in ev.GATE_CONDITIONS:
            d=read(path/(condition+'.json'));ev._protocol(d)
            if set(d['metrics'])!=expected:raise ValueError('Missing Ski interventions')
            for m in d['metrics'].values():ext.check_metrics(m['common13'],'ski')
            if d['gate_diagnostics']['joint_frame_count']!=30*30*13:raise ValueError('Incomplete Ski diagnostics')
            artifacts.append(path/(condition+'.json'))
        artifacts += [path/'config.json',path/'progress.json']
    elif kind=='profile':
        return ext.validate({**job,'validation':dict(kind='profile',path=str(path),method='metapose')},ctx)
    elif kind in ('pocket_shift','realworld','gate_figures'):
        d=read(path/'manifest.json')
        if d['evaluation_protocol']!=EVALUATION_PROTOCOL:raise ValueError('Wrong supplement protocol')
        for name,digest in d['files'].items():
            p=path/name
            if flow.sha(p)!=digest:raise ValueError('Supplement artifact mismatch')
            artifacts.append(p)
        if kind=='pocket_shift':
            if d['shifts']!=list(range(-4,5)) or d['common_point_count']<=0:raise ValueError('Missing shifts/common points')
            rows=list(csv.DictReader((path/'metrics_mm.csv').open()))
            if len(rows)!=9*len(d['methods']) or any(int(r['point_count'])!=d['common_point_count'] for r in rows):raise ValueError('Shift cohort changed')
            if any(not math.isfinite(float(r['mpjpe_mm'])) for r in rows):raise ValueError('Nonfinite shift scores')
        if kind=='realworld':
            if [r['run'] for r in d['rows']]!=['pro_2','run_4','chen_4']:raise ValueError('Missing real videos')
            x=np.load(path/'poses_mm.npz')
            if x['poses'].shape!=(3,4,30,13,3) or not np.isfinite(x['poses']).all():raise ValueError('Bad qualitative poses')
            record=read(Path(ctx['old_root'])/'pocket/pocket_forward_unity_seed0/summary.json')
            if d['rows'][2]['window_start']!=qualitative_window(record):raise ValueError('Qualitative selection mismatch')
        artifacts.append(path/'manifest.json')
    elif kind=='final':
        d=read(path/'coverage.json')
        if d['protected_mismatches'] or not d['all_prior_jobs_complete']:raise ValueError('Incomplete final coverage')
        artifacts=[path/'coverage.json',path/'main_comparison_mm.csv']
    else:raise ValueError(kind)
    return {'artifacts':{str(p):flow.sha(p) for p in artifacts},'validated_kind':kind}


def build(root, old):
    root,old=Path(root).resolve(),Path(old).resolve()
    if (root/'manifest.json').exists():raise FileExistsError(root/'manifest.json')
    ctx=read(old/'context.json');ctx.update(root=str(root),old_root=str(old),python=sys.executable,opencv_runtime=str(root/'opencv_runtime'))
    root.mkdir(parents=True,exist_ok=True)
    jobs=build_jobs(ctx)
    state=read(old/'state.json')
    if state['status']!='complete':raise ValueError('Prior run is not complete')
    meta={key:state['metadata'][key] for key in ('main_unity','main_ski')}
    for key,record in meta.items():
        if flow.sha(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('Upstream weights changed')
    source=root/'source';source.mkdir()
    repo=Path(ctx['repo'])
    for name in ('dual2pose','configs'):
        shutil.copytree(repo/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('logs','ckpt','.cache','data'):
        if (repo/name).exists():(source/name).symlink_to(repo/name,target_is_directory=True)
    for name,value in [('context.json',ctx),('manifest.json',{'created_at':flow.now(),'jobs':jobs}),
                       ('state.json',{'jobs':{j['id']:{'status':'pending'} for j in jobs},'metadata':meta}),
                       ('source_hashes.json',{str(p.relative_to(source)):flow.sha(p) for d in ('dual2pose','configs') for p in (source/d).rglob('*') if p.is_file()})]:
        flow.atomic_json(root/name,value)
    print(json.dumps({'root':str(root),'jobs':len(jobs)},indent=2))


def retrained_row(row, metrics_path):
    path = Path(metrics_path).resolve()
    return {**row, 'metrics': read(path)['metapose'], 'source': str(path),
            'source_sha256': flow.sha(path),
            'prediction_stage': 'fresh_training_and_full_inference_this_completion_run'}


def finalize(ctx):
    root,old=Path(ctx['root']),Path(ctx['old_root']);out=root/'final';out.mkdir(exist_ok=False)
    state=read(root/'state.json');prior=[v for k,v in state['jobs'].items() if k!='final_coverage']
    if not all(v['status']=='complete' for v in prior):raise ValueError('Unfinished dependencies')
    # Re-run every validator, then independently verify hashes and prior files.
    checks={}
    for j in read(root/'manifest.json')['jobs']:
        if j['id']=='final_coverage':continue
        flow.verify_checkpoint_tokens(j,state['metadata'])
        checks[j['id']]=validate(flow.resolve_tokens(j,state['metadata']),ctx)
    bad=verify_protection(read(root/'protected_before.json'))
    source_bad=[p for p,h in read(root/'source_hashes.json').items() if flow.sha(root/'source'/p)!=h]
    if bad or source_bad:raise ValueError(f'Protected/source files changed: {bad+source_bad}')
    table=read(old/'external/main_comparison.json')
    table['rows'] = [retrained_row(row,root/'metapose'/row['dataset']/'evaluation/metrics.json')
                     if row['method']=='metapose' else row for row in table['rows']]
    flow.atomic_json(out/'main_comparison_native.json',table)
    with (out/'main_comparison_mm.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','method','mpjpe_mm','pa_mpjpe_mm','acceleration_mm_per_frame2']);w.writeheader()
        for r in table['rows']:
            m=r['metrics'];w.writerow(dict(dataset=r['dataset'],method=r['method'],mpjpe_mm=m['mpjpe']*1000,pa_mpjpe_mm=m['pa_mpjpe']*1000,acceleration_mm_per_frame2=m['acceleration_error']*1000))
    flow.atomic_json(out/'coverage.json',dict(all_prior_jobs_complete=True,validated_tasks=len(checks),checks=checks,
        protected_mismatches=bad,source_mismatches=source_bad,completed_at=flow.now(),
        original_audit=str(old/'final_audit_20261029T041144Z/manuscript_consistency_audit.json'),
        paper_modified=False,metapose='new three-stage training and full test inference for both datasets',
        supplemental_groups=['52 corrected alpha exports','Ski gate 5x9','Pocket reference shifts 9 on fixed points','three real videos','gate figures and regional summaries','MetaPose efficiency with new weights'],
        remaining_manuscript_work='Integrate new figures/numbers and update protocol descriptions only when requested.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',required=True,choices=['build','pocket-shift','realworld','gate-figures','final'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--old-root',type=Path);a=p.parse_args()
    if a.action=='build':build(a.root,a.old_root);return
    ctx=read(a.root/'context.json')
    if a.action=='final':finalize(ctx)
    else:
        from dual2pose.experiments import completion_artifacts as artifacts
        {'pocket-shift':artifacts.pocket_shift,'realworld':artifacts.realworld,'gate-figures':artifacts.gate_figures}[a.action](ctx)

if __name__=='__main__':main()
